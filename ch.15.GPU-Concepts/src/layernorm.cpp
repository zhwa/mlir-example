#include "common.h"

#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

#include "llvm/Support/TargetSelect.h"

using namespace mlir;

// ============================================================================
// LayerNorm: output[i] = (input[i] - mean) / sqrt(variance + epsilon)
// ============================================================================
//
// Algorithm (3-pass):
// Pass 1: Compute mean = sum(input) / N
// Pass 2: Compute variance = sum((input - mean)²) / N
// Pass 3: Normalize each element
//
// This is the operation that caused the LLVM 20 JIT bug (21 failed attempts!)
// With AOT compilation, it should work perfectly.
//

void buildLayerNormKernel(OpBuilder& builder, Location loc,
                          Value input, Value output, Value N, Value epsilon) {
    // Constants
    Value c0 = ch15::createIndex(builder, loc, 0);
    Value c1 = ch15::createIndex(builder, loc, 1);
    Value c256 = ch15::createIndex(builder, loc, 256);
    Value c255 = ch15::createIndex(builder, loc, 255);
    Value zero_f = ch15::createFloat(builder, loc, 0.0f);

    // Convert N (index) to float for division
    Value N_float = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), N);
    Value N_f32 = builder.create<arith::SIToFPOp>(loc, builder.getF32Type(), N_float);

    // ========================================================================
    // Pass 1: Compute mean
    // ========================================================================

    auto sumLoop = builder.create<scf::ForOp>(loc, c0, N, c1, ValueRange{zero_f});
    builder.setInsertionPointToStart(sumLoop.getBody());

    Value i = sumLoop.getInductionVar();
    Value currentSum = sumLoop.getRegionIterArgs()[0];

    Value val = builder.create<memref::LoadOp>(loc, input, i);
    Value newSum = builder.create<arith::AddFOp>(loc, currentSum, val);
    builder.create<scf::YieldOp>(loc, newSum);

    builder.setInsertionPointAfter(sumLoop);
    Value sumVal = sumLoop.getResult(0);
    Value mean = builder.create<arith::DivFOp>(loc, sumVal, N_f32);

    // ========================================================================
    // Pass 2: Compute variance
    // ========================================================================

    auto varLoop = builder.create<scf::ForOp>(loc, c0, N, c1, ValueRange{zero_f});
    builder.setInsertionPointToStart(varLoop.getBody());

    Value j = varLoop.getInductionVar();
    Value currentVarSum = varLoop.getRegionIterArgs()[0];

    Value val2 = builder.create<memref::LoadOp>(loc, input, j);
    Value diff = builder.create<arith::SubFOp>(loc, val2, mean);
    Value diffSq = builder.create<arith::MulFOp>(loc, diff, diff);
    Value newVarSum = builder.create<arith::AddFOp>(loc, currentVarSum, diffSq);
    builder.create<scf::YieldOp>(loc, newVarSum);

    builder.setInsertionPointAfter(varLoop);
    Value varSum = varLoop.getResult(0);
    Value variance = builder.create<arith::DivFOp>(loc, varSum, N_f32);

    // Add epsilon for numerical stability
    Value var_plus_eps = builder.create<arith::AddFOp>(loc, variance, epsilon);

    // Compute 1 / sqrt(variance + epsilon)
    Value stddev = builder.create<math::SqrtOp>(loc, var_plus_eps);

    // ========================================================================
    // Pass 3: Normalize (parallel)
    // ========================================================================

    Value N_plus_255 = builder.create<arith::AddIOp>(loc, N, c255);
    Value numBlocks = builder.create<arith::DivUIOp>(loc, N_plus_255, c256);

    auto blockLoop = builder.create<scf::ForOp>(loc, c0, numBlocks, c1);
    builder.setInsertionPointToStart(blockLoop.getBody());
    Value blockIdx = blockLoop.getInductionVar();

    auto threadLoop = builder.create<scf::ForOp>(loc, c0, c256, c1);
    builder.setInsertionPointToStart(threadLoop.getBody());
    Value threadIdx = threadLoop.getInductionVar();

    Value blockOffset = builder.create<arith::MulIOp>(loc, blockIdx, c256);
    Value globalIdx = builder.create<arith::AddIOp>(loc, blockOffset, threadIdx);

    Value inBounds = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalIdx, N
    );

    auto ifOp = builder.create<scf::IfOp>(loc, inBounds, /*withElseRegion=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    Value x = builder.create<memref::LoadOp>(loc, input, globalIdx);
    Value x_minus_mean = builder.create<arith::SubFOp>(loc, x, mean);
    Value normalized = builder.create<arith::DivFOp>(loc, x_minus_mean, stddev);
    builder.create<memref::StoreOp>(loc, normalized, output, globalIdx);
}

extern "C" void layernorm_kernel(float* input, float* output, int N, float epsilon) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    MLIRContext* context = ch15::createContext();
    OpBuilder builder(context);
    auto loc = builder.getUnknownLoc();

    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // Create function: func @layernorm(%input: memref<?xf32>, %output: memref<?xf32>, 
    //                                  %N: index, %epsilon: f32)
    auto f32Type = builder.getF32Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, f32Type);
    auto indexType = builder.getIndexType();

    auto funcType = builder.getFunctionType(
        {memrefType, memrefType, indexType, f32Type}, {}
    );
    auto func = builder.create<func::FuncOp>(loc, "layernorm", funcType);
    func.setPublic();

    Block* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    Value input_arg = entryBlock->getArgument(0);
    Value output_arg = entryBlock->getArgument(1);
    Value N_arg = entryBlock->getArgument(2);
    Value epsilon_arg = entryBlock->getArgument(3);

    buildLayerNormKernel(builder, loc, input_arg, output_arg, N_arg, epsilon_arg);

    builder.setInsertionPointToEnd(entryBlock);
    builder.create<func::ReturnOp>(loc);

    if (failed(verify(module))) {
        llvm::errs() << "Module verification failed\n";
        module->dump();
        return;
    }

    if (failed(ch15::lowerToLLVMDialect(module))) {
        llvm::errs() << "Lowering failed\n";
        return;
    }

    llvm::LLVMContext llvmContext;
    auto llvmModule = ch15::translateToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Translation to LLVM IR failed\n";
        return;
    }

    ExecutionEngineOptions options;
    auto transformer = mlir::makeOptimizingTransformer(2, 0, nullptr);
    options.transformer = transformer;
    options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;

    auto engine = ExecutionEngine::create(module, options);
    if (!engine) {
        llvm::errs() << "Failed to create execution engine\n";
        return;
    }

    // Prepare arguments
    // 2 memrefs (5 fields each) + 1 index + 1 f32 = 12 arguments
    int64_t N_val = static_cast<int64_t>(N);
    int64_t zero = 0;
    int64_t one = 1;

    void* args[] = {
        &input, &input, &zero, &N_val, &one,   // input: 5 args
        &output, &output, &zero, &N_val, &one, // output: 5 args
        &N_val,                                  // N: 1 arg
        &epsilon                                 // epsilon: 1 arg (f32)
    };

    auto err = (*engine)->invokePacked("layernorm", llvm::MutableArrayRef<void*>(args, 12));
    if (err) {
        llvm::errs() << "Execution failed: " << err << "\n";
        return;
    }

    delete context;
}