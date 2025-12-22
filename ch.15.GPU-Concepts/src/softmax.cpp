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
// Softmax: softmax(x[i]) = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
// ============================================================================
//
// Algorithm (3-pass for numerical stability):
// Pass 1: Find max value
// Pass 2: Compute exp(x - max) and accumulate sum
// Pass 3: Normalize by dividing by sum
//
// GPU Concept: Each pass uses 1D grid (like Phase 0/2)
// Reduction operations use sequential loops (not optimized for GPU)
//

void buildSoftmaxKernel(OpBuilder& builder, Location loc,
                        Value input, Value output, Value N) {
    // Constants
    Value c0 = ch15::createIndex(builder, loc, 0);
    Value c1 = ch15::createIndex(builder, loc, 1);
    Value c256 = ch15::createIndex(builder, loc, 256);
    Value c255 = ch15::createIndex(builder, loc, 255);
    
    Value negInf = ch15::createFloat(builder, loc, -std::numeric_limits<float>::infinity());
    Value zero_f = ch15::createFloat(builder, loc, 0.0f);
    
    // Allocate temporary buffer for exp values
    auto f32Type = builder.getF32Type();
    auto tempType = MemRefType::get({ShapedType::kDynamic}, f32Type);
    Value temp = builder.create<memref::AllocaOp>(loc, tempType, ValueRange{N});
    
    // ========================================================================
    // Pass 1: Find maximum value (reduction)
    // ========================================================================
    // Use sequential loop for reduction (not parallelized - educational version)
    
    auto maxLoop = builder.create<scf::ForOp>(loc, c0, N, c1, ValueRange{negInf});
    builder.setInsertionPointToStart(maxLoop.getBody());
    
    Value i = maxLoop.getInductionVar();
    Value currentMax = maxLoop.getRegionIterArgs()[0];
    
    Value val = builder.create<memref::LoadOp>(loc, input, i);
    Value newMax = builder.create<arith::MaximumFOp>(loc, currentMax, val);
    builder.create<scf::YieldOp>(loc, newMax);
    
    builder.setInsertionPointAfter(maxLoop);
    Value maxVal = maxLoop.getResult(0);
    
    // ========================================================================
    // Pass 2: Compute exp(x - max) and sum (parallel + reduction)
    // ========================================================================
    
    // Grid size for parallel computation
    Value N_plus_255 = builder.create<arith::AddIOp>(loc, N, c255);
    Value numBlocks = builder.create<arith::DivUIOp>(loc, N_plus_255, c256);
    
    // Parallel: Compute exp(x[i] - max) for all i
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
    Value x_minus_max = builder.create<arith::SubFOp>(loc, x, maxVal);
    Value expVal = builder.create<math::ExpOp>(loc, x_minus_max);
    builder.create<memref::StoreOp>(loc, expVal, temp, globalIdx);
    
    // After parallel computation, do reduction for sum
    builder.setInsertionPointAfter(blockLoop);
    
    auto sumLoop = builder.create<scf::ForOp>(loc, c0, N, c1, ValueRange{zero_f});
    builder.setInsertionPointToStart(sumLoop.getBody());
    
    Value j = sumLoop.getInductionVar();
    Value currentSum = sumLoop.getRegionIterArgs()[0];
    
    Value expVal2 = builder.create<memref::LoadOp>(loc, temp, j);
    Value newSum = builder.create<arith::AddFOp>(loc, currentSum, expVal2);
    builder.create<scf::YieldOp>(loc, newSum);
    
    builder.setInsertionPointAfter(sumLoop);
    Value sumVal = sumLoop.getResult(0);
    
    // ========================================================================
    // Pass 3: Normalize (parallel)
    // ========================================================================
    
    auto blockLoop2 = builder.create<scf::ForOp>(loc, c0, numBlocks, c1);
    builder.setInsertionPointToStart(blockLoop2.getBody());
    Value blockIdx2 = blockLoop2.getInductionVar();
    
    auto threadLoop2 = builder.create<scf::ForOp>(loc, c0, c256, c1);
    builder.setInsertionPointToStart(threadLoop2.getBody());
    Value threadIdx2 = threadLoop2.getInductionVar();
    
    Value blockOffset2 = builder.create<arith::MulIOp>(loc, blockIdx2, c256);
    Value globalIdx2 = builder.create<arith::AddIOp>(loc, blockOffset2, threadIdx2);
    
    Value inBounds2 = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalIdx2, N
    );
    
    auto ifOp2 = builder.create<scf::IfOp>(loc, inBounds2, /*withElseRegion=*/false);
    builder.setInsertionPointToStart(&ifOp2.getThenRegion().front());
    
    Value expVal3 = builder.create<memref::LoadOp>(loc, temp, globalIdx2);
    Value result = builder.create<arith::DivFOp>(loc, expVal3, sumVal);
    builder.create<memref::StoreOp>(loc, result, output, globalIdx2);
}

extern "C" void softmax_kernel(float* input, float* output, int N) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    MLIRContext* context = ch15::createContext();
    OpBuilder builder(context);
    auto loc = builder.getUnknownLoc();

    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // Create function: func @softmax(%input: memref<?xf32>, %output: memref<?xf32>, %N: index)
    auto f32Type = builder.getF32Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, f32Type);
    auto indexType = builder.getIndexType();

    auto funcType = builder.getFunctionType({memrefType, memrefType, indexType}, {});
    auto func = builder.create<func::FuncOp>(loc, "softmax", funcType);
    func.setPublic();

    Block* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    Value input_arg = entryBlock->getArgument(0);
    Value output_arg = entryBlock->getArgument(1);
    Value N_arg = entryBlock->getArgument(2);

    buildSoftmaxKernel(builder, loc, input_arg, output_arg, N_arg);

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

    // Prepare arguments (1D memrefs: 5 fields each)
    int64_t N_val = static_cast<int64_t>(N);
    int64_t zero = 0;
    int64_t one = 1;

    void* args[] = {
        &input, &input, &zero, &N_val, &one,   // input: 5 args
        &output, &output, &zero, &N_val, &one, // output: 5 args
        &N_val                                   // N: 1 arg
    };

    auto err = (*engine)->invokePacked("softmax", llvm::MutableArrayRef<void*>(args, 11));
    if (err) {
        llvm::errs() << "Execution failed: " << err << "\n";
        return;
    }

    delete context;
}
