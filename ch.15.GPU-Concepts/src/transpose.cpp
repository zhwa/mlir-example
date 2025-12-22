#include "common.h"

#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

#include "llvm/Support/TargetSelect.h"

using namespace mlir;

// ============================================================================
// Transpose: output[j][i] = input[i][j]
// ============================================================================
//
// GPU Concept: 2D grid with memory access pattern considerations
// - Each thread reads from input[row][col]
// - Each thread writes to output[col][row]
// - Demonstrates non-coalesced memory access (educational version)
//
// Real GPU optimization would use shared memory tiling to improve coalescing.
// Here we show the basic concept.
//

void buildTransposeKernel(OpBuilder& builder, Location loc,
                          Value input, Value output, Value M, Value N) {
    // Constants
    Value c0 = ch15::createIndex(builder, loc, 0);
    Value c1 = ch15::createIndex(builder, loc, 1);
    Value c16 = ch15::createIndex(builder, loc, 16);
    Value c15 = ch15::createIndex(builder, loc, 15);

    // Grid size: ceil(M/16) × ceil(N/16) blocks
    Value M_plus_15 = builder.create<arith::AddIOp>(loc, M, c15);
    Value numBlocksX = builder.create<arith::DivUIOp>(loc, M_plus_15, c16);

    Value N_plus_15 = builder.create<arith::AddIOp>(loc, N, c15);
    Value numBlocksY = builder.create<arith::DivUIOp>(loc, N_plus_15, c16);

    // 2D grid of 2D blocks (like Phase 1 MatMul)
    auto blockLoopX = builder.create<scf::ForOp>(loc, c0, numBlocksX, c1);
    builder.setInsertionPointToStart(blockLoopX.getBody());
    Value blockIdxX = blockLoopX.getInductionVar();

    auto blockLoopY = builder.create<scf::ForOp>(loc, c0, numBlocksY, c1);
    builder.setInsertionPointToStart(blockLoopY.getBody());
    Value blockIdxY = blockLoopY.getInductionVar();

    auto threadLoopX = builder.create<scf::ForOp>(loc, c0, c16, c1);
    builder.setInsertionPointToStart(threadLoopX.getBody());
    Value threadIdxX = threadLoopX.getInductionVar();

    auto threadLoopY = builder.create<scf::ForOp>(loc, c0, c16, c1);
    builder.setInsertionPointToStart(threadLoopY.getBody());
    Value threadIdxY = threadLoopY.getInductionVar();

    // Compute global indices
    Value blockOffsetX = builder.create<arith::MulIOp>(loc, blockIdxX, c16);
    Value row = builder.create<arith::AddIOp>(loc, blockOffsetX, threadIdxX);

    Value blockOffsetY = builder.create<arith::MulIOp>(loc, blockIdxY, c16);
    Value col = builder.create<arith::AddIOp>(loc, blockOffsetY, threadIdxY);

    // Bounds check
    Value validRow = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, row, M
    );
    Value validCol = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, col, N
    );
    Value valid = builder.create<arith::AndIOp>(loc, validRow, validCol);

    auto ifOp = builder.create<scf::IfOp>(loc, valid, /*withElseRegion=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    // Transpose: read input[row][col], write to output[col][row]
    Value val = builder.create<memref::LoadOp>(loc, input, ValueRange{row, col});
    builder.create<memref::StoreOp>(loc, val, output, ValueRange{col, row});
}

extern "C" void transpose_kernel(float* input, float* output, int M, int N) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    MLIRContext* context = ch15::createContext();
    OpBuilder builder(context);
    auto loc = builder.getUnknownLoc();

    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // Create function: func @transpose(%input: memref<?x?xf32>, %output: memref<?x?xf32>,
    //                                  %M: index, %N: index)
    auto f32Type = builder.getF32Type();
    auto memrefType2D = MemRefType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic}, f32Type
    );
    auto indexType = builder.getIndexType();

    auto funcType = builder.getFunctionType(
        {memrefType2D, memrefType2D, indexType, indexType}, {}
    );
    auto func = builder.create<func::FuncOp>(loc, "transpose", funcType);
    func.setPublic();

    Block* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    Value input_arg = entryBlock->getArgument(0);
    Value output_arg = entryBlock->getArgument(1);
    Value M_arg = entryBlock->getArgument(2);
    Value N_arg = entryBlock->getArgument(3);

    buildTransposeKernel(builder, loc, input_arg, output_arg, M_arg, N_arg);

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
    // Input: M×N, Output: N×M (transposed dimensions)
    int64_t M_val = static_cast<int64_t>(M);
    int64_t N_val = static_cast<int64_t>(N);
    int64_t zero = 0;

    // Input: M×N (row-major: stride[0]=N, stride[1]=1)
    int64_t input_stride0 = N_val;
    int64_t input_stride1 = 1;

    // Output: N×M (row-major: stride[0]=M, stride[1]=1)
    int64_t output_stride0 = M_val;
    int64_t output_stride1 = 1;

    void* args[] = {
        // Input: M×N
        &input, &input, &zero,
        &M_val, &input_stride0,
        &N_val, &input_stride1,

        // Output: N×M (transposed dimensions!)
        &output, &output, &zero,
        &N_val, &output_stride0,  // First dim is now N
        &M_val, &output_stride1,  // Second dim is now M

        // Dimensions
        &M_val, &N_val
    };

    // Total: 2 memrefs × 7 fields + 2 indices = 16 arguments
    auto err = (*engine)->invokePacked("transpose", llvm::MutableArrayRef<void*>(args, 16));
    if (err) {
        llvm::errs() << "Execution failed: " << err << "\n";
        return;
    }

    delete context;
}