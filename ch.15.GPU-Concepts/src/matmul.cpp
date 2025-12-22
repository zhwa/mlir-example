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
// Matrix Multiplication Kernel: C[M×N] = A[M×K] @ B[K×N]
// ============================================================================
//
// GPU Concept (emulated on CPU):
// - Grid: (M/16) × (N/16) blocks (2D grid)
// - Block: 16 × 16 threads (2D block)
// - Each thread computes ONE element of C
// - Global indices:
//   row = blockIdx.x * 16 + threadIdx.x
//   col = blockIdx.y * 16 + threadIdx.y
//
// CPU Emulation using 4 nested SCF loops:
// - Loop 1: blockIdx.x (blocks in X dimension)
// - Loop 2: blockIdx.y (blocks in Y dimension)
// - Loop 3: threadIdx.x (threads in X dimension)
// - Loop 4: threadIdx.y (threads in Y dimension)
// - Loop 5: reduction over K dimension
//

void buildMatMulKernel(OpBuilder& builder, Location loc,
                       Value A, Value B, Value C,
                       Value M, Value N, Value K) {
    // Constants
    Value c0 = ch15::createIndex(builder, loc, 0);
    Value c1 = ch15::createIndex(builder, loc, 1);
    Value c16 = ch15::createIndex(builder, loc, 16);
    Value c15 = ch15::createIndex(builder, loc, 15);
    Value initValue = ch15::createFloat(builder, loc, 0.0f);

    // Grid size: ceil(M/16) × ceil(N/16) blocks
    Value M_plus_15 = builder.create<arith::AddIOp>(loc, M, c15);
    Value numBlocksX = builder.create<arith::DivUIOp>(loc, M_plus_15, c16);

    Value N_plus_15 = builder.create<arith::AddIOp>(loc, N, c15);
    Value numBlocksY = builder.create<arith::DivUIOp>(loc, N_plus_15, c16);

    // Loop 1: Blocks in X dimension (blockIdx.x - iterates over row blocks)
    auto blockLoopX = builder.create<scf::ForOp>(loc, c0, numBlocksX, c1);
    builder.setInsertionPointToStart(blockLoopX.getBody());
    Value blockIdxX = blockLoopX.getInductionVar();

    // Loop 2: Blocks in Y dimension (blockIdx.y - iterates over col blocks)
    auto blockLoopY = builder.create<scf::ForOp>(loc, c0, numBlocksY, c1);
    builder.setInsertionPointToStart(blockLoopY.getBody());
    Value blockIdxY = blockLoopY.getInductionVar();

    // Loop 3: Threads in X dimension (threadIdx.x - rows within block)
    auto threadLoopX = builder.create<scf::ForOp>(loc, c0, c16, c1);
    builder.setInsertionPointToStart(threadLoopX.getBody());
    Value threadIdxX = threadLoopX.getInductionVar();

    // Loop 4: Threads in Y dimension (threadIdx.y - cols within block)
    auto threadLoopY = builder.create<scf::ForOp>(loc, c0, c16, c1);
    builder.setInsertionPointToStart(threadLoopY.getBody());
    Value threadIdxY = threadLoopY.getInductionVar();

    // Compute global indices
    // row = blockIdx.x * 16 + threadIdx.x
    Value blockOffsetX = builder.create<arith::MulIOp>(loc, blockIdxX, c16);
    Value row = builder.create<arith::AddIOp>(loc, blockOffsetX, threadIdxX);

    // col = blockIdx.y * 16 + threadIdx.y
    Value blockOffsetY = builder.create<arith::MulIOp>(loc, blockIdxY, c16);
    Value col = builder.create<arith::AddIOp>(loc, blockOffsetY, threadIdxY);

    // Bounds check: if (row < M && col < N)
    Value validRow = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, row, M
    );
    Value validCol = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, col, N
    );
    Value valid = builder.create<arith::AndIOp>(loc, validRow, validCol);

    // Conditional computation
    auto ifOp = builder.create<scf::IfOp>(loc, valid, /*withElseRegion=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    // Loop 5: Reduction loop over K dimension
    // sum = 0
    // for k in 0..K:
    //   sum += A[row][k] * B[k][col]
    // C[row][col] = sum
    auto reductionLoop = builder.create<scf::ForOp>(
        loc, c0, K, c1, ValueRange{initValue}
    );
    builder.setInsertionPointToStart(reductionLoop.getBody());

    Value k = reductionLoop.getInductionVar();
    Value acc = reductionLoop.getRegionIterArgs()[0];

    // Load A[row][k] and B[k][col]
    Value aVal = builder.create<memref::LoadOp>(loc, A, ValueRange{row, k});
    Value bVal = builder.create<memref::LoadOp>(loc, B, ValueRange{k, col});

    // Compute product and accumulate
    Value prod = builder.create<arith::MulFOp>(loc, aVal, bVal);
    Value newAcc = builder.create<arith::AddFOp>(loc, acc, prod);
    builder.create<scf::YieldOp>(loc, newAcc);

    // Store result: C[row][col] = sum
    builder.setInsertionPointAfter(reductionLoop);
    Value result = reductionLoop.getResult(0);
    builder.create<memref::StoreOp>(loc, result, C, ValueRange{row, col});
}

// ============================================================================
// Compile and Execute MatMul Kernel
// ============================================================================

extern "C" void matmul_kernel(float* A_ptr, float* B_ptr, float* C_ptr,
                              int M, int N, int K) {
    // Initialize LLVM
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Create MLIR context
    MLIRContext* context = ch15::createContext();
    OpBuilder builder(context);
    auto loc = builder.getUnknownLoc();

    // Create module
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // Create function signature:
    // func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>,
    //              %M: index, %N: index, %K: index)
    auto f32Type = builder.getF32Type();
    auto memrefType2D = MemRefType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic}, f32Type
    );
    auto indexType = builder.getIndexType();

    auto funcType = builder.getFunctionType(
        {memrefType2D, memrefType2D, memrefType2D, indexType, indexType, indexType},
        {}
    );

    // Create function
    auto func = builder.create<func::FuncOp>(loc, "matmul", funcType);
    func.setPublic();

    // Build function body
    Block* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    Value A = entryBlock->getArgument(0);
    Value B = entryBlock->getArgument(1);
    Value C = entryBlock->getArgument(2);
    Value M_idx = entryBlock->getArgument(3);
    Value N_idx = entryBlock->getArgument(4);
    Value K_idx = entryBlock->getArgument(5);

    // Build the kernel computation
    buildMatMulKernel(builder, loc, A, B, C, M_idx, N_idx, K_idx);

    // Set insertion point back to function level (after all loops)
    builder.setInsertionPointToEnd(entryBlock);

    // Return
    builder.create<func::ReturnOp>(loc);

    // Verify module
    if (failed(verify(module))) {
        llvm::errs() << "Module verification failed\n";
        module->dump();
        return;
    }

    // Lower to LLVM dialect
    if (failed(ch15::lowerToLLVMDialect(module))) {
        llvm::errs() << "Lowering failed\n";
        return;
    }

    // Translate to LLVM IR
    llvm::LLVMContext llvmContext;
    auto llvmModule = ch15::translateToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Translation to LLVM IR failed\n";
        return;
    }

    // Create execution engine
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
    // 2D memref descriptor has 10 fields: {allocated, aligned, offset, size[0], size[1], stride[0], stride[1]}
    // Actually for dynamic 2D: {allocated, aligned, offset, size[0], stride[0], size[1], stride[1]}
    int64_t M_val = static_cast<int64_t>(M);
    int64_t N_val = static_cast<int64_t>(N);
    int64_t K_val = static_cast<int64_t>(K);
    int64_t zero = 0;

    // A: M×K matrix (row-major: stride[0]=K, stride[1]=1)
    int64_t A_stride0 = K_val;
    int64_t A_stride1 = 1;

    // B: K×N matrix (row-major: stride[0]=N, stride[1]=1)
    int64_t B_stride0 = N_val;
    int64_t B_stride1 = 1;

    // C: M×N matrix (row-major: stride[0]=N, stride[1]=1)
    int64_t C_stride0 = N_val;
    int64_t C_stride1 = 1;

    void* args[] = {
        // A: memref<?x?xf32> → 7 arguments
        &A_ptr, &A_ptr, &zero,
        &M_val, &A_stride0,  // dim 0: size, stride
        &K_val, &A_stride1,  // dim 1: size, stride

        // B: memref<?x?xf32> → 7 arguments
        &B_ptr, &B_ptr, &zero,
        &K_val, &B_stride0,  // dim 0: size, stride
        &N_val, &B_stride1,  // dim 1: size, stride

        // C: memref<?x?xf32> → 7 arguments
        &C_ptr, &C_ptr, &zero,
        &M_val, &C_stride0,  // dim 0: size, stride
        &N_val, &C_stride1,  // dim 1: size, stride

        // M, N, K: 3 index arguments
        &M_val, &N_val, &K_val
    };

    // Total: 3 memrefs × 7 fields + 3 indices = 24 arguments
    auto err = (*engine)->invokePacked(
        "matmul",
        llvm::MutableArrayRef<void*>(args, 24)
    );

    if (err) {
        llvm::errs() << "Execution failed: " << err << "\n";
        return;
    }

    // Cleanup
    delete context;
}