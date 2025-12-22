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
// Vector Addition Kernel: C[i] = A[i] + B[i]
// ============================================================================
//
// GPU Concept (emulated on CPU):
// - Grid: N/256 blocks (blockIdx from 0 to numBlocks-1)
// - Block: 256 threads (threadIdx from 0 to 255)
// - Global index: i = blockIdx * 256 + threadIdx
// - Bounds check: if (i < N) then compute
//
// CPU Emulation using SCF loops:
// - Outer loop iterates over blocks
// - Inner loop iterates over threads within each block
// - Direct mapping: no actual GPU involved, runs on CPU
//

void buildVectorAddKernel(OpBuilder& builder, Location loc,
                          Value A, Value B, Value C, Value N) {
    // Constants
    Value c0 = ch15::createIndex(builder, loc, 0);
    Value c1 = ch15::createIndex(builder, loc, 1);
    Value c256 = ch15::createIndex(builder, loc, 256);
    Value c255 = ch15::createIndex(builder, loc, 255);

    // Compute grid size: numBlocks = (N + 255) / 256 (ceiling division)
    Value N_plus_255 = builder.create<arith::AddIOp>(loc, N, c255);
    Value numBlocks = builder.create<arith::DivUIOp>(loc, N_plus_255, c256);

    // Outer loop: iterate over blocks (emulates GPU grid)
    auto blockLoop = builder.create<scf::ForOp>(loc, c0, numBlocks, c1);
    builder.setInsertionPointToStart(blockLoop.getBody());
    Value blockIdx = blockLoop.getInductionVar();

    // Inner loop: iterate over threads (emulates GPU block)
    auto threadLoop = builder.create<scf::ForOp>(loc, c0, c256, c1);
    builder.setInsertionPointToStart(threadLoop.getBody());
    Value threadIdx = threadLoop.getInductionVar();

    // Compute global index: i = blockIdx * 256 + threadIdx
    Value blockOffset = builder.create<arith::MulIOp>(loc, blockIdx, c256);
    Value globalIdx = builder.create<arith::AddIOp>(loc, blockOffset, threadIdx);

    // Bounds check: if (i < N)
    Value inBounds = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalIdx, N
    );

    // Conditional computation
    auto ifOp = builder.create<scf::IfOp>(loc, inBounds, /*withElseRegion=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    // Load inputs
    Value aVal = builder.create<memref::LoadOp>(loc, A, ValueRange{globalIdx});
    Value bVal = builder.create<memref::LoadOp>(loc, B, ValueRange{globalIdx});

    // Compute sum
    Value sum = builder.create<arith::AddFOp>(loc, aVal, bVal);

    // Store result
    builder.create<memref::StoreOp>(loc, sum, C, ValueRange{globalIdx});

    // Restore insertion point after all loops
    builder.setInsertionPointAfter(blockLoop);
}

// ============================================================================
// Compile and Execute Vector Add Kernel
// ============================================================================

extern "C" void vector_add_kernel(float* A_ptr, float* B_ptr, float* C_ptr, int N) {
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

    // Create function signature: (memref<?xf32>, memref<?xf32>, memref<?xf32>, index) -> ()
    auto f32Type = builder.getF32Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, f32Type);
    auto indexType = builder.getIndexType();

    auto funcType = builder.getFunctionType(
        {memrefType, memrefType, memrefType, indexType},
        {}
    );

    // Create function
    auto func = builder.create<func::FuncOp>(loc, "vector_add", funcType);
    func.setPublic();

    // Build function body
    Block* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    Value A = entryBlock->getArgument(0);
    Value B = entryBlock->getArgument(1);
    Value C = entryBlock->getArgument(2);
    Value N_idx = entryBlock->getArgument(3);

    // Build the kernel computation
    buildVectorAddKernel(builder, loc, A, B, C, N_idx);

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

    // Prepare memref descriptors for JIT call
    // After lowering to LLVM, memref<?xf32> becomes 5 separate arguments:
    // (allocated_ptr, aligned_ptr, offset, size, stride)
    int64_t N_val = static_cast<int64_t>(N);
    int64_t zero = 0;
    int64_t one = 1;

    // Prepare argument array (3 memrefs × 5 fields + 1 index = 16 args)
    void* args[] = {
        // A: memref<?xf32>
        &A_ptr,             // allocated
        &A_ptr,             // aligned
        &zero,              // offset
        &N_val,             // size
        &one,               // stride

        // B: memref<?xf32>
        &B_ptr,             // allocated
        &B_ptr,             // aligned
        &zero,              // offset
        &N_val,             // size
        &one,               // stride

        // C: memref<?xf32>
        &C_ptr,             // allocated
        &C_ptr,             // aligned
        &zero,              // offset
        &N_val,             // size
        &one,               // stride

        // N: index
        &N_val
    };

    // Invoke the compiled function
    auto err = (*engine)->invokePacked(
        "vector_add",
        llvm::MutableArrayRef<void*>(args, 16)
    );

    if (err) {
        llvm::errs() << "Execution failed: " << err << "\n";
        return;
    }

    // Cleanup
    delete context;
}