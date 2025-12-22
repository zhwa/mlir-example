// Phase 6: Single-Head Attention
// Demonstrates composing multiple GPU operations into a complete attention mechanism
//
// Algorithm:
//   1. scores = Q @ K^T        (matmul + transpose)
//   2. scores = scores / √d_k  (scaling)
//   3. attn = softmax(scores)  (normalization)
//   4. output = attn @ V       (weighted sum)
//
// This is the core of transformer models!

#include "common.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <cmath>

using namespace mlir;

// ============================================================================
// Kernel 1: Element-wise Scale (multiply by constant)
// ============================================================================

void buildScaleKernel(OpBuilder& builder, Location loc,
                      Value input, Value output, Value N, Value scale_factor) {
    // Grid: ceil(N/256) blocks
    // Block: 256 threads
    // Each thread: output[i] = input[i] * scale_factor

    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value c256 = builder.create<arith::ConstantIndexOp>(loc, 256);

    // Grid size: ceil(N / 256)
    Value c255 = builder.create<arith::ConstantIndexOp>(loc, 255);
    Value N_plus_255 = builder.create<arith::AddIOp>(loc, N, c255);
    Value numBlocks = builder.create<arith::DivUIOp>(loc, N_plus_255, c256);

    // Loop over blocks
    builder.create<scf::ForOp>(loc, c0, numBlocks, c1, ValueRange{},
        [&](OpBuilder& builder, Location loc, Value blockIdx, ValueRange) {

        // Loop over threads in block
        builder.create<scf::ForOp>(loc, c0, c256, c1, ValueRange{},
            [&](OpBuilder& builder, Location loc, Value threadIdx, ValueRange) {

            // Global index: i = blockIdx * 256 + threadIdx
            Value blockOffset = builder.create<arith::MulIOp>(loc, blockIdx, c256);
            Value i = builder.create<arith::AddIOp>(loc, blockOffset, threadIdx);

            // Bounds check
            Value inBounds = builder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::slt, i, N);

            builder.create<scf::IfOp>(loc, inBounds, [&](OpBuilder& builder, Location loc) {
                // Load input[i]
                Value val = builder.create<memref::LoadOp>(loc, input, ValueRange{i});

                // Multiply by scale factor
                Value scaled = builder.create<arith::MulFOp>(loc, val, scale_factor);

                // Store to output[i]
                builder.create<memref::StoreOp>(loc, scaled, output, ValueRange{i});

                builder.create<scf::YieldOp>(loc);
            });

            builder.create<scf::YieldOp>(loc);
        });

        builder.create<scf::YieldOp>(loc);
    });
}

extern "C" void scale_kernel(float* input, float* output, int N, float scale_factor) {
    auto context = ch15::createContext();
    auto builder = OpBuilder(context);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // Types
    auto f32Type = builder.getF32Type();
    auto scalarType = builder.getIndexType();
    auto memrefType = MemRefType::get({-1}, f32Type);

    // Function signature: (memref<?xf32>, memref<?xf32>, index, f32) -> ()
    auto funcType = builder.getFunctionType(
        {memrefType, memrefType, scalarType, f32Type}, {});
    auto func = builder.create<func::FuncOp>(loc, "scale", funcType);

    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    Value input_arg = entryBlock->getArgument(0);
    Value output_arg = entryBlock->getArgument(1);
    Value N_idx = entryBlock->getArgument(2);
    Value scale_arg = entryBlock->getArgument(3);

    buildScaleKernel(builder, loc, input_arg, output_arg, N_idx, scale_arg);

    builder.setInsertionPointToEnd(entryBlock);
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

    // Prepare arguments (11 total for 2 memrefs + 2 scalars)
    int64_t zero = 0;
    int64_t one = 1;
    int64_t N_val = static_cast<int64_t>(N);

    void* args[] = {
        &input, &input, &zero, &N_val, &one,  // input memref (5)
        &output, &output, &zero, &N_val, &one,  // output memref (5)
        &N_val,  // N index (1)
        &scale_factor  // scale_factor f32 (1)
    };

    auto result = (*engine)->invokePacked("scale", args);
    if (result) {
        llvm::errs() << "JIT invocation failed\n";
    }
}

// ============================================================================
// Kernel 2: Scaled Dot-Product Attention (Single-Head)
// ============================================================================
//
// Input:
//   Q: [seq_len, d_k] - Query matrix
//   K: [seq_len, d_k] - Key matrix  
//   V: [seq_len, d_v] - Value matrix
//
// Output:
//   output: [seq_len, d_v] - Attention output
//
// Algorithm:
//   1. scores = Q @ K^T          → [seq_len, seq_len]
//   2. scores = scores / √d_k     (scaling for numerical stability)
//   3. attn_weights = softmax(scores, dim=-1)
//   4. output = attn_weights @ V  → [seq_len, d_v]
//
// Note: This is single-head for simplicity. Multi-head would add reshape ops.

// Forward declarations of external kernels we'll reuse
extern "C" {
    void matmul_kernel(float* A, float* B, float* C, int M, int N, int K);
    void transpose_kernel(float* input, float* output, int M, int N);
    void scale_kernel(float* input, float* output, int N, float scale_factor);
    void softmax_kernel(float* input, float* output, int N);
}

extern "C" void attention_kernel(
    float* Q,      // Query: [seq_len, d_k]
    float* K,      // Key:   [seq_len, d_k]
    float* V,      // Value: [seq_len, d_v]
    float* output, // Output: [seq_len, d_v]
    int seq_len,   // Sequence length
    int d_k,       // Key/Query dimension
    int d_v        // Value dimension
) {
    // Allocate temporary buffers
    int scores_size = seq_len * seq_len;
    int kT_size = d_k * seq_len;  // K^T is [d_k, seq_len]

    float* K_T = new float[kT_size];
    float* scores = new float[scores_size];
    float* scaled_scores = new float[scores_size];
    float* attn_weights = new float[scores_size];

    // Step 1: Transpose K to get K^T
    // K: [seq_len, d_k] → K_T: [d_k, seq_len]
    transpose_kernel(K, K_T, seq_len, d_k);

    // Step 2: Compute scores = Q @ K^T
    // Q: [seq_len, d_k] @ K_T: [d_k, seq_len] → scores: [seq_len, seq_len]
    matmul_kernel(Q, K_T, scores, seq_len, seq_len, d_k);

    // Step 3: Scale scores by 1/√d_k
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));

    // Apply scaling to entire scores matrix (treat as flat array)
    scale_kernel(scores, scaled_scores, scores_size, scale_factor);

    // Step 4: Apply softmax row-wise
    // Each row of [seq_len, seq_len] needs independent softmax
    for (int i = 0; i < seq_len; i++) {
        softmax_kernel(scaled_scores + i * seq_len, 
                      attn_weights + i * seq_len, 
                      seq_len);
    }

    // Step 5: Compute final output = attn_weights @ V
    // attn_weights: [seq_len, seq_len] @ V: [seq_len, d_v] → output: [seq_len, d_v]
    matmul_kernel(attn_weights, V, output, seq_len, d_v, seq_len);

    // Cleanup
    delete[] K_T;
    delete[] scores;
    delete[] scaled_scores;
    delete[] attn_weights;
}