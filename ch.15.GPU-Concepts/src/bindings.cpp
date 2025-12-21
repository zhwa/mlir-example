#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

// Dialects
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"

// Transforms
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

// Conversions
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

#include "llvm/Support/TargetSelect.h"

namespace py = pybind11;
using namespace mlir;

// ============================================================================
// Phase 0: Vector Addition - GPU Concepts with Direct SCF Emulation
// ============================================================================

// Build GPU-style computation using SCF loops (CPU emulation)
// C[i] = A[i] + B[i]
// 
// GPU Concept: Grid of blocks, each block has threads
// - Grid: N/256 blocks (blockIdx.x = 0..numBlocks-1)
// - Block: 256 threads (threadIdx.x = 0..255)
// - Global index: i = blockIdx.x * 256 + threadIdx.x
//
// CPU Emulation: Nested loops
// - Outer loop: iterate over blocks
// - Inner loop: iterate over threads
void buildVectorAddGPU(OpBuilder &builder, Location loc, 
                       Value A, Value B, Value C, Value N) {

  // Constants
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c256 = builder.create<arith::ConstantIndexOp>(loc, 256);

  // Compute grid size: num_blocks = (N + 255) / 256
  Value c255 = builder.create<arith::ConstantIndexOp>(loc, 255);
  Value N_plus_255 = builder.create<arith::AddIOp>(loc, N, c255);
  Value numBlocks = builder.create<arith::DivUIOp>(loc, N_plus_255, c256);

  // Outer loop: blocks (blockIdx.x)
  auto blockLoop = builder.create<scf::ForOp>(loc, c0, numBlocks, c1);
  builder.setInsertionPointToStart(blockLoop.getBody());
  Value blockIdx = blockLoop.getInductionVar();

  // Inner loop: threads (threadIdx.x)
  auto threadLoop = builder.create<scf::ForOp>(loc, c0, c256, c1);
  builder.setInsertionPointToStart(threadLoop.getBody());
  Value threadIdx = threadLoop.getInductionVar();

  // Compute global index: i = blockIdx * 256 + threadIdx
  Value blockOffset = builder.create<arith::MulIOp>(loc, blockIdx, c256);
  Value globalIdx = builder.create<arith::AddIOp>(loc, blockOffset, threadIdx);

  // Bounds check: if (i < N)
  Value inBounds = builder.create<arith::CmpIOp>(
    loc, arith::CmpIPredicate::ult, globalIdx, N);

  auto ifOp = builder.create<scf::IfOp>(loc, inBounds, /*withElseRegion=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

  // Load A[i] and B[i]
  Value aVal = builder.create<memref::LoadOp>(loc, A, ValueRange{globalIdx});
  Value bVal = builder.create<memref::LoadOp>(loc, B, ValueRange{globalIdx});

  // Compute C[i] = A[i] + B[i]
  Value sum = builder.create<arith::AddFOp>(loc, aVal, bVal);

  // Store result
  builder.create<memref::StoreOp>(loc, sum, C, ValueRange{globalIdx});

  // NOTE: Do NOT manually add scf.yield for the if operation - it's automatically added
  // The scf.for operations also automatically manage their terminators
}

// Build test kernel to verify thread indexing
// Output[i] = 1.0 if i == target_index else 0.0
//
// GPU Concept: blockIdx.x * blockSize + threadIdx.x = global_index
// CPU Emulation: Nested loops compute same index
void buildThreadIndexingTest(OpBuilder &builder, Location loc,
                              Value output, Value N, Value targetIndex,
                              Value one, Value zero) {
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c256 = builder.create<arith::ConstantIndexOp>(loc, 256);
  Value c255 = builder.create<arith::ConstantIndexOp>(loc, 255);

  Value N_plus_255 = builder.create<arith::AddIOp>(loc, N, c255);
  Value numBlocks = builder.create<arith::DivUIOp>(loc, N_plus_255, c256);

  // Outer loop: blocks
  auto blockLoop = builder.create<scf::ForOp>(loc, c0, numBlocks, c1);
  builder.setInsertionPointToStart(blockLoop.getBody());
  Value blockIdx = blockLoop.getInductionVar();

  // Inner loop: threads
  auto threadLoop = builder.create<scf::ForOp>(loc, c0, c256, c1);
  builder.setInsertionPointToStart(threadLoop.getBody());
  Value threadIdx = threadLoop.getInductionVar();

  Value blockOffset = builder.create<arith::MulIOp>(loc, blockIdx, c256);
  Value globalIdx = builder.create<arith::AddIOp>(loc, blockOffset, threadIdx);

  Value inBounds = builder.create<arith::CmpIOp>(
    loc, arith::CmpIPredicate::ult, globalIdx, N);

  auto ifOp = builder.create<scf::IfOp>(loc, inBounds, false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

  // Check if this is the target thread
  Value isTarget = builder.create<arith::CmpIOp>(
    loc, arith::CmpIPredicate::eq, globalIdx, targetIndex);

  // Create if-else to set output[i] = 1.0 or 0.0
  auto innerIf = builder.create<scf::IfOp>(loc, isTarget, /*withElseRegion=*/true);

  // Then region: output[i] = 1.0
  builder.setInsertionPointToStart(&innerIf.getThenRegion().front());
  builder.create<memref::StoreOp>(loc, one, output, ValueRange{globalIdx});

  // Else region: output[i] = 0.0
  builder.setInsertionPointToStart(&innerIf.getElseRegion().front());
  builder.create<memref::StoreOp>(loc, zero, output, ValueRange{globalIdx});

  // NOTE: Do NOT add extra yields - scf.for automatically manages terminators
}

// ============================================================================
// Phase 1: Matrix Multiplication - 2D GPU Thread Hierarchy
// ============================================================================

// Build 2D MatMul using GPU thread hierarchy concepts
// C[row][col] = sum_k(A[row][k] * B[k][col])
//
// GPU Concept: 2D grid of blocks, each block has 2D threads
// - Grid: (M/16) × (N/16) blocks
// - Block: 16 × 16 threads
// - Global indices: 
//   row = blockIdx.x * 16 + threadIdx.x
//   col = blockIdx.y * 16 + threadIdx.y
//
// CPU Emulation: 4 nested loops (blocks_x, blocks_y, threads_x, threads_y)
void buildMatMulGPU(OpBuilder &builder, Location loc,
                    Value A, Value B, Value C,
                    Value M, Value N, Value K,
                    Value initValue) {
  // Constants
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c16 = builder.create<arith::ConstantIndexOp>(loc, 16);
  Value c15 = builder.create<arith::ConstantIndexOp>(loc, 15);
  
  // Grid size: (M+15)/16 × (N+15)/16 blocks
  Value M_plus_15 = builder.create<arith::AddIOp>(loc, M, c15);
  Value numBlocksX = builder.create<arith::DivUIOp>(loc, M_plus_15, c16);
  
  Value N_plus_15 = builder.create<arith::AddIOp>(loc, N, c15);
  Value numBlocksY = builder.create<arith::DivUIOp>(loc, N_plus_15, c16);
  
  // Loop 1: Blocks in X dimension (blockIdx.x)
  auto blockLoopX = builder.create<scf::ForOp>(loc, c0, numBlocksX, c1);
  builder.setInsertionPointToStart(blockLoopX.getBody());
  Value blockIdxX = blockLoopX.getInductionVar();
  
  // Loop 2: Blocks in Y dimension (blockIdx.y)
  auto blockLoopY = builder.create<scf::ForOp>(loc, c0, numBlocksY, c1);
  builder.setInsertionPointToStart(blockLoopY.getBody());
  Value blockIdxY = blockLoopY.getInductionVar();
  
  // Loop 3: Threads in X dimension (threadIdx.x)
  auto threadLoopX = builder.create<scf::ForOp>(loc, c0, c16, c1);
  builder.setInsertionPointToStart(threadLoopX.getBody());
  Value threadIdxX = threadLoopX.getInductionVar();
  
  // Loop 4: Threads in Y dimension (threadIdx.y)
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
    loc, arith::CmpIPredicate::ult, row, M);
  Value validCol = builder.create<arith::CmpIOp>(
    loc, arith::CmpIPredicate::ult, col, N);
  Value valid = builder.create<arith::AndIOp>(loc, validRow, validCol);
  
  auto ifOp = builder.create<scf::IfOp>(loc, valid, /*withElseRegion=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  
  // Reduction loop: sum += A[row][k] * B[k][col]
  auto reductionLoop = builder.create<scf::ForOp>(
    loc, c0, K, c1, ValueRange{initValue});
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
// Phase 2: Element-wise Operations - 1D Thread Hierarchy
// ============================================================================

// Element-wise GELU activation
// GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
//
// TEST K: ALL constants passed as function arguments to bypass MLIR 19 constant pool bug
void buildGELU(OpBuilder &builder, Location loc, Value input, Value output, Value N,
               Value c_half, Value c_sqrt_2_over_pi, Value c_coeff, Value c_one, Value c_27, Value c_9) {
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  
  // Single flat loop from 0 to N
  auto loop = builder.create<scf::ForOp>(loc, c0, N, c1);
  builder.setInsertionPointToStart(loop.getBody());
  Value i = loop.getInductionVar();
  
  // Load input: x
  Value x = builder.create<memref::LoadOp>(loc, input, ValueRange{i});
  
  // x^2
  Value x2 = builder.create<arith::MulFOp>(loc, x, x);
  // x^3
  Value x3 = builder.create<arith::MulFOp>(loc, x2, x);
  // 0.044715 * x^3
  Value term = builder.create<arith::MulFOp>(loc, c_coeff, x3);
  // x + 0.044715 * x^3
  Value inner = builder.create<arith::AddFOp>(loc, x, term);
  // sqrt(2/pi) * (x + 0.044715 * x^3)
  Value scaled = builder.create<arith::MulFOp>(loc, c_sqrt_2_over_pi, inner);
  
  // tanh(scaled) using Padé approximation:  x * (27 + x^2) / (27 + 9*x^2)
  Value scaled2 = builder.create<arith::MulFOp>(loc, scaled, scaled);
  
  // NO arith::ConstantOp here! c_27 and c_9 passed as arguments!
  
  Value numer_inner = builder.create<arith::AddFOp>(loc, c_27, scaled2);
  Value scaled2_times_9 = builder.create<arith::MulFOp>(loc, c_9, scaled2);
  Value denom = builder.create<arith::AddFOp>(loc, c_27, scaled2_times_9);
  Value ratio = builder.create<arith::DivFOp>(loc, numer_inner, denom);
  Value tanh_approx = builder.create<arith::MulFOp>(loc, scaled, ratio);
  
  // 1 + tanh(...)
  Value one_plus_tanh = builder.create<arith::AddFOp>(loc, c_one, tanh_approx);
  // x * (1 + tanh(...))
  Value x_times = builder.create<arith::MulFOp>(loc, x, one_plus_tanh);
  // 0.5 * x * (1 + tanh(...))
  Value result = builder.create<arith::MulFOp>(loc, c_half, x_times);
  
  builder.create<memref::StoreOp>(loc, result, output, ValueRange{i});
  
  // Reset insertion point after loop
  builder.setInsertionPointAfter(loop);
}

// Element-wise addition: z[i] = x[i] + y[i]
void buildElementwiseAdd(OpBuilder &builder, Location loc, 
                         Value x, Value y, Value z, Value N) {
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c256 = builder.create<arith::ConstantIndexOp>(loc, 256);
  Value c255 = builder.create<arith::ConstantIndexOp>(loc, 255);
  
  Value N_plus_255 = builder.create<arith::AddIOp>(loc, N, c255);
  Value numBlocks = builder.create<arith::DivUIOp>(loc, N_plus_255, c256);
  
  // Blocks
  auto blockLoop = builder.create<scf::ForOp>(loc, c0, numBlocks, c1);
  builder.setInsertionPointToStart(blockLoop.getBody());
  Value blockIdx = blockLoop.getInductionVar();
  
  // Threads
  auto threadLoop = builder.create<scf::ForOp>(loc, c0, c256, c1);
  builder.setInsertionPointToStart(threadLoop.getBody());
  Value threadIdx = threadLoop.getInductionVar();
  
  // Global index
  Value blockOffset = builder.create<arith::MulIOp>(loc, blockIdx, c256);
  Value i = builder.create<arith::AddIOp>(loc, blockOffset, threadIdx);
  
  // Bounds check
  Value inBounds = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, i, N);
  auto ifOp = builder.create<scf::IfOp>(loc, inBounds, false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  
  // Load, add, store
  Value xVal = builder.create<memref::LoadOp>(loc, x, ValueRange{i});
  Value yVal = builder.create<memref::LoadOp>(loc, y, ValueRange{i});
  Value sum = builder.create<arith::AddFOp>(loc, xVal, yVal);
  builder.create<memref::StoreOp>(loc, sum, z, ValueRange{i});
  
  // Reset insertion point after loops
  builder.setInsertionPointAfter(blockLoop);
}

// Element-wise multiplication: z[i] = x[i] * y[i]
void buildElementwiseMul(OpBuilder &builder, Location loc,
                         Value x, Value y, Value z, Value N) {
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c256 = builder.create<arith::ConstantIndexOp>(loc, 256);
  Value c255 = builder.create<arith::ConstantIndexOp>(loc, 255);
  
  Value N_plus_255 = builder.create<arith::AddIOp>(loc, N, c255);
  Value numBlocks = builder.create<arith::DivUIOp>(loc, N_plus_255, c256);
  
  // Blocks
  auto blockLoop = builder.create<scf::ForOp>(loc, c0, numBlocks, c1);
  builder.setInsertionPointToStart(blockLoop.getBody());
  Value blockIdx = blockLoop.getInductionVar();
  
  // Threads
  auto threadLoop = builder.create<scf::ForOp>(loc, c0, c256, c1);
  builder.setInsertionPointToStart(threadLoop.getBody());
  Value threadIdx = threadLoop.getInductionVar();
  
  // Global index
  Value blockOffset = builder.create<arith::MulIOp>(loc, blockIdx, c256);
  Value i = builder.create<arith::AddIOp>(loc, blockOffset, threadIdx);
  
  // Bounds check
  Value inBounds = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, i, N);
  auto ifOp = builder.create<scf::IfOp>(loc, inBounds, false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  
  // Load, multiply, store
  Value xVal = builder.create<memref::LoadOp>(loc, x, ValueRange{i});
  Value yVal = builder.create<memref::LoadOp>(loc, y, ValueRange{i});
  Value prod = builder.create<arith::MulFOp>(loc, xVal, yVal);
  builder.create<memref::StoreOp>(loc, prod, z, ValueRange{i});
  
  // Reset insertion point after loops
  builder.setInsertionPointAfter(blockLoop);
}

// ============================================================================
// Phase 3: Softmax - Block-level Reductions and Synchronization
// ============================================================================

// Softmax Pass 1: Find maximum value (block-level reduction)
// Each block computes max over its chunk using reduction pattern
// Returns the global max value across all elements
//
// GPU Concept: gpu.all_reduce(max) with gpu.barrier synchronization
// CPU Emulation: Nested loops with scf.reduce pattern
void buildSoftmaxMaxReduction(OpBuilder &builder, Location loc,
                              Value input, Value maxValue, Value N,
                              Value cNegInf) {
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c256 = builder.create<arith::ConstantIndexOp>(loc, 256);
  Value c255 = builder.create<arith::ConstantIndexOp>(loc, 255);
  
  Value N_plus_255 = builder.create<arith::AddIOp>(loc, N, c255);
  Value numBlocks = builder.create<arith::DivUIOp>(loc, N_plus_255, c256);
  
  // Initialize maxValue[0] = -inf
  builder.create<memref::StoreOp>(loc, cNegInf, maxValue, ValueRange{c0});
  
  // Block loop: each block finds local max
  auto blockLoop = builder.create<scf::ForOp>(loc, c0, numBlocks, c1);
  builder.setInsertionPointToStart(blockLoop.getBody());
  Value blockIdx = blockLoop.getInductionVar();
  
  // Thread loop: each thread finds its max
  auto threadLoop = builder.create<scf::ForOp>(loc, c0, c256, c1);
  builder.setInsertionPointToStart(threadLoop.getBody());
  Value threadIdx = threadLoop.getInductionVar();
  
  // Global index
  Value blockOffset = builder.create<arith::MulIOp>(loc, blockIdx, c256);
  Value i = builder.create<arith::AddIOp>(loc, blockOffset, threadIdx);
  
  // Bounds check
  Value inBounds = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, i, N);
  auto ifOp = builder.create<scf::IfOp>(loc, inBounds, false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  
  // Load current value
  Value val = builder.create<memref::LoadOp>(loc, input, ValueRange{i});
  
  // Update global max (emulating atomic max)
  Value currentMax = builder.create<memref::LoadOp>(loc, maxValue, ValueRange{c0});
  Value newMax = builder.create<arith::MaximumFOp>(loc, currentMax, val);
  builder.create<memref::StoreOp>(loc, newMax, maxValue, ValueRange{c0});
  
  // Reset insertion point after loops
  builder.setInsertionPointAfter(blockLoop);
}

// Softmax Pass 2: Compute exp(x - max) and sum reduction
// For numerical stability, subtract max before exp
// 
// GPU Concept: gpu.all_reduce(sum) with gpu.barrier
// CPU Emulation: Nested loops with accumulation
void buildSoftmaxExpSum(OpBuilder &builder, Location loc,
                        Value input, Value expOutput, Value sumValue, Value maxValue, Value N,
                        Value cZero, Value c1f, Value c2f, Value c6f, Value c24f, Value c120f) {
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c256 = builder.create<arith::ConstantIndexOp>(loc, 256);
  Value c255 = builder.create<arith::ConstantIndexOp>(loc, 255);
  
  Value N_plus_255 = builder.create<arith::AddIOp>(loc, N, c255);
  Value numBlocks = builder.create<arith::DivUIOp>(loc, N_plus_255, c256);
  
  // Initialize sum[0] = 0 (cZero passed as argument)
  builder.create<memref::StoreOp>(loc, cZero, sumValue, ValueRange{c0});
  
  // Load the global max (computed in Pass 1)
  Value max = builder.create<memref::LoadOp>(loc, maxValue, ValueRange{c0});
  
  // Block loop
  auto blockLoop = builder.create<scf::ForOp>(loc, c0, numBlocks, c1);
  builder.setInsertionPointToStart(blockLoop.getBody());
  Value blockIdx = blockLoop.getInductionVar();
  
  // Thread loop
  auto threadLoop = builder.create<scf::ForOp>(loc, c0, c256, c1);
  builder.setInsertionPointToStart(threadLoop.getBody());
  Value threadIdx = threadLoop.getInductionVar();
  
  // Global index
  Value blockOffset = builder.create<arith::MulIOp>(loc, blockIdx, c256);
  Value i = builder.create<arith::AddIOp>(loc, blockOffset, threadIdx);
  
  // Bounds check
  Value inBounds = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, i, N);
  auto ifOp = builder.create<scf::IfOp>(loc, inBounds, false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  
  // Compute exp(x[i] - max)
  Value x = builder.create<memref::LoadOp>(loc, input, ValueRange{i});
  Value xMinusMax = builder.create<arith::SubFOp>(loc, x, max);
  
  // Compute exp using Taylor series: exp(x) ≈ 1 + x + x²/2! + x³/3! + x⁴/4! + x⁵/5!
  // All constants (1.0, 2.0, 6.0, 24.0, 120.0) passed as function arguments
  
  Value x2 = builder.create<arith::MulFOp>(loc, xMinusMax, xMinusMax);
  Value x3 = builder.create<arith::MulFOp>(loc, x2, xMinusMax);
  Value x4 = builder.create<arith::MulFOp>(loc, x3, xMinusMax);
  Value x5 = builder.create<arith::MulFOp>(loc, x4, xMinusMax);
  
  Value term1 = xMinusMax;
  Value term2 = builder.create<arith::DivFOp>(loc, x2, c2f);
  Value term3 = builder.create<arith::DivFOp>(loc, x3, c6f);
  Value term4 = builder.create<arith::DivFOp>(loc, x4, c24f);
  Value term5 = builder.create<arith::DivFOp>(loc, x5, c120f);
  
  Value expVal = builder.create<arith::AddFOp>(loc, c1f, term1);
  expVal = builder.create<arith::AddFOp>(loc, expVal, term2);
  expVal = builder.create<arith::AddFOp>(loc, expVal, term3);
  expVal = builder.create<arith::AddFOp>(loc, expVal, term4);
  expVal = builder.create<arith::AddFOp>(loc, expVal, term5);
  
  // Store exp value
  builder.create<memref::StoreOp>(loc, expVal, expOutput, ValueRange{i});
  
  // Accumulate to sum (emulating atomic add)
  Value currentSum = builder.create<memref::LoadOp>(loc, sumValue, ValueRange{c0});
  Value newSum = builder.create<arith::AddFOp>(loc, currentSum, expVal);
  builder.create<memref::StoreOp>(loc, newSum, sumValue, ValueRange{c0});
  
  // Reset insertion point after loops
  builder.setInsertionPointAfter(blockLoop);
}

// Softmax Pass 3: Normalize exp values by sum
// output[i] = exp[i] / sum
void buildSoftmaxNormalize(OpBuilder &builder, Location loc,
                           Value expValues, Value output, Value sumValue, Value N) {
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c256 = builder.create<arith::ConstantIndexOp>(loc, 256);
  Value c255 = builder.create<arith::ConstantIndexOp>(loc, 255);
  
  Value N_plus_255 = builder.create<arith::AddIOp>(loc, N, c255);
  Value numBlocks = builder.create<arith::DivUIOp>(loc, N_plus_255, c256);
  
  // Load the sum (computed in Pass 2)
  Value sum = builder.create<memref::LoadOp>(loc, sumValue, ValueRange{c0});
  
  // Block loop
  auto blockLoop = builder.create<scf::ForOp>(loc, c0, numBlocks, c1);
  builder.setInsertionPointToStart(blockLoop.getBody());
  Value blockIdx = blockLoop.getInductionVar();
  
  // Thread loop
  auto threadLoop = builder.create<scf::ForOp>(loc, c0, c256, c1);
  builder.setInsertionPointToStart(threadLoop.getBody());
  Value threadIdx = threadLoop.getInductionVar();
  
  // Global index
  Value blockOffset = builder.create<arith::MulIOp>(loc, blockIdx, c256);
  Value i = builder.create<arith::AddIOp>(loc, blockOffset, threadIdx);
  
  // Bounds check
  Value inBounds = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, i, N);
  auto ifOp = builder.create<scf::IfOp>(loc, inBounds, false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  
  // Normalize: output[i] = exp[i] / sum
  Value expVal = builder.create<memref::LoadOp>(loc, expValues, ValueRange{i});
  Value normalized = builder.create<arith::DivFOp>(loc, expVal, sum);
  builder.create<memref::StoreOp>(loc, normalized, output, ValueRange{i});
  
  // Reset insertion point after loops
  builder.setInsertionPointAfter(blockLoop);
}

// Configure lowering pipeline: SCF loops → CF → LLVM
// Note: We're emulating GPU concepts with SCF loops directly,
// so we don't need GPU dialect lowering passes
void configureLoweringPipeline(PassManager &pm) {
  // Phase 1: Lower SCF to Control Flow
  pm.addPass(createConvertSCFToCFPass());
  
  // Phase 1.5: Clean up messy control flow (MLIR 19 bug fix)
  // Fixes "daisy-chain" llvm.mlir.cast bug that causes infinite loops
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  
  // Phase 2: Lower remaining dialects to LLVM
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

// JIT compile and execute
class GPUEmulator {
private:
  MLIRContext context;
  std::unique_ptr<ExecutionEngine> engine;

public:
  GPUEmulator() {
    // Register dialects
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();
    context.getOrLoadDialect<LLVM::LLVMDialect>();
    context.getOrLoadDialect<cf::ControlFlowDialect>();

    // Register translations
    registerBuiltinDialectTranslation(context);
    registerLLVMDialectTranslation(context);

    // Initialize LLVM
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }

  // Vector addition: C = A + B
  py::array_t<float> vectorAdd(py::array_t<float> pyA, py::array_t<float> pyB) {
    auto bufA = pyA.request();
    auto bufB = pyB.request();

    if (bufA.ndim != 1 || bufB.ndim != 1)
      throw std::runtime_error("Expected 1D arrays");
    if (bufA.shape[0] != bufB.shape[0])
      throw std::runtime_error("Arrays must have same length");

    int64_t N = bufA.shape[0];

    // Allocate output
    auto pyC = py::array_t<float>(N);
    auto bufC = pyC.request();

    // Build IR
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());

    // Function signature: (memref<Nxf32>, memref<Nxf32>, memref<Nxf32>) -> ()
    auto memrefType = MemRefType::get({N}, builder.getF32Type());
    auto funcType = builder.getFunctionType(
      {memrefType, memrefType, memrefType}, {});

    auto func = builder.create<func::FuncOp>(loc, "vector_add", funcType);
    func.setPrivate();
    // Use C interface for proper memref ABI
    func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());

    Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    Value A = entryBlock->getArgument(0);
    Value B = entryBlock->getArgument(1);
    Value C = entryBlock->getArgument(2);
    Value size = builder.create<arith::ConstantIndexOp>(loc, N);

    // Build GPU kernel
    buildVectorAddGPU(builder, loc, A, B, C, size);

    // Reset insertion point to end of function (after all the loops)
    builder.setInsertionPointToEnd(entryBlock);
    builder.create<func::ReturnOp>(loc);

    // Verify module
    if (failed(verify(module))) {
      module.dump();
      throw std::runtime_error("Module verification failed");
    }

    // Apply lowering passes
    PassManager pm(&context);
    configureLoweringPipeline(pm);

    if (failed(pm.run(module))) {
      module.dump();
      throw std::runtime_error("Lowering failed");
    }
        // DEBUG: Dump LLVM IR to see what's actually being generated
    llvm::errs() << "=== LLVM IR after lowering ===\n";
    module->print(llvm::errs());
    llvm::errs() << "=== End LLVM IR ===\n";
        // Create execution engine
    ExecutionEngineOptions options;
    options.transformer = mlir::makeOptimizingTransformer(3, 0, nullptr);

    auto maybeEngine = ExecutionEngine::create(module, options);
    if (!maybeEngine) {
      throw std::runtime_error("Failed to create execution engine");
    }
    engine = std::move(*maybeEngine);

    // Use C wrapper interface for proper memref ABI
    // Define descriptor struct matching MLIR's 1D memref layout
    struct MemRefDescriptor1D {
      float *allocated;
      float *aligned;
      int64_t offset;
      int64_t size;
      int64_t stride;
    };

    auto* A_ptr = static_cast<float*>(bufA.ptr);
    auto* B_ptr = static_cast<float*>(bufB.ptr);
    auto* C_ptr = static_cast<float*>(bufC.ptr);

    MemRefDescriptor1D descA = {A_ptr, A_ptr, 0, N, 1};
    MemRefDescriptor1D descB = {B_ptr, B_ptr, 0, N, 1};
    MemRefDescriptor1D descC = {C_ptr, C_ptr, 0, N, 1};

    // Lookup C wrapper function (has _mlir_ciface_ prefix)
    using FnPtr = void(*)(MemRefDescriptor1D*, MemRefDescriptor1D*, MemRefDescriptor1D*);
    auto expectedFPtr = engine->lookup("_mlir_ciface_vector_add");
    if (!expectedFPtr) {
      throw std::runtime_error("Failed to lookup C interface function");
    }

    auto* vector_add_fn = reinterpret_cast<FnPtr>(*expectedFPtr);
    vector_add_fn(&descA, &descB, &descC);

    return pyC;
  }

  // Test thread indexing
  py::array_t<float> testIndexing(int64_t N, int64_t targetIndex) {
    auto pyOutput = py::array_t<float>(N);
    auto bufOutput = pyOutput.request();

    // Initialize to zeros
    std::fill_n((float*)bufOutput.ptr, N, 0.0f);

    // Build IR
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());

    auto memrefType = MemRefType::get({N}, builder.getF32Type());
    // Add float arguments to pass 1.0 and 0.0 as parameters (workaround for constant bug)
    auto funcType = builder.getFunctionType({memrefType, builder.getF32Type(), builder.getF32Type()}, {});

    auto func = builder.create<func::FuncOp>(loc, "test_indexing", funcType);
    func.setPrivate();
    // Use C interface for proper memref ABI
    func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());

    Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    Value output = entryBlock->getArgument(0);
    Value one = entryBlock->getArgument(1);    // Get 1.0 from function arg
    Value zero = entryBlock->getArgument(2);   // Get 0.0 from function arg
    Value size = builder.create<arith::ConstantIndexOp>(loc, N);
    Value target = builder.create<arith::ConstantIndexOp>(loc, targetIndex);

    buildThreadIndexingTest(builder, loc, output, size, target, one, zero);

    // Reset insertion point to end of function
    builder.setInsertionPointToEnd(entryBlock);
    builder.create<func::ReturnOp>(loc);

    if (failed(verify(module))) {
      module.dump();
      throw std::runtime_error("Module verification failed");
    }

    // DEBUG: Dump IR before lowering
    llvm::errs() << "\n=== MLIR before lowering ===\n";
    module->print(llvm::errs());
    llvm::errs() << "=== End MLIR ===\n\n";

    PassManager pm(&context);
    configureLoweringPipeline(pm);

    if (failed(pm.run(module))) {
      module.dump();
      throw std::runtime_error("Lowering failed");
    }

    // DEBUG: Dump LLVM IR after lowering
    llvm::errs() << "\n=== LLVM IR after lowering ===\n";
    module->print(llvm::errs());
    llvm::errs() << "=== End LLVM IR ===\n\n";

    ExecutionEngineOptions options;
    // Try with -O2 optimization (default is -O2)
    options.transformer = mlir::makeOptimizingTransformer(2, 0, nullptr);

    auto maybeEngine = ExecutionEngine::create(module, options);
    if (!maybeEngine) {
      throw std::runtime_error("Failed to create execution engine");
    }
    engine = std::move(*maybeEngine);

    // Use C wrapper interface for proper memref ABI
    // Define descriptor struct matching MLIR's 1D memref layout
    struct MemRefDescriptor1D {
      float *allocated;
      float *aligned;
      int64_t offset;
      int64_t size;
      int64_t stride;
    };

    auto* output_ptr = static_cast<float*>(bufOutput.ptr);
    MemRefDescriptor1D desc = {output_ptr, output_ptr, 0, N, 1};

    // Lookup C wrapper function (has _mlir_ciface_ prefix)
    // Signature: void func(MemRefDescriptor1D*, float, float)
    using FnPtr = void(*)(MemRefDescriptor1D*, float, float);
    auto expectedFPtr = engine->lookup("_mlir_ciface_test_indexing");
    if (!expectedFPtr) {
      // Try without prefix as fallback
      expectedFPtr = engine->lookup("test_indexing");
      if (!expectedFPtr) {
        throw std::runtime_error("Failed to lookup C interface function (tried both _mlir_ciface_test_indexing and test_indexing)");
      }
      llvm::errs() << "Warning: Using direct function call instead of C interface\n";
    } else {
      llvm::errs() << "Found C interface function: _mlir_ciface_test_indexing\n";
    }

    auto* test_indexing_fn = reinterpret_cast<FnPtr>(*expectedFPtr);
    // Pass 1.0 and 0.0 as function arguments (workaround for MLIR constant bug)
    test_indexing_fn(&desc, 1.0f, 0.0f);

    return pyOutput;
  }

  // Matrix multiplication: C = A @ B
  py::array_t<float> matmul(py::array_t<float> pyA, py::array_t<float> pyB) {
    auto bufA = pyA.request();
    auto bufB = pyB.request();
    
    if (bufA.ndim != 2 || bufB.ndim != 2)
      throw std::runtime_error("Expected 2D arrays for matrix multiplication");
    if (bufA.shape[1] != bufB.shape[0])
      throw std::runtime_error("Matrix dimensions incompatible: A cols must equal B rows");
    
    int64_t M = bufA.shape[0];  // A rows
    int64_t K = bufA.shape[1];  // A cols = B rows
    int64_t N = bufB.shape[1];  // B cols
    
    // Ensure input arrays are contiguous
    py::array_t<float> pyA_contig = py::array_t<float, py::array::c_style | py::array::forcecast>(pyA);
    py::array_t<float> pyB_contig = py::array_t<float, py::array::c_style | py::array::forcecast>(pyB);
    bufA = pyA_contig.request();
    bufB = pyB_contig.request();
    
    // Allocate output C (M × N)
    auto pyC = py::array_t<float>({M, N});
    auto bufC = pyC.request();
    
    // Build IR
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());
    
    // Create function signature with DYNAMIC shapes: (A: ?x?, B: ?x?, C: ?x?) -> ()
    // NO init parameter - just like Chapter 2!
    auto memrefTypeA = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, builder.getF32Type());
    auto memrefTypeB = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, builder.getF32Type());
    auto memrefTypeC = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, builder.getF32Type());
    auto funcType = builder.getFunctionType({
      memrefTypeA, memrefTypeB, memrefTypeC
    }, {});
    
    auto func = builder.create<func::FuncOp>(loc, "matmul", funcType);
    func.setPublic();  // Make it public so ExecutionEngine can find it!
    func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    
    Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    Value A = entryBlock->getArgument(0);
    Value B = entryBlock->getArgument(1);
    Value C = entryBlock->getArgument(2);
    Value initValue = builder.create<arith::ConstantFloatOp>(loc, APFloat(0.0f), builder.getF32Type());
    
    Value sizeM = builder.create<arith::ConstantIndexOp>(loc, M);
    Value sizeN = builder.create<arith::ConstantIndexOp>(loc, N);
    Value sizeK = builder.create<arith::ConstantIndexOp>(loc, K);
    
    buildMatMulGPU(builder, loc, A, B, C, sizeM, sizeN, sizeK, initValue);
    
    // Return
    builder.setInsertionPointToEnd(entryBlock);
    builder.create<func::ReturnOp>(loc);
    
    if (failed(verify(module))) {
      llvm::errs() << "ERROR: Module verification failed!\n";
      module.dump();
      throw std::runtime_error("Module verification failed");
    }
    
    llvm::errs() << "\n=== MatMul MLIR (before lowering) ===\n";
    module->print(llvm::errs());
    llvm::errs() << "=== End MLIR ===\n\n";
    
    llvm::errs() << "Starting lowering passes...\n";
    PassManager pm(&context);
    configureLoweringPipeline(pm);
    
    if (failed(pm.run(module))) {
      llvm::errs() << "ERROR: Lowering failed!\n";
      module.dump();
      throw std::runtime_error("Lowering failed");
    }
    llvm::errs() << "Lowering complete.\n";
    
    llvm::errs() << "\n=== LLVM Dialect (after lowering) ===\n";
    module->print(llvm::errs());
    llvm::errs() << "\n=== End LLVM Dialect ===\n\n";
    
    llvm::errs() << "Creating execution engine...\n";
    ExecutionEngineOptions options;
    options.transformer = mlir::makeOptimizingTransformer(0, 0, nullptr);  // O0 to avoid optimization bugs
    
    auto maybeEngine = ExecutionEngine::create(module.getOperation(), options);  // Pass Operation*
    if (!maybeEngine) {
      throw std::runtime_error("Failed to create execution engine");
    }
    engine = std::move(*maybeEngine);
    llvm::errs() << "Execution engine created.\n";
    
    // Define 2D memref descriptor
    struct MemRefDescriptor2D {
      float *allocated;
      float *aligned;
      int64_t offset;
      int64_t sizes[2];
      int64_t strides[2];
    };
    
    auto* A_ptr = static_cast<float*>(bufA.ptr);
    auto* B_ptr = static_cast<float*>(bufB.ptr);
    auto* C_ptr = static_cast<float*>(bufC.ptr);
    
    // Compute strides in elements (bufX.strides is in bytes, divide by sizeof(float))
    int64_t strideA0 = bufA.strides[0] / sizeof(float);
    int64_t strideA1 = bufA.strides[1] / sizeof(float);
    int64_t strideB0 = bufB.strides[0] / sizeof(float);
    int64_t strideB1 = bufB.strides[1] / sizeof(float);
    int64_t strideC0 = bufC.strides[0] / sizeof(float);
    int64_t strideC1 = bufC.strides[1] / sizeof(float);
    
    // Create descriptors
    MemRefDescriptor2D descA = {A_ptr, A_ptr, 0, {M, K}, {strideA0, strideA1}};
    MemRefDescriptor2D descB = {B_ptr, B_ptr, 0, {K, N}, {strideB0, strideB1}};
    MemRefDescriptor2D descC = {C_ptr, C_ptr, 0, {M, N}, {strideC0, strideC1}};
    
    llvm::errs() << "Looking up matmul function...\n";
    
    // Function signature: expanded memref descriptors (21 arguments: 7+7+7)
    // Each memref<?x?xf32> expands to: ptr, ptr, offset, size0, size1, stride0, stride1
    using FnPtr = void(*)(
        float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // A
        float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // B
        float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t   // C
    );
    
    auto expectedFPtr = engine->lookup("matmul");
    if (!expectedFPtr) {
      llvm::errs() << "Lookup failed: " << expectedFPtr.takeError() << "\n";
      throw std::runtime_error("Failed to lookup matmul function");
    }
    llvm::errs() << "Found matmul function.\n";
    
    auto* matmul_fn = reinterpret_cast<FnPtr>(*expectedFPtr);
    llvm::errs() << "Calling matmul...\n";
    
    // Pass memref descriptors as expanded arguments (like Chapter 2)
    matmul_fn(
        descA.allocated, descA.aligned, descA.offset, 
        descA.sizes[0], descA.sizes[1], descA.strides[0], descA.strides[1],  // A: M×K
        
        descB.allocated, descB.aligned, descB.offset,
        descB.sizes[0], descB.sizes[1], descB.strides[0], descB.strides[1],  // B: K×N
        
        descC.allocated, descC.aligned, descC.offset,
        descC.sizes[0], descC.sizes[1], descC.strides[0], descC.strides[1]   // C: M×N
    );
    llvm::errs() << "Matmul completed successfully.\n";
    
    return pyC;
  }
  
  // ========================================================================
  // Phase 2: Element-wise Operations
  // ========================================================================
  
  // GELU activation
  // TEST K: Pass float constants as function arguments (bypass constant pool alignment bug)
  py::array_t<float> gelu(py::array_t<float> pyInput) {
    auto bufInput = pyInput.request();
    
    if (bufInput.ndim != 1)
      throw std::runtime_error("Input must be 1D");
    
    int64_t N = bufInput.shape[0];
    
    // Create output array
    py::array_t<float> pyOutput(N);
    auto bufOutput = pyOutput.request();
    
    // TEST K: Create FRESH context - don't reuse class member!
    MLIRContext freshContext;
    freshContext.getOrLoadDialect<arith::ArithDialect>();
    freshContext.getOrLoadDialect<func::FuncDialect>();
    freshContext.getOrLoadDialect<memref::MemRefDialect>();
    freshContext.getOrLoadDialect<scf::SCFDialect>();
    freshContext.getOrLoadDialect<LLVM::LLVMDialect>();
    freshContext.getOrLoadDialect<cf::ControlFlowDialect>();
    registerBuiltinDialectTranslation(freshContext);
    registerLLVMDialectTranslation(freshContext);
    
    // Build MLIR module with fresh context
    OpBuilder builder(&freshContext);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // TEST K: Pass ALL GELU constants as function parameters (not arith.constant!)
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, builder.getF32Type());
    auto f32Type = builder.getF32Type();
    auto funcType = builder.getFunctionType({memrefType, memrefType, f32Type, f32Type, f32Type, f32Type, f32Type, f32Type}, {});
    
    // Use UNIQUE function name to avoid symbol conflicts
    auto func = builder.create<func::FuncOp>(loc, "test_k_unique_2024", funcType);
    func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    
    Block *entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    
    Value input = entry->getArgument(0);
    Value output = entry->getArgument(1);
    Value c_half = entry->getArgument(2);           // 0.5
    Value c_sqrt_2_over_pi = entry->getArgument(3); // sqrt(2/pi) = 0.7978845608
    Value c_coeff = entry->getArgument(4);          // 0.044715
    Value c_one = entry->getArgument(5);            // 1.0
    Value c_27 = entry->getArgument(6);             // 27.0 (for tanh approx)
    Value c_9 = entry->getArgument(7);              // 9.0 (for tanh approx)
    Value N_val = builder.create<arith::ConstantIndexOp>(loc, N);
    
    // Build GELU computation with all constants passed as arguments
    buildGELU(builder, loc, input, output, N_val, c_half, c_sqrt_2_over_pi, c_coeff, c_one, c_27, c_9);
    
    builder.create<func::ReturnOp>(loc);
    
    // Lower and execute
    PassManager pm(&freshContext);
    configureLoweringPipeline(pm);
    
    if (failed(pm.run(module))) {
      throw std::runtime_error("Lowering failed");
    }
    
    // JIT compile (disable LLVM IR dump for cleaner output)
    mlir::ExecutionEngineOptions options;
    options.transformer = mlir::makeOptimizingTransformer(0, 0, nullptr);
    auto maybeEngine = mlir::ExecutionEngine::create(module, options);
    
    if (!maybeEngine) {
      throw std::runtime_error("Failed to create execution engine");
    }
    
    auto engine = std::move(*maybeEngine);
    
    // Prepare memref descriptors
    struct MemRefDescriptor1D {
      float *allocated;
      float *aligned;
      int64_t offset;
      int64_t size;
      int64_t stride;
    };
    
    MemRefDescriptor1D descInput = {
      static_cast<float*>(bufInput.ptr),
      static_cast<float*>(bufInput.ptr),
      0, N, 1
    };
    
    MemRefDescriptor1D descOutput = {
      static_cast<float*>(bufOutput.ptr),
      static_cast<float*>(bufOutput.ptr),
      0, N, 1
    };
    
    // Call function with 8 arguments: 2 memrefs + 6 floats
    using FnPtr = void(*)(MemRefDescriptor1D*, MemRefDescriptor1D*, float, float, float, float, float, float);
    auto expectedFPtr = engine->lookup("_mlir_ciface_test_k_unique_2024");
    if (!expectedFPtr) {
      throw std::runtime_error("Function lookup failed");
    }
    auto gelu_fn = reinterpret_cast<FnPtr>(*expectedFPtr);
    
    // Pass ALL GELU constants: 0.5, sqrt(2/pi), 0.044715, 1.0, 27.0, 9.0
    gelu_fn(&descInput, &descOutput, 0.5f, 0.7978845608f, 0.044715f, 1.0f, 27.0f, 9.0f);
    
    return pyOutput;
  }
  
  // Element-wise addition
  py::array_t<float> add(py::array_t<float> pyX, py::array_t<float> pyY) {
    auto bufX = pyX.request();
    auto bufY = pyY.request();
    
    if (bufX.ndim != 1 || bufY.ndim != 1)
      throw std::runtime_error("Inputs must be 1D");
    if (bufX.shape[0] != bufY.shape[0])
      throw std::runtime_error("Input sizes must match");
    
    int64_t N = bufX.shape[0];
    
    // Create output
    py::array_t<float> pyZ(N);
    auto bufZ = pyZ.request();
    
    // Build MLIR
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, builder.getF32Type());
    auto funcType = builder.getFunctionType({memrefType, memrefType, memrefType}, {});
    auto func = builder.create<func::FuncOp>(loc, "add", funcType);
    func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    
    Block *entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    
    Value x = entry->getArgument(0);
    Value y = entry->getArgument(1);
    Value z = entry->getArgument(2);
    Value N_val = builder.create<arith::ConstantIndexOp>(loc, N);
    
    buildElementwiseAdd(builder, loc, x, y, z, N_val);
    builder.create<func::ReturnOp>(loc);
    
    // Lower and execute
    PassManager pm(&context);
    configureLoweringPipeline(pm);
    
    if (failed(pm.run(module))) {
      throw std::runtime_error("Lowering failed");
    }
    
    mlir::ExecutionEngineOptions options;
    options.transformer = mlir::makeOptimizingTransformer(0, 0, nullptr);
    auto maybeEngine = mlir::ExecutionEngine::create(module, options);
    
    if (!maybeEngine) {
      throw std::runtime_error("Failed to create execution engine");
    }
    
    auto engine = std::move(*maybeEngine);
    
    struct MemRefDescriptor1D {
      float *allocated;
      float *aligned;
      int64_t offset;
      int64_t size;
      int64_t stride;
    };
    
    MemRefDescriptor1D descX = {static_cast<float*>(bufX.ptr), static_cast<float*>(bufX.ptr), 0, N, 1};
    MemRefDescriptor1D descY = {static_cast<float*>(bufY.ptr), static_cast<float*>(bufY.ptr), 0, N, 1};
    MemRefDescriptor1D descZ = {static_cast<float*>(bufZ.ptr), static_cast<float*>(bufZ.ptr), 0, N, 1};
    
    using FnPtr = void(*)(MemRefDescriptor1D*, MemRefDescriptor1D*, MemRefDescriptor1D*);
    auto expectedFPtr = engine->lookup("_mlir_ciface_add");
    if (!expectedFPtr) {
      throw std::runtime_error("Function lookup failed");
    }
    auto add_fn = reinterpret_cast<FnPtr>(*expectedFPtr);
    add_fn(&descX, &descY, &descZ);
    
    return pyZ;
  }
  
  // Element-wise multiplication
  py::array_t<float> mul(py::array_t<float> pyX, py::array_t<float> pyY) {
    auto bufX = pyX.request();
    auto bufY = pyY.request();
    
    if (bufX.ndim != 1 || bufY.ndim != 1)
      throw std::runtime_error("Inputs must be 1D");
    if (bufX.shape[0] != bufY.shape[0])
      throw std::runtime_error("Input sizes must match");
    
    int64_t N = bufX.shape[0];
    
    // Create output
    py::array_t<float> pyZ(N);
    auto bufZ = pyZ.request();
    
    // Build MLIR
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, builder.getF32Type());
    auto funcType = builder.getFunctionType({memrefType, memrefType, memrefType}, {});
    auto func = builder.create<func::FuncOp>(loc, "mul", funcType);
    func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    
    Block *entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    
    Value x = entry->getArgument(0);
    Value y = entry->getArgument(1);
    Value z = entry->getArgument(2);
    Value N_val = builder.create<arith::ConstantIndexOp>(loc, N);
    
    buildElementwiseMul(builder, loc, x, y, z, N_val);
    builder.create<func::ReturnOp>(loc);
    
    // Lower and execute
    PassManager pm(&context);
    configureLoweringPipeline(pm);
    
    if (failed(pm.run(module))) {
      throw std::runtime_error("Lowering failed");
    }
    
    mlir::ExecutionEngineOptions options;
    options.transformer = mlir::makeOptimizingTransformer(0, 0, nullptr);
    auto maybeEngine = mlir::ExecutionEngine::create(module, options);
    
    if (!maybeEngine) {
      throw std::runtime_error("Failed to create execution engine");
    }
    
    auto engine = std::move(*maybeEngine);
    
    struct MemRefDescriptor1D {
      float *allocated;
      float *aligned;
      int64_t offset;
      int64_t size;
      int64_t stride;
    };
    
    MemRefDescriptor1D descX = {static_cast<float*>(bufX.ptr), static_cast<float*>(bufX.ptr), 0, N, 1};
    MemRefDescriptor1D descY = {static_cast<float*>(bufY.ptr), static_cast<float*>(bufY.ptr), 0, N, 1};
    MemRefDescriptor1D descZ = {static_cast<float*>(bufZ.ptr), static_cast<float*>(bufZ.ptr), 0, N, 1};
    
    using FnPtr = void(*)(MemRefDescriptor1D*, MemRefDescriptor1D*, MemRefDescriptor1D*);
    auto expectedFPtr = engine->lookup("_mlir_ciface_mul");
    if (!expectedFPtr) {
      throw std::runtime_error("Function lookup failed");
    }
    auto mul_fn = reinterpret_cast<FnPtr>(*expectedFPtr);
    mul_fn(&descX, &descY, &descZ);
    
    return pyZ;
  }
  
  // ========================================================================
  // Phase 3: Softmax with Reductions
  // ========================================================================
  
  // Softmax: output = exp(x - max(x)) / sum(exp(x - max(x)))
  // Three-pass algorithm with block-level reductions
  py::array_t<float> softmax(py::array_t<float> pyInput) {
    auto bufInput = pyInput.request();
    
    if (bufInput.ndim != 1)
      throw std::runtime_error("Input must be 1D");
    
    int64_t N = bufInput.shape[0];
    
    // Create output array
    py::array_t<float> pyOutput(N);
    auto bufOutput = pyOutput.request();
    
    // Create fresh context (Lesson 3)
    MLIRContext freshContext;
    freshContext.getOrLoadDialect<arith::ArithDialect>();
    freshContext.getOrLoadDialect<func::FuncDialect>();
    freshContext.getOrLoadDialect<memref::MemRefDialect>();
    freshContext.getOrLoadDialect<scf::SCFDialect>();
    freshContext.getOrLoadDialect<LLVM::LLVMDialect>();
    freshContext.getOrLoadDialect<cf::ControlFlowDialect>();
    registerBuiltinDialectTranslation(freshContext);
    registerLLVMDialectTranslation(freshContext);
    
    // Build MLIR module
    OpBuilder builder(&freshContext);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Function signature: (input, output, max_scratch, exp_scratch, sum_scratch, neg_inf, zero, 1.0, 2.0, 6.0, 24.0, 120.0)
    // Pass ALL float constants as arguments to avoid MLIR 19 constant pool bug!
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, builder.getF32Type());
    auto scalarType = MemRefType::get({1}, builder.getF32Type()); // Scratch for max/sum
    auto f32Type = builder.getF32Type();
    auto funcType = builder.getFunctionType(
      {memrefType, memrefType, scalarType, memrefType, scalarType, f32Type, f32Type, f32Type, f32Type, f32Type, f32Type, f32Type}, {});
    
    // Unique function name (Lesson 4)
    auto func = builder.create<func::FuncOp>(loc, "softmax_phase3_2024", funcType);
    func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    
    Block *entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    
    Value input = entry->getArgument(0);
    Value output = entry->getArgument(1);
    Value maxScratch = entry->getArgument(2);
    Value expScratch = entry->getArgument(3);
    Value sumScratch = entry->getArgument(4);
    Value cNegInf = entry->getArgument(5); // Pass -inf as argument (Lesson 1)
    Value cZero = entry->getArgument(6);   // 0.0
    Value c1f = entry->getArgument(7);     // 1.0
    Value c2f = entry->getArgument(8);     // 2.0
    Value c6f = entry->getArgument(9);     // 6.0
    Value c24f = entry->getArgument(10);   // 24.0
    Value c120f = entry->getArgument(11);  // 120.0
    
    Value N_val = builder.create<arith::ConstantIndexOp>(loc, N);
    
    // Pass 1: Find max
    buildSoftmaxMaxReduction(builder, loc, input, maxScratch, N_val, cNegInf);
    
    // Pass 2: Compute exp(x - max) and sum (pass Taylor series constants)
    buildSoftmaxExpSum(builder, loc, input, expScratch, sumScratch, maxScratch, N_val,
                       cZero, c1f, c2f, c6f, c24f, c120f);
    
    // Pass 3: Normalize
    buildSoftmaxNormalize(builder, loc, expScratch, output, sumScratch, N_val);
    
    builder.create<func::ReturnOp>(loc);
    
    // Lower and execute
    PassManager pm(&freshContext);
    configureLoweringPipeline(pm);
    
    if (failed(pm.run(module))) {
      throw std::runtime_error("Lowering failed");
    }
    
    // JIT compile with O0 optimization (Lesson 2)
    mlir::ExecutionEngineOptions options;
    options.transformer = mlir::makeOptimizingTransformer(0, 0, nullptr);
    auto maybeEngine = mlir::ExecutionEngine::create(module, options);
    
    if (!maybeEngine) {
      throw std::runtime_error("Failed to create execution engine");
    }
    
    auto engine = std::move(*maybeEngine);
    
    // Prepare scratch buffers
    std::vector<float> maxBuf(1, -std::numeric_limits<float>::infinity());
    std::vector<float> expBuf(N);
    std::vector<float> sumBuf(1, 0.0f);
    
    // Memref descriptors
    struct MemRefDescriptor1D {
      float *allocated;
      float *aligned;
      int64_t offset;
      int64_t size;
      int64_t stride;
    };
    
    MemRefDescriptor1D descInput = {
      static_cast<float*>(bufInput.ptr),
      static_cast<float*>(bufInput.ptr),
      0, N, 1
    };
    
    MemRefDescriptor1D descOutput = {
      static_cast<float*>(bufOutput.ptr),
      static_cast<float*>(bufOutput.ptr),
      0, N, 1
    };
    
    MemRefDescriptor1D descMax = {
      maxBuf.data(), maxBuf.data(), 0, 1, 1
    };
    
    MemRefDescriptor1D descExp = {
      expBuf.data(), expBuf.data(), 0, N, 1
    };
    
    MemRefDescriptor1D descSum = {
      sumBuf.data(), sumBuf.data(), 0, 1, 1
    };
    
    float neg_inf = -std::numeric_limits<float>::infinity();
    
    // Pass all float constants as arguments (MLIR 19 workaround)
    using FnPtr = void(*)(MemRefDescriptor1D*, MemRefDescriptor1D*, MemRefDescriptor1D*,
                          MemRefDescriptor1D*, MemRefDescriptor1D*, float, float, float, float, float, float, float);
    auto expectedFPtr = engine->lookup("_mlir_ciface_softmax_phase3_2024");
    if (!expectedFPtr) {
      throw std::runtime_error("Function lookup failed");
    }
    auto softmax_fn = reinterpret_cast<FnPtr>(*expectedFPtr);
    softmax_fn(&descInput, &descOutput, &descMax, &descExp, &descSum, neg_inf, 0.0f, 1.0f, 2.0f, 6.0f, 24.0f, 120.0f);
    
    return pyOutput;
  }
};


PYBIND11_MODULE(ch15, m) {
  m.doc() = "Chapter 15: GPU Concepts with CPU Emulation (Direct SCF Loops)";

  py::class_<GPUEmulator>(m, "GPUEmulator")
    .def(py::init<>())
    .def("vector_add", &GPUEmulator::vectorAdd, "Vector addition using GPU concepts")
    .def("test_indexing", &GPUEmulator::testIndexing, "Test thread indexing")
    .def("matmul", &GPUEmulator::matmul, "Matrix multiplication with 2D GPU thread hierarchy")
    .def("gelu", &GPUEmulator::gelu, "GELU activation (element-wise)")
    .def("add", &GPUEmulator::add, "Element-wise addition")
    .def("mul", &GPUEmulator::mul, "Element-wise multiplication")
    .def("softmax", &GPUEmulator::softmax, "Softmax with block-level reductions");


  m.def("vector_add", [](py::array_t<float> a, py::array_t<float> b) {
    GPUEmulator emulator;
    return emulator.vectorAdd(a, b);
  }, "Convenience function for vector addition");

  m.def("test_indexing", [](int64_t N, int64_t targetIndex) {
    GPUEmulator emulator;
    return emulator.testIndexing(N, targetIndex);
  }, "Convenience function for testing indexing");

  m.def("matmul", [](py::array_t<float> a, py::array_t<float> b) {
    GPUEmulator emulator;
    return emulator.matmul(a, b);
  }, "Convenience function for matrix multiplication (C = A @ B)");
  
  m.def("gelu", [](py::array_t<float> x) {
    GPUEmulator emulator;
    return emulator.gelu(x);
  }, "GELU activation function");
  
  m.def("add", [](py::array_t<float> x, py::array_t<float> y) {
    GPUEmulator emulator;
    return emulator.add(x, y);
  }, "Element-wise addition");
  
  m.def("mul", [](py::array_t<float> x, py::array_t<float> y) {
    GPUEmulator emulator;
    return emulator.mul(x, y);
  }, "Element-wise multiplication");

  m.def("softmax", [](py::array_t<float> x) {
    GPUEmulator emulator;
    return emulator.softmax(x);
  }, "Softmax with block-level reductions");
}
