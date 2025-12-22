# Chapter 15: GPU Programming with MLIR (AOT Compilation)

**Learn GPU programming concepts through CPU emulation - no GPU hardware required!**

This tutorial implements a complete Nano-GPT transformer using MLIR, compiling ahead-of-time (AOT) to native code that runs on CPU by emulating GPU thread patterns.

## Table of Contents

- [Overview](#overview)
- [Why AOT Instead of JIT?](#why-aot-instead-of-jit)
- [GPU Concepts](#gpu-concepts)
- [Phase 0: Vector Operations](#phase-0-vector-operations-1d-parallelism)
- [Phase 1: Matrix Multiplication](#phase-1-matrix-multiplication-2d-parallelism)
- [Phase 2: Element-wise Operations](#phase-2-element-wise-operations)
- [Phase 3: Softmax](#phase-3-softmax-reductions)
- [Phase 4: Layer Normalization](#phase-4-layer-normalization)
- [Phase 5: Transpose](#phase-5-transpose-memory-patterns)
- [Phase 6: Attention Mechanism](#phase-6-attention-mechanism)
- [Phase 7: Complete Transformer](#phase-7-complete-transformer-nano-gpt)
- [Build and Test](#build-and-test)

---

## Overview

**Status:** ✅ **ALL 25 TESTS PASSING!** Nano-GPT complete with KV cache!

This chapter teaches GPU programming fundamentals using MLIR's SCF (Structured Control Flow) dialect to emulate GPU thread hierarchies on CPU. We progress from simple vector operations to a complete transformer architecture.

### What You'll Build

- **25 GPU kernels** across 7 phases
- **Complete GPT architecture:** embeddings, attention, FFN, layer norm
- **KV cache:** Efficient O(n) autoregressive generation
- **Production patterns:** Same concepts used in CUDA/ROCm programming

### Architecture: AOT Compilation

We use **Ahead-Of-Time (AOT)** compilation instead of JIT:

```
Build Time:  MLIR IR → LLVM Dialect → LLVM IR → Object File
Run Time:    Link executable → Execute (no compilation overhead)
```

**Benefits:**
- ✅ No JIT bugs (LLVM 20 ORC JIT had issues with LayerNorm)
- ✅ Faster execution (no runtime compilation)
- ✅ Better debugging (inspect assembly, use gdb)
- ✅ Production-ready (matches IREE, XLA, TVM)

---

## Why AOT Instead of JIT?

**Original Approach (Chapters 1-14):** JIT compilation with Python bindings

**Chapter 15 Switch to AOT:**

During development, we encountered an unfixable LLVM 20 ORC JIT bug where `engine->lookup()` would hang indefinitely on LayerNorm operations. After 21 different workaround attempts (see git history), we switched to AOT compilation.

**Result:** All 25 kernels now work perfectly, including LayerNorm! ✅

**Trade-offs:**
- ❌ Lost: Python integration, dynamic code generation
- ✅ Gained: Reliability, speed, debuggability, production architecture

---

## GPU Concepts

### Thread Hierarchy

GPUs organize computation in a 3-level hierarchy:

```
Grid (entire computation)
  └─ Blocks (fixed-size groups, typically 256 threads)
      └─ Threads (individual execution units)
```

**Example:** Process 10,000 elements with 256 threads per block
- Need ⌈10000/256⌉ = 40 blocks
- Each block has 256 threads
- Total: 40 × 256 = 10,240 thread slots (some unused)

### Index Calculation

Every thread computes its global index:

```cpp
globalIdx = blockIdx * blockSize + threadIdx
```

**Example:** Block 5, Thread 123, blockSize 256
```
globalIdx = 5 * 256 + 123 = 1403
```

### Bounds Checking (Critical!)

Grids don't align perfectly with data sizes:

```cpp
if (globalIdx < N) {  // ← REQUIRED!
  output[globalIdx] = process(input[globalIdx]);
}
```

Without this check, you access out-of-bounds memory → crashes or corruption.

### 2D Thread Hierarchy

For matrix operations, use 2D organization:

```
Block (2D):     Grid (2D):
┌───┬───┬───┐   ┌─────┬─────┬─────┐
│0,0│0,1│0,2│   │ B   │ B   │ B   │
├───┼───┼───┤   │(0,0)│(0,1)│(0,2)│
│1,0│1,1│1,2│   ├─────┼─────┼─────┤
├───┼───┼───┤   │ B   │ B   │ B   │
│2,0│2,1│2,2│   │(1,0)│(1,1)│(1,2)│
└───┴───┴───┘   └─────┴─────┴─────┘
```

**2D Index Calculation:**
```cpp
row = blockIdx.x * 16 + threadIdx.x
col = blockIdx.y * 16 + threadIdx.y
```

---

## Phase 0: Vector Operations (1D Parallelism)

### Goal

Establish AOT infrastructure and implement the simplest GPU kernel: vector addition.

**Tests:** 3/3 passing ✅

### Implementation

**File:** [src/vector_add.cpp](src/vector_add.cpp)

**Kernel:** `C[i] = A[i] + B[i]` for all i

```cpp
void buildVectorAddKernel(OpBuilder& builder, Location loc,
                          Value A, Value B, Value C, Value N) {
    Value c0 = createIndex(builder, loc, 0);
    Value c1 = createIndex(builder, loc, 1);
    Value c256 = createIndex(builder, loc, 256);
    Value c255 = createIndex(builder, loc, 255);

    // Grid size: ceil(N / 256)
    Value N_plus_255 = builder.create<arith::AddIOp>(loc, N, c255);
    Value numBlocks = builder.create<arith::DivUIOp>(loc, N_plus_255, c256);

    // Outer loop: blocks (emulate GPU grid)
    auto blockLoop = builder.create<scf::ForOp>(loc, c0, numBlocks, c1);
    builder.setInsertionPointToStart(blockLoop.getBody());
    Value blockIdx = blockLoop.getInductionVar();

    // Inner loop: threads (emulate GPU block)
    auto threadLoop = builder.create<scf::ForOp>(loc, c0, c256, c1);
    builder.setInsertionPointToStart(threadLoop.getBody());
    Value threadIdx = threadLoop.getInductionVar();

    // Compute global index
    Value blockOffset = builder.create<arith::MulIOp>(loc, blockIdx, c256);
    Value globalIdx = builder.create<arith::AddIOp>(loc, blockOffset, threadIdx);

    // Bounds check
    Value inBounds = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalIdx, N
    );

    auto ifOp = builder.create<scf::IfOp>(loc, inBounds, false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    // Load, compute, store
    Value a = builder.create<memref::LoadOp>(loc, A, globalIdx);
    Value b = builder.create<memref::LoadOp>(loc, B, globalIdx);
    Value sum = builder.create<arith::AddFOp>(loc, a, b);
    builder.create<memref::StoreOp>(loc, sum, C, globalIdx);
}
```

**Generated MLIR:**
```mlir
func.func @vector_add(%A: memref<?xf32>, %B: memref<?xf32>, 
                      %C: memref<?xf32>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  
  %numBlocks = arith.divui (arith.addi %N, %c255), %c256
  
  scf.for %blockIdx = %c0 to %numBlocks step %c1 {
    scf.for %threadIdx = %c0 to %c256 step %c1 {
      %globalIdx = arith.addi (arith.muli %blockIdx, %c256), %threadIdx
      %inBounds = arith.cmpi ult, %globalIdx, %N
      
      scf.if %inBounds {
        %a = memref.load %A[%globalIdx] : memref<?xf32>
        %b = memref.load %B[%globalIdx] : memref<?xf32>
        %c = arith.addf %a, %b : f32
        memref.store %c, %C[%globalIdx] : memref<?xf32>
      }
    }
  }
  return
}
```

### Test Results

```
test_vector_add (N=1024)... ✅ PASSED (max error: 0.00e+00)
test_vector_add_large (N=10000)... ✅ PASSED (max error: 0.00e+00)
test_vector_add_unaligned (N=1337)... ✅ PASSED (max error: 0.00e+00)
```

### Key Lessons

1. **Two nested loops** emulate GPU hierarchy (blocks × threads)
2. **Bounds checking is critical** for non-aligned sizes
3. **256 threads per block** is a common choice (GPU hardware typically supports 256-1024)
4. **AOT compilation is straightforward** - no JIT complexity!

---

## Phase 1: Matrix Multiplication (2D Parallelism)

### Goal

Implement 2D matrix multiplication using 2D thread blocks.

**Tests:** 3/3 passing ✅

### Implementation

**File:** [src/matmul.cpp](src/matmul.cpp)

**Kernel:** `C[i,j] = Σ A[i,k] × B[k,j]`

**Thread Organization:**
- Block size: 16×16 threads
- Grid size: ⌈M/16⌉ × ⌈N/16⌉ blocks
- Each thread computes one output element

```cpp
void buildMatMulKernel(OpBuilder& builder, Location loc,
                       Value A, Value B, Value C, Value M, Value N, Value K) {
    Value c16 = createIndex(builder, loc, 16);
    Value c15 = createIndex(builder, loc, 15);
    
    // Grid dimensions: ceil(M/16), ceil(N/16)
    Value gridDimX = builder.create<arith::DivUIOp>(loc,
        builder.create<arith::AddIOp>(loc, M, c15), c16);
    Value gridDimY = builder.create<arith::DivUIOp>(loc,
        builder.create<arith::AddIOp>(loc, N, c15), c16);
    
    // Outer loops: 2D grid (blockIdx.x, blockIdx.y)
    auto blockLoopX = builder.create<scf::ForOp>(loc, c0, gridDimX, c1);
    builder.setInsertionPointToStart(blockLoopX.getBody());
    Value blockX = blockLoopX.getInductionVar();
    
    auto blockLoopY = builder.create<scf::ForOp>(loc, c0, gridDimY, c1);
    builder.setInsertionPointToStart(blockLoopY.getBody());
    Value blockY = blockLoopY.getInductionVar();
    
    // Inner loops: 2D block (threadIdx.x, threadIdx.y)
    auto threadLoopX = builder.create<scf::ForOp>(loc, c0, c16, c1);
    builder.setInsertionPointToStart(threadLoopX.getBody());
    Value threadX = threadLoopX.getInductionVar();
    
    auto threadLoopY = builder.create<scf::ForOp>(loc, c0, c16, c1);
    builder.setInsertionPointToStart(threadLoopY.getBody());
    Value threadY = threadLoopY.getInductionVar();
    
    // Compute global indices
    Value row = builder.create<arith::AddIOp>(loc,
        builder.create<arith::MulIOp>(loc, blockX, c16), threadX);
    Value col = builder.create<arith::AddIOp>(loc,
        builder.create<arith::MulIOp>(loc, blockY, c16), threadY);
    
    // Bounds check
    Value rowValid = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, row, M);
    Value colValid = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, col, N);
    Value valid = builder.create<arith::AndIOp>(loc, rowValid, colValid);
    
    auto ifOp = builder.create<scf::IfOp>(loc, valid, false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    
    // Compute dot product: C[row,col] = Σ A[row,k] * B[k,col]
    Value sum_init = createFloat(builder, loc, 0.0f);
    auto sumLoop = builder.create<scf::ForOp>(loc, c0, K, c1,
                                               ValueRange{sum_init});
    builder.setInsertionPointToStart(sumLoop.getBody());
    Value k = sumLoop.getInductionVar();
    Value sum = sumLoop.getRegionIterArgs()[0];
    
    Value a_val = builder.create<memref::LoadOp>(loc, A, ValueRange{row, k});
    Value b_val = builder.create<memref::LoadOp>(loc, B, ValueRange{k, col});
    Value prod = builder.create<arith::MulFOp>(loc, a_val, b_val);
    Value new_sum = builder.create<arith::AddFOp>(loc, sum, prod);
    builder.create<scf::YieldOp>(loc, new_sum);
    
    // Store result
    builder.setInsertionPointAfter(sumLoop);
    Value finalSum = sumLoop.getResult(0);
    builder.create<memref::StoreOp>(loc, finalSum, C, ValueRange{row, col});
}
```

### Test Results

```
test_matmul_square (32×32)... ✅ PASSED (max error: 2.98e-07)
test_matmul_rectangular (64×128 @ 128×96)... ✅ PASSED (max error: 1.19e-06)
test_matmul_unaligned (33×47 @ 47×29)... ✅ PASSED (max error: 4.47e-07)
```

### Key Lessons

1. **4 nested loops:** 2D grid × 2D block = 4 dimensions
2. **Innermost reduction loop:** Each thread computes a full dot product
3. **2D bounds checking:** Must validate both row and col
4. **scf.for with loop-carried values:** Accumulate sum across k dimension

---

## Phase 2: Element-wise Operations

### Goal

Implement element-wise neural network operations: GELU, Add, BiasAdd.

**Tests:** 3/3 passing ✅

### GELU Activation

**File:** [src/elementwise.cpp](src/elementwise.cpp)

**Formula:** `GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`

**Implementation uses Math dialect:**

```cpp
void buildGELUKernel(OpBuilder& builder, Location loc,
                     Value input, Value output, Value N) {
    // Constants
    Value c_half = createFloat(builder, loc, 0.5f);
    Value c_one = createFloat(builder, loc, 1.0f);
    Value c_sqrt_2_over_pi = createFloat(builder, loc, 0.7978845608f);
    Value c_0_044715 = createFloat(builder, loc, 0.044715f);
    
    // Standard 1D grid (256 threads per block)
    // ... grid setup ...
    
    Value x = builder.create<memref::LoadOp>(loc, input, globalIdx);
    
    // Compute x + 0.044715 * x³
    Value x2 = builder.create<arith::MulFOp>(loc, x, x);
    Value x3 = builder.create<arith::MulFOp>(loc, x2, x);
    Value term1 = builder.create<arith::MulFOp>(loc, c_0_044715, x3);
    Value inner = builder.create<arith::AddFOp>(loc, x, term1);
    
    // Scale by √(2/π)
    Value scaled = builder.create<arith::MulFOp>(loc, c_sqrt_2_over_pi, inner);
    
    // math.tanh (requires MathToLibm lowering pass)
    Value tanh_val = builder.create<math::TanhOp>(loc, scaled);
    
    // Final: 0.5 * x * (1 + tanh(...))
    Value one_plus_tanh = builder.create<arith::AddFOp>(loc, c_one, tanh_val);
    Value x_times = builder.create<arith::MulFOp>(loc, x, one_plus_tanh);
    Value result = builder.create<arith::MulFOp>(loc, c_half, x_times);
    
    builder.create<memref::StoreOp>(loc, result, output, globalIdx);
}
```

**Critical:** Must add `MathToLibm` pass in lowering pipeline!

```cpp
// In common.cpp lowerToLLVMDialect():
pm.addPass(mlir::createConvertMathToLLVMPass());
pm.addPass(mlir::createConvertMathToLibmPass());  // ← Essential!
```

### Element-wise Add

Simple kernel demonstrating baseline pattern:

```cpp
void buildAddKernel(OpBuilder& builder, Location loc,
                    Value A, Value B, Value C, Value N) {
    // ... 1D grid ...
    Value a = builder.create<memref::LoadOp>(loc, A, globalIdx);
    Value b = builder.create<memref::LoadOp>(loc, B, globalIdx);
    Value sum = builder.create<arith::AddFOp>(loc, a, b);
    builder.create<memref::StoreOp>(loc, sum, C, globalIdx);
}
```

### BiasAdd (Scalar Broadcasting)

**Pattern:** Add scalar to all elements

```cpp
extern "C" void bias_add_kernel(float* input, float bias, 
                                float* output, int N) {
    // Note: bias is f32 scalar, not pointer!
    // Function signature: memref<?xf32>, f32, memref<?xf32>, index
}
```

### Test Results

```
test_gelu (N=1024)... ✅ PASSED (max error: 1.49e-07)
test_add (N=512)... ✅ PASSED (max error: 0.00e+00)
test_bias_add (N=768, bias=0.5)... ✅ PASSED (max error: 0.00e+00)
```

### Key Lessons

1. **Math dialect:** Use `math.tanh`, `math.exp`, etc. for library functions
2. **MathToLibm pass:** Required to lower to actual libm calls
3. **Scalar broadcasting:** Pass scalars as f32 arguments, not memrefs
4. **AOT reliability:** All work on first try (no JIT bugs!)

---

## Phase 3: Softmax (Reductions)

### Goal

Implement numerically stable softmax with multi-pass reduction pattern.

**Tests:** 3/3 passing ✅

### Algorithm

**Formula:** `softmax(x)ᵢ = exp(xᵢ - max(x)) / Σ exp(xⱼ - max(x))`

**Three passes:**
1. Find maximum value (for numerical stability)
2. Compute exp(x - max) and sum
3. Divide by sum (normalize)

### Implementation

**File:** [src/softmax.cpp](src/softmax.cpp)

**Pass 1: Find Maximum**

```cpp
void buildSoftmaxMaxReduction(OpBuilder& builder, Location loc,
                               Value input, Value max_out, Value N) {
    // Initialize max to -infinity
    Value neg_inf = createFloat(builder, loc, -1e38f);
    builder.create<memref::StoreOp>(loc, neg_inf, max_out, c0);
    
    // Sequential scan (reduction requires coordination)
    auto loop = builder.create<scf::ForOp>(loc, c0, N, c1);
    builder.setInsertionPointToStart(loop.getBody());
    Value i = loop.getInductionVar();
    
    Value x = builder.create<memref::LoadOp>(loc, input, i);
    Value current_max = builder.create<memref::LoadOp>(loc, max_out, c0);
    Value new_max = builder.create<arith::MaximumFOp>(loc, x, current_max);
    builder.create<memref::StoreOp>(loc, new_max, max_out, c0);
}
```

**Pass 2: Exp and Sum**

```cpp
void buildSoftmaxExpSum(OpBuilder& builder, Location loc,
                        Value input, Value exp_out, Value sum_out,
                        Value max_val, Value N) {
    // Initialize sum to 0
    Value zero = createFloat(builder, loc, 0.0f);
    builder.create<memref::StoreOp>(loc, zero, sum_out, c0);
    
    // Load max value (computed in pass 1)
    Value max_scalar = builder.create<memref::LoadOp>(loc, max_val, c0);
    
    auto loop = builder.create<scf::ForOp>(loc, c0, N, c1);
    builder.setInsertionPointToStart(loop.getBody());
    Value i = loop.getInductionVar();
    
    // Compute exp(x - max)
    Value x = builder.create<memref::LoadOp>(loc, input, i);
    Value x_shifted = builder.create<arith::SubFOp>(loc, x, max_scalar);
    Value exp_val = builder.create<math::ExpOp>(loc, x_shifted);
    
    // Store exp value
    builder.create<memref::StoreOp>(loc, exp_val, exp_out, i);
    
    // Accumulate sum
    Value current_sum = builder.create<memref::LoadOp>(loc, sum_out, c0);
    Value new_sum = builder.create<arith::AddFOp>(loc, current_sum, exp_val);
    builder.create<memref::StoreOp>(loc, new_sum, sum_out, c0);
}
```

**Pass 3: Normalize**

```cpp
void buildSoftmaxNormalize(OpBuilder& builder, Location loc,
                           Value exp_vals, Value output, 
                           Value sum_val, Value N) {
    Value sum_scalar = builder.create<memref::LoadOp>(loc, sum_val, c0);
    
    auto loop = builder.create<scf::ForOp>(loc, c0, N, c1);
    builder.setInsertionPointToStart(loop.getBody());
    Value i = loop.getInductionVar();
    
    Value exp_val = builder.create<memref::LoadOp>(loc, exp_vals, i);
    Value normalized = builder.create<arith::DivFOp>(loc, exp_val, sum_scalar);
    builder.create<memref::StoreOp>(loc, normalized, output, i);
}
```

### Test Results

```
test_softmax (N=8, uniform)... ✅ PASSED (sum=1.000, max error: 1.19e-07)
test_softmax_large (N=1024)... ✅ PASSED (sum=1.000, max error: 8.34e-07)
test_softmax_extreme (values: [-100, 100])... ✅ PASSED (no NaN/inf)
```

### Key Lessons

1. **Numerical stability:** Subtract max before exp prevents overflow
2. **Multi-pass pattern:** Some algorithms can't be parallelized trivially
3. **Reduction requires coordination:** Can't use independent threads
4. **Intermediate storage:** Need buffers for exp values
5. **Math dialect:** `math.ExpOp` lowers to libm's `expf()`

---

## Phase 4: Layer Normalization

### Goal

Implement layer normalization with learned scale/shift parameters.

**Tests:** 3/3 passing ✅ (This was the operation that hung in JIT!)

### Algorithm

**Formula:** `LN(x)ᵢ = γᵢ × (xᵢ - μ) / √(σ² + ε) + βᵢ`

Where:
- μ = mean
- σ² = variance
- γ = learned scale parameter
- β = learned shift parameter
- ε = small constant (1e-5) for numerical stability

**Three stages:**
1. Compute mean
2. Compute variance
3. Normalize and apply affine transform

### Implementation

**File:** [src/layernorm.cpp](src/layernorm.cpp)

**Stage 1: Mean**

```cpp
void buildLayerNormMean(OpBuilder& builder, Location loc,
                        Value input, Value mean_out, Value N) {
    // Initialize sum to 0
    Value zero = createFloat(builder, loc, 0.0f);
    builder.create<memref::StoreOp>(loc, zero, mean_out, c0);
    
    // Sum all values
    auto sumLoop = builder.create<scf::ForOp>(loc, c0, N, c1);
    builder.setInsertionPointToStart(sumLoop.getBody());
    Value i = sumLoop.getInductionVar();
    
    Value x = builder.create<memref::LoadOp>(loc, input, i);
    Value current_sum = builder.create<memref::LoadOp>(loc, mean_out, c0);
    Value new_sum = builder.create<arith::AddFOp>(loc, current_sum, x);
    builder.create<memref::StoreOp>(loc, new_sum, mean_out, c0);
    
    // Divide by N to get mean
    builder.setInsertionPointAfter(sumLoop);
    Value sum_val = builder.create<memref::LoadOp>(loc, mean_out, c0);
    Value N_float = builder.create<arith::IndexCastOp>(loc, i64Type, N);
    Value N_f32 = builder.create<arith::SIToFPOp>(loc, f32Type, N_float);
    Value mean = builder.create<arith::DivFOp>(loc, sum_val, N_f32);
    builder.create<memref::StoreOp>(loc, mean, mean_out, c0);
}
```

**Stage 2: Variance**

```cpp
void buildLayerNormVariance(OpBuilder& builder, Location loc,
                            Value input, Value mean_val, Value var_out, Value N) {
    Value mean_scalar = builder.create<memref::LoadOp>(loc, mean_val, c0);
    Value zero = createFloat(builder, loc, 0.0f);
    builder.create<memref::StoreOp>(loc, zero, var_out, c0);
    
    // Sum of squared differences
    auto loop = builder.create<scf::ForOp>(loc, c0, N, c1);
    builder.setInsertionPointToStart(loop.getBody());
    Value i = loop.getInductionVar();
    
    Value x = builder.create<memref::LoadOp>(loc, input, i);
    Value diff = builder.create<arith::SubFOp>(loc, x, mean_scalar);
    Value diff_sq = builder.create<arith::MulFOp>(loc, diff, diff);
    
    Value current_sum = builder.create<memref::LoadOp>(loc, var_out, c0);
    Value new_sum = builder.create<arith::AddFOp>(loc, current_sum, diff_sq);
    builder.create<memref::StoreOp>(loc, new_sum, var_out, c0);
    
    // Divide by N
    builder.setInsertionPointAfter(loop);
    Value sum_val = builder.create<memref::LoadOp>(loc, var_out, c0);
    Value N_float = builder.create<arith::IndexCastOp>(loc, i64Type, N);
    Value N_f32 = builder.create<arith::SIToFPOp>(loc, f32Type, N_float);
    Value variance = builder.create<arith::DivFOp>(loc, sum_val, N_f32);
    builder.create<memref::StoreOp>(loc, variance, var_out, c0);
}
```

**Stage 3: Normalize**

```cpp
void buildLayerNormNormalize(OpBuilder& builder, Location loc,
                             Value input, Value output, Value gamma, Value beta,
                             Value mean_val, Value var_val, Value N) {
    Value mean_scalar = builder.create<memref::LoadOp>(loc, mean_val, c0);
    Value var_scalar = builder.create<memref::LoadOp>(loc, var_val, c0);
    Value eps = createFloat(builder, loc, 1e-5f);
    
    // Compute 1/√(variance + ε)
    Value var_plus_eps = builder.create<arith::AddFOp>(loc, var_scalar, eps);
    Value std = builder.create<math::SqrtOp>(loc, var_plus_eps);
    Value one = createFloat(builder, loc, 1.0f);
    Value inv_std = builder.create<arith::DivFOp>(loc, one, std);
    
    auto loop = builder.create<scf::ForOp>(loc, c0, N, c1);
    builder.setInsertionPointToStart(loop.getBody());
    Value i = loop.getInductionVar();
    
    // Load values
    Value x = builder.create<memref::LoadOp>(loc, input, i);
    Value g = builder.create<memref::LoadOp>(loc, gamma, i);
    Value b = builder.create<memref::LoadOp>(loc, beta, i);
    
    // Normalize: (x - mean) / std
    Value centered = builder.create<arith::SubFOp>(loc, x, mean_scalar);
    Value normalized = builder.create<arith::MulFOp>(loc, centered, inv_std);
    
    // Affine: γ * normalized + β
    Value scaled = builder.create<arith::MulFOp>(loc, g, normalized);
    Value result = builder.create<arith::AddFOp>(loc, scaled, b);
    
    builder.create<memref::StoreOp>(loc, result, output, i);
}
```

### Test Results

```
test_layernorm (N=8)... ✅ PASSED (mean≈0, var≈1, max error: 8.94e-08)
test_layernorm_gamma_beta (N=64)... ✅ PASSED (scaled correctly)
test_layernorm_large (N=512)... ✅ PASSED (max error: 2.38e-07)
```

### Why This Works Now (AOT)

**JIT Problem (LLVM 20):** `engine->lookup()` hung indefinitely on any LayerNorm function

**AOT Solution:** Compile to object file at build time, link directly
- No symbol lookup at runtime
- No JIT compilation
- No mysterious hangs!

### Key Lessons

1. **Three-stage reduction:** Mean → Variance → Normalize
2. **Type conversions:** index → i64 → f32 for division
3. **Math dialect:** `math.SqrtOp` for square root
4. **AOT reliability:** Conquered the JIT bug that blocked this operation!
5. **Numerical stability:** Add ε before sqrt to prevent division by zero

---

## Phase 5: Transpose (Memory Patterns)

### Goal

Implement matrix transpose with 2D thread hierarchy.

**Tests:** 3/3 passing ✅

### Implementation

**File:** [src/transpose.cpp](src/transpose.cpp)

**Kernel:** `output[j,i] = input[i,j]`

**Thread Organization:**
- 2D grid and blocks (like matmul)
- Each thread transposes one element

```cpp
void buildTransposeKernel(OpBuilder& builder, Location loc,
                          Value input, Value output, Value M, Value N) {
    Value c16 = createIndex(builder, loc, 16);
    Value c15 = createIndex(builder, loc, 15);
    
    // Grid: ceil(M/16) × ceil(N/16)
    Value gridDimX = builder.create<arith::DivUIOp>(loc,
        builder.create<arith::AddIOp>(loc, M, c15), c16);
    Value gridDimY = builder.create<arith::DivUIOp>(loc,
        builder.create<arith::AddIOp>(loc, N, c15), c16);
    
    // 2D grid + 2D block (4 nested loops)
    auto blockLoopX = builder.create<scf::ForOp>(loc, c0, gridDimX, c1);
    // ... (similar structure to matmul) ...
    
    // Compute row, col indices
    Value row = builder.create<arith::AddIOp>(loc,
        builder.create<arith::MulIOp>(loc, blockX, c16), threadX);
    Value col = builder.create<arith::AddIOp>(loc,
        builder.create<arith::MulIOp>(loc, blockY, c16), threadY);
    
    // Bounds check
    Value rowValid = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, row, M);
    Value colValid = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, col, N);
    Value valid = builder.create<arith::AndIOp>(loc, rowValid, colValid);
    
    auto ifOp = builder.create<scf::IfOp>(loc, valid, false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    
    // Load from input[row, col], store to output[col, row]
    Value val = builder.create<memref::LoadOp>(loc, input, 
                                                ValueRange{row, col});
    builder.create<memref::StoreOp>(loc, val, output, 
                                     ValueRange{col, row});  // Swapped!
}
```

### Test Results

```
test_transpose_square (4×4)... ✅ PASSED
test_transpose_rectangular (32×64 → 64×32)... ✅ PASSED
test_transpose_identity (A^T^T = A)... ✅ PASSED
```

### Key Lessons

1. **Dimension swapping:** Store with indices reversed
2. **Memory access pattern:** Input coalesced, output strided (on real GPU)
3. **Reusable 2D pattern:** Same grid structure as matmul
4. **Mathematical properties:** Can verify with A^T^T = A

---

## Phase 6: Attention Mechanism

### Goal

Implement scaled dot-product attention by composing previous operations.

**Tests:** 3/3 passing ✅

### Algorithm

**Formula:** `Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V`

**Steps:**
1. Transpose K to get K^T
2. Matrix multiply: scores = Q @ K^T
3. Scale by 1/√d_k
4. Softmax (row-wise)
5. Matrix multiply: output = attention_weights @ V

### Implementation

**File:** [src/attention.cpp](src/attention.cpp)

**Scale Kernel** (element-wise multiply):

```cpp
extern "C" void scale_kernel(float* input, float* output, 
                             int N, float scale_factor) {
    // Standard 1D grid
    // ... grid setup ...
    Value x = builder.create<memref::LoadOp>(loc, input, globalIdx);
    Value scale_val = /* scale_factor as SSA value */;
    Value result = builder.create<arith::MulFOp>(loc, x, scale_val);
    builder.create<memref::StoreOp>(loc, result, output, globalIdx);
}
```

**Attention Kernel** (composition):

```cpp
extern "C" void attention_kernel(float* Q, float* K, float* V, float* output,
                                 int seq_len, int d_k, int d_v) {
    // Allocate temporaries
    float* K_T = new float[d_k * seq_len];
    float* scores = new float[seq_len * seq_len];
    float* scaled_scores = new float[seq_len * seq_len];
    float* attn_weights = new float[seq_len * seq_len];
    
    // Step 1: Transpose K
    transpose_kernel(K, K_T, seq_len, d_k);
    
    // Step 2: Q @ K^T
    matmul_kernel(Q, K_T, scores, seq_len, seq_len, d_k);
    
    // Step 3: Scale by 1/√d_k
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
    scale_kernel(scores, scaled_scores, seq_len * seq_len, scale_factor);
    
    // Step 4: Softmax (row-wise)
    for (int i = 0; i < seq_len; i++) {
        softmax_kernel(scaled_scores + i * seq_len,
                      attn_weights + i * seq_len,
                      seq_len);
    }
    
    // Step 5: Attention @ V
    matmul_kernel(attn_weights, V, output, seq_len, d_v, seq_len);
    
    // Cleanup
    delete[] K_T;
    delete[] scores;
    delete[] scaled_scores;
    delete[] attn_weights;
}
```

### Test Results

```
test_scale (N=1024, scale=0.5)... ✅ PASSED (max error: 0.00e+00)
test_attention_small (seq=4, d_k=8)... ✅ PASSED (max error: 2.68e-07)
test_attention_properties (no NaN/inf)... ✅ PASSED
```

### Key Lessons

1. **Composability:** Complex operations from simple building blocks
2. **No new GPU patterns:** Reuse matmul, transpose, softmax, scale
3. **Memory management:** Allocate intermediate buffers
4. **Scaling importance:** Prevents softmax saturation (keeps gradients healthy)
5. **Row-wise softmax:** Each query attends independently

---

## Phase 7: Complete Transformer (Nano-GPT!)

### Goal

Build a complete GPT-style transformer with all components.

**Tests:** 4/4 passing ✅

### Architecture

```
Input: token_ids [seq_len]
  ↓
Token Embedding + Positional Embedding [seq_len, d_model]
  ↓
Transformer Block:
  ├─ LayerNorm
  ├─ Causal Self-Attention (masked)
  ├─ Residual Connection
  ├─ LayerNorm
  ├─ Feed-Forward Network (2-layer MLP with GELU)
  └─ Residual Connection
  ↓
Final LayerNorm → Output Projection
  ↓
Logits [seq_len, vocab_size]
```

### Kernels Implemented

**File:** [src/transformer.cpp](src/transformer.cpp)

#### 1. Embedding Lookup

```cpp
extern "C" void embedding_lookup(int* token_ids, float* embedding_table,
                                 float* output, int seq_len, 
                                 int vocab_size, int d_model) {
    for (int i = 0; i < seq_len; i++) {
        int token_id = token_ids[i];
        if (token_id < 0 || token_id >= vocab_size) {
            // Zero out invalid tokens
            for (int j = 0; j < d_model; j++) {
                output[i * d_model + j] = 0.0f;
            }
        } else {
            // Copy embedding
            for (int j = 0; j < d_model; j++) {
                output[i * d_model + j] = 
                    embedding_table[token_id * d_model + j];
            }
        }
    }
}
```

#### 2. Causal Attention (with Masking)

**Key concept:** Prevent attending to future tokens (autoregressive property)

```cpp
extern "C" void causal_attention_kernel(float* Q, float* K, float* V,
                                        float* output, int seq_len,
                                        int d_k, int d_v) {
    // ... (similar to attention_kernel) ...
    
    // After computing scores, before softmax:
    // Apply causal mask: mask[i,j] = -1e9 if j > i
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            if (j > i) {
                scaled_scores[i * seq_len + j] = -1e9f;  // Mask future
            }
        }
    }
    
    // Then apply softmax (masked positions → ~0 weight)
    // ...
}
```

**Masking visualization (seq_len=4):**
```
    0   1   2   3
0  [X   -   -   -]  ← Token 0 only sees itself
1  [X   X   -   -]  ← Token 1 sees 0,1
2  [X   X   X   -]  ← Token 2 sees 0,1,2
3  [X   X   X   X]  ← Token 3 sees all
```

#### 3. Feed-Forward Network

**Architecture:** `Linear(d_model → 4*d_model) → GELU → Linear(4*d_model → d_model)`

```cpp
extern "C" void feedforward_kernel(float* input, float* W1, float* b1,
                                   float* W2, float* b2, float* output,
                                   int seq_len, int d_model, int d_ff) {
    float* hidden = new float[seq_len * d_ff];
    
    // Step 1: input @ W1 + b1
    matmul_kernel(input, W1, hidden, seq_len, d_ff, d_model);
    for (int i = 0; i < seq_len; i++) {
        bias_add_kernel(hidden + i * d_ff, b1, hidden + i * d_ff, d_ff);
    }
    
    // Step 2: GELU activation
    gelu_kernel(hidden, hidden, seq_len * d_ff);
    
    // Step 3: hidden @ W2 + b2
    matmul_kernel(hidden, W2, output, seq_len, d_model, d_ff);
    for (int i = 0; i < seq_len; i++) {
        bias_add_kernel(output + i * d_model, b2, 
                       output + i * d_model, d_model);
    }
    
    delete[] hidden;
}
```

#### 4. Transformer Block

**Pre-LayerNorm architecture:**

```cpp
extern "C" void transformer_block(float* input, float* output,
                                  /* weights... */, 
                                  int seq_len, int d_model) {
    float* ln1_out = new float[seq_len * d_model];
    float* attn_out = new float[seq_len * d_model];
    float* after_attn = new float[seq_len * d_model];
    float* ln2_out = new float[seq_len * d_model];
    float* ffn_out = new float[seq_len * d_model];
    
    // Attention block with residual
    layernorm_kernel(input, ln1_out, gamma1, beta1, seq_len, d_model);
    causal_attention_kernel(ln1_out, /* ... */);
    add_kernel(input, attn_out, after_attn, seq_len * d_model);  // Residual
    
    // FFN block with residual
    layernorm_kernel(after_attn, ln2_out, gamma2, beta2, seq_len, d_model);
    feedforward_kernel(ln2_out, /* ... */);
    add_kernel(after_attn, ffn_out, output, seq_len * d_model);  // Residual
    
    // Cleanup
    delete[] ln1_out;
    // ...
}
```

#### 5. KV Cache (Generation Optimization)

**Problem:** Autoregressive generation recomputes K, V for all previous tokens → O(n²)

**Solution:** Cache K, V tensors, only compute for new token → O(n)

```cpp
extern "C" void kv_cached_attention(float* Q_new,       // [1, d_k]
                                    float* K_cache,      // [cached_len, d_k]
                                    float* V_cache,      // [cached_len, d_v]
                                    float* K_new,        // [1, d_k]
                                    float* V_new,        // [1, d_v]
                                    float* output,       // [1, d_v]
                                    int cached_len, int d_k, int d_v) {
    // Concatenate caches
    int full_len = cached_len + 1;
    float* K_full = new float[full_len * d_k];
    float* V_full = new float[full_len * d_v];
    
    std::memcpy(K_full, K_cache, cached_len * d_k * sizeof(float));
    std::memcpy(K_full + cached_len * d_k, K_new, d_k * sizeof(float));
    
    std::memcpy(V_full, V_cache, cached_len * d_v * sizeof(float));
    std::memcpy(V_full + cached_len * d_v, V_new, d_v * sizeof(float));
    
    // Attend: Q_new @ K_full^T → softmax → @ V_full
    attention_kernel(Q_new, K_full, V_full, output, 1, full_len, d_k, d_v);
    
    delete[] K_full;
    delete[] V_full;
}
```

**Efficiency comparison (50 tokens):**
- **Without cache:** 1+2+3+...+50 = 1,275 token computations → O(n²)
- **With cache:** 1+1+1+...+1 = 50 token computations → O(n)
- **Speedup:** 25× faster! 🚀

#### 6. Complete Forward Pass

See [src/transformer.cpp](src/transformer.cpp) for full implementation including autoregressive generation with KV cache.

### Test Results

```
test_embedding_lookup (seq_len=4, vocab=256, d_model=64)... ✅ PASSED
test_causal_attention (seq_len=4, d_k=8, masked correctly)... ✅ PASSED
test_transformer_block (seq_len=8, d_model=64)... ✅ PASSED (output mean: 0.566)
test_kv_cache (cached_len=3, new=1 token)... ✅ PASSED (attends to 4 total)
```

### What This Means

**You have a complete, production-ready GPT architecture!** 🎉

Given trained weights, this code can:
- ✅ Process sequences (forward pass)
- ✅ Generate text token-by-token (with KV cache)
- ✅ Attend causally (no future information leakage)
- ✅ Scale efficiently (O(n) generation complexity)

**Only missing for full ChatGPT-style inference:**
- Temperature sampling (trivial: `logits / temperature`)
- Top-k/top-p sampling (minor: sort + threshold)
- Multi-layer stacking (easy: loop over `transformer_block()` N times)

### Key Lessons

1. **Causal masking:** Essential for autoregressive generation
2. **Residual connections:** Enable deep networks (gradient flow)
3. **Pre-LayerNorm:** Stabilizes training (norm before each sub-layer)
4. **KV cache:** Algorithmic breakthrough (25× speedup for generation!)
5. **Composability:** 25 kernels combine into complete transformer
6. **AOT reliability:** All tests passing (no JIT bugs!)

---

## Build and Test

### Build

```bash
cd ~/mlir-example
cmake --preset x64-release
cmake --build --preset x64-release --target ch15_test
```

**Output location:** `build/x64-release/ch.15.GPU-Concepts/ch15_test`

### Run Tests

```bash
cd build/x64-release
./ch.15.GPU-Concepts/ch15_test
```

**Expected output:**

```
╔════════════════════════════════════════════════════════════╗
║  Chapter 15: GPU Concepts (AOT Compilation)                ║
║  Complete Nano-GPT Implementation                          ║
╚════════════════════════════════════════════════════════════╝

Architecture: AOT (No JIT, No GPU - CPU emulation)
Environment: WSL/Linux CPU
Compilation: MLIR → LLVM → Native Code (built at compile time)

Running 25 tests across 7 phases...
[... all tests ...]
════════════════════════════════════════════════════════════
Summary: 25/25 tests PASSED ✅
Chapter 15 COMPLETE! Nano-GPT fully operational! 🚀
════════════════════════════════════════════════════════════
```