# MLIR GEMM Project: Improvement Roadmap

This document tracks the progressive improvements to the MLIR matrix multiplication project, from fixed-size matrices to a fully flexible, production-ready implementation.

---

## Table of Contents

- [Chapter 1: Understanding the Fixed-Size Limitation](#chapter-1-understanding-the-fixed-size-limitation)
- [Chapter 2: Dynamic Shapes Implementation](#chapter-2-dynamic-shapes-implementation) ✅ **COMPLETE**
- [Chapter 3: Function Caching](#chapter-3-function-caching) 📋 *Coming next*
- [Chapter 4: Tensor-Based Approach](#chapter-4-tensor-based-approach) 📋 *Future*
- [Chapter 5: Advanced Optimizations](#chapter-5-advanced-optimizations) 📋 *Future*

---

## Chapter 1: Understanding the Fixed-Size Limitation

### Current State: Why Fixed Sizes?

The original implementation was hardcoded to multiply **8×32 × 32×16 matrices**. This limitation existed in **three places**:

#### 1. **IR Generation** (`src/ir.cpp`)
```cpp
auto matrixA_type = MemRefType::get({8, 32}, f32Type);  // Fixed dimensions
auto matrixB_type = MemRefType::get({32, 16}, f32Type);
auto matrixC_type = MemRefType::get({8, 16}, f32Type);
```

**Why hardcoded?**
- `MemRefType::get({rows, cols}, elementType)` creates a **static shape** memref type
- The dimensions `{8, 32}` are compile-time constants baked into the MLIR type system
- The generated MLIR function signature literally says `memref<8x32xf32>` - not flexible!

#### 2. **Python Validation** (`src/bindings.cpp`)
```cpp
if (A_buf.shape[0] != 8 || A_buf.shape[1] != 32) {
    throw std::runtime_error("Matrix A must be 8x32");
}
```

**Why checked?**
- Safety guard: ensures Python arrays match the hardcoded MLIR function
- This is a **consequence** of the fixed MLIR signature, not the root cause

#### 3. **JIT Function Call** (`src/jit.cpp`)
```cpp
gemm_func(
    A, A, 0, 8, 32, 32, 1,      // A: memref<8x32xf32>
    B, B, 0, 32, 16, 16, 1,     // B: memref<32x16xf32>
    C, C, 0, 8, 16, 16, 1       // C: memref<8x16xf32>
);
```

**Why hardcoded?**
- Memrefs expand to 7 arguments (allocated, aligned, offset, size0, size1, stride0, stride1)
- The sizes `8, 32` are passed as **compile-time known constants**
- The JIT-compiled function **expects these exact values**

### The Root Cause: Static vs Dynamic Shapes

The fundamental issue is the **type system**:

```mlir
// Static shape (compile-time constant)
func.func @gemm_8x16x32(
    %arg0: memref<8x32xf32>,    // ← Shape is part of the TYPE
    %arg1: memref<32x16xf32>,
    %arg2: memref<8x16xf32>
)
```

In MLIR's type system:
- `memref<8x32xf32>` means "8×32 array of float32" - dimensions are **part of the type**
- Different sizes = different types = need different functions!
- This is like how `int` vs `float` are incompatible types

### The Solution: Two Approaches

#### **Approach A: Dynamic Memrefs** ✅ (What we implemented)
```mlir
func.func @gemm(
    %arg0: memref<?x?xf32>,    // ← ? means "unknown at compile time"
    %arg1: memref<?x?xf32>,
    %arg2: memref<?x?xf32>
)
```

**Pros:**
- Still uses memrefs (no bufferization needed)
- Natural extension of current approach
- Sizes are runtime parameters in the memref descriptor

#### **Approach B: Tensors + Bufferization** (Future work)
```mlir
func.func @gemm(
    %arg0: tensor<?x?xf32>,    // ← Immutable tensor (value semantics)
    %arg1: tensor<?x?xf32>
) -> tensor<?x?xf32> {
    %result = linalg.matmul ins(%arg0, %arg1) ...
    return %result
}
```

**Pros:**
- More idiomatic MLIR (value semantics)
- Better optimization opportunities
- Easier to reason about transformations

---

## Chapter 2: Dynamic Shapes Implementation

**Status:** ✅ **COMPLETE!**

### What We Achieved

**Before:**
```python
# Only worked with 8×32 × 32×16 matrices
A = np.ones((8, 32), dtype=np.float32)
B = np.ones((32, 16), dtype=np.float32)
C = llvm_example.gemm(A, B)  # OK

A = np.ones((10, 20), dtype=np.float32)
B = np.ones((20, 15), dtype=np.float32)
C = llvm_example.gemm(A, B)  # ERROR: Wrong size!
```

**After:**
```python
# Works with ANY compatible matrix sizes!
test_cases = [
    ((8, 32), (32, 16)),      # Original size
    ((10, 20), (20, 15)),     # Custom size
    ((5, 5), (5, 5)),         # Square
    ((100, 50), (50, 25)),    # Large
    ((3, 100), (100, 3)),     # Thin × wide
]

for shape_A, shape_B in test_cases:
    A = np.ones(shape_A, dtype=np.float32)
    B = np.ones(shape_B, dtype=np.float32)
    C = llvm_example.gemm(A, B)  # All work perfectly!
```

### Implementation Strategy: Baby Steps

We used a **parallel path** approach to keep the project building at each step:

1. **Step 1:** Add new `createDynamicGemmModule()` alongside old function
2. **Step 2:** Add test function to verify dynamic IR
3. **Step 3:** Add new `executeDynamicGemm()` alongside old function
4. **Step 4:** Add new `gemm_dynamic()` Python function
5. **Step 5:** Make dynamic the default, remove old code

This strategy ensured:
- ✅ Project builds at every step
- ✅ Old code keeps working during development
- ✅ Easy to test and verify each change
- ✅ Safe rollback if something breaks

### Key Technical Changes

#### 1. IR Generation (`src/ir.cpp`)

**Before:**
```cpp
auto matrixA_type = MemRefType::get({8, 32}, f32Type);
```

**After:**
```cpp
auto matrixA_type = MemRefType::get(
    {ShapedType::kDynamic, ShapedType::kDynamic},  // ?x? (any size)
    f32Type
);
```

**Key insight:** `ShapedType::kDynamic` is a special constant (`-1`) meaning "unknown at compile time"

#### 2. JIT Execution (`src/jit.cpp`)

**Before:**
```cpp
void executeGemm(float* A, float* B, float* C) {
    // ... hardcoded dimensions ...
    gemm_func(A, A, 0, 8, 32, 32, 1, ...);
}
```

**After:**
```cpp
void executeGemm(float* A, float* B, float* C, 
                 int64_t M, int64_t N, int64_t K) {
    // ... pass runtime dimensions ...
    gemm_func(A, A, 0, M, K, K, 1, ...);  // Runtime values!
}
```

**Key insight:** Memref descriptor has size0/size1 fields that carry runtime dimensions

#### 3. Python Bindings (`src/bindings.cpp`)

**Before:**
```cpp
if (A_buf.shape[0] != 8 || A_buf.shape[1] != 32) {
    throw std::runtime_error("Matrix A must be 8x32");
}
```

**After:**
```cpp
// Extract dimensions from NumPy arrays
int64_t M = A_buf.shape[0];
int64_t K = A_buf.shape[1];
int64_t N = B_buf.shape[1];

// Validate compatibility (not fixed size!)
if (K != B_buf.shape[0]) {
    throw std::runtime_error("A.cols must equal B.rows");
}
```

**Key insight:** Validate **compatibility**, not specific sizes

### Understanding ShapedType::kDynamic

**What is it?**
- A special constant (value: `-1`) meaning "dimension size unknown at compile time"
- Used in `MemRefType::get({dim0, dim1}, elementType)` to create dynamic memrefs

**What MLIR IR does it generate?**
```mlir
// Static:
func.func @gemm_8x16x32(%arg0: memref<8x32xf32>, ...)

// Dynamic:
func.func @gemm(%arg0: memref<?x?xf32>, ...)
//                              ↑  ↑
//                              |  └─ Dynamic dimension (column count)
//                              └──── Dynamic dimension (row count)
```

### Shape Polymorphism

The beauty of `linalg.matmul`:
```mlir
// Works with static shapes:
linalg.matmul ins(%A: memref<8x32xf32>, %B: memref<32x16xf32>)
              outs(%C: memref<8x16xf32>)

// Also works with dynamic shapes:
linalg.matmul ins(%A: memref<?x?xf32>, %B: memref<?x?xf32>)
              outs(%C: memref<?x?xf32>)
```

**The operation doesn't change** - only the type annotations! The lowering passes handle the rest automatically.

### Code Changes Summary

| File | Lines Changed | What Changed |
|------|--------------|--------------|
| `ir.cpp` | ~90 | Used `ShapedType::kDynamic` for dimensions |
| `jit.cpp` | ~120 | Added M, N, K parameters; pass runtime dimensions |
| `bindings.cpp` | ~80 | Extract dimensions from NumPy; validate compatibility |

**Total:** ~290 lines changed to enable full dynamic shape support!

### Key Learnings

1. **ShapedType::kDynamic** - Special constant (-1) for unknown dimensions
2. **Shape polymorphism** - `linalg.matmul` works with any size automatically
3. **Memref descriptors** - Runtime dimensions passed through size0/size1 fields
4. **No changes to lowering** - Same optimization passes work for dynamic shapes!
5. **Parallel path strategy** - Safest way to refactor incrementally

### Test Results

```python
# All tests pass with perfect accuracy!
Test: 8×32 × 32×16   → C(8, 16)   ✓ (original size)
Test: 10×20 × 20×15  → C(10, 15)  ✓ (custom size)
Test: 5×5 × 5×5      → C(5, 5)    ✓ (square)
Test: 100×50 × 50×25 → C(100, 25) ✓ (large)
Test: 3×100 × 100×3  → C(3, 3)    ✓ (thin × wide)

Max error vs NumPy: 0.00e+00  ✓ (perfect accuracy)
```

---

## Chapter 3: Function Caching

**Status:** 📋 *Planned for future*

### The Performance Issue

Currently, we recompile the MLIR function **on every call**:
```python
C1 = gemm(A1, B1)  # Compiles: takes ~100ms
C2 = gemm(A2, B2)  # Recompiles again: another ~100ms
C3 = gemm(A3, B3)  # Recompiles again: another ~100ms
```

This is wasteful! The compiled code is the same for all calls with the same shape.

### The Solution: JIT Function Cache

**Goal:** Compile once per unique shape, cache the result:
```python
C1 = gemm(A1, B1)  # Compiles: ~100ms (first time)
C2 = gemm(A2, B2)  # Cached: <1ms (same shape!)
C3 = gemm(A3, B3)  # Cached: <1ms (same shape!)
```

### Implementation Plan

1. Create a cache key from (M, N, K) dimensions
2. Store compiled function pointers in a hash map
3. Check cache before compiling
4. Warm up cache for common sizes

### Expected Benefits

- 100x speedup for repeated operations
- Amortized compilation cost
- Better for interactive/iterative workloads

---

## Chapter 4: Tensor-Based Approach

**Status:** 📋 *Planned for future*

### Why Tensors?

Tensors represent **immutable values** (functional style), while memrefs represent **mutable memory** (imperative style).

**Tensor advantages:**
- Value semantics (easier to reason about)
- Better for optimization passes
- More idiomatic MLIR
- Enables transformations like fusion

### Migration Plan

1. Change IR generation to use `RankedTensorType` instead of `MemRefType`
2. Add bufferization passes to convert tensors to memrefs
3. Update Python bindings for tensor semantics
4. Verify correctness and performance

### Bufferization

Bufferization is the process of converting tensor operations to memref operations:

```mlir
// Before: Tensor (value semantics)
%result = linalg.matmul ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                        -> tensor<?x?xf32>

// After: Memref (memory semantics)
linalg.matmul ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
              outs(%C: memref<?x?xf32>)
```

---

## Chapter 5: Advanced Optimizations

**Status:** 📋 *Planned for future*

### Loop Tiling

Break large loops into smaller tiles for better cache locality:

```mlir
// Before: Simple nested loops
for i = 0 to M:
  for j = 0 to N:
    for k = 0 to K:
      C[i,j] += A[i,k] * B[k,j]

// After: Tiled loops
for ii = 0 to M step 32:
  for jj = 0 to N step 32:
    for kk = 0 to K step 32:
      for i = ii to min(ii+32, M):
        for j = jj to min(jj+32, N):
          for k = kk to min(kk+32, K):
            C[i,j] += A[i,k] * B[k,j]
```

### Vectorization

Use SIMD instructions for parallel computation:
```mlir
// Vector operations (process 8 floats at once)
%vec_a = vector.load %A[%i, %k] : vector<8xf32>
%vec_b = vector.load %B[%k, %j] : vector<8xf32>
%vec_c = vector.fma %vec_a, %vec_b, %vec_c : vector<8xf32>
```

### Parallelization

Use multiple CPU cores:
```mlir
scf.parallel (%i, %j) = (%c0, %c0) to (%M, %N) step (%c1, %c1) {
  // Each iteration can run on different cores
  ...
}
```

---

## Progress Summary

| Chapter | Status | Description |
|---------|--------|-------------|
| Chapter 1 | ✅ Complete | Understanding fixed-size limitations |
| Chapter 2 | ✅ Complete | Dynamic shapes with memrefs |
| Chapter 3 | 📋 Planned | Function caching for performance |
| Chapter 4 | 📋 Planned | Tensor-based approach + bufferization |
| Chapter 5 | 📋 Planned | Advanced optimizations |

**Current Achievement:** Fully functional dynamic shape support with perfect accuracy!

**Next Milestone:** Implement function caching to avoid recompilation overhead.

### Current State: Why Fixed Sizes?

The current implementation is hardcoded to multiply **8×32 × 32×16 matrices**. This limitation exists in **three places** in the code:

#### 1. **IR Generation** (`src/ir.cpp` lines 67-69)
```cpp
auto matrixA_type = MemRefType::get({8, 32}, f32Type);
auto matrixB_type = MemRefType::get({32, 16}, f32Type);
auto matrixC_type = MemRefType::get({8, 16}, f32Type);
```

**Why hardcoded here?**
- `MemRefType::get({rows, cols}, elementType)` creates a **static shape** memref type
- The dimensions `{8, 32}` are compile-time constants baked into the MLIR type system
- The generated MLIR function signature literally says `memref<8x32xf32>` - not flexible!

#### 2. **Python Validation** (`src/bindings.cpp` lines 58-68)
```cpp
if (A_buf.shape[0] != 8 || A_buf.shape[1] != 32) {
    throw std::runtime_error("Matrix A must be 8x32");
}
if (B_buf.shape[0] != 32 || B_buf.shape[1] != 16) {
    throw std::runtime_error("Matrix B must be 32x16");
}
```

**Why checked here?**
- Safety guard: ensures Python arrays match the hardcoded MLIR function
- Without this check, passing wrong sizes would cause memory corruption
- This is a **consequence** of the fixed MLIR signature, not the root cause

#### 3. **JIT Function Call** (`src/jit.cpp` lines 150-152)
```cpp
gemm_func(
    A, A, 0, 8, 32, 32, 1,      // A: memref<8x32xf32>
    B, B, 0, 32, 16, 16, 1,     // B: memref<32x16xf32>
    C, C, 0, 8, 16, 16, 1       // C: memref<8x16xf32>
);
```

**Why hardcoded here?**
- Remember: memrefs expand to 7 arguments (allocated, aligned, offset, size0, size1, stride0, stride1)
- The sizes `8, 32` are passed as **compile-time known constants**
- The JIT-compiled function **expects these exact values** because they're part of the function signature

### The Root Cause: Static vs Dynamic Shapes

The fundamental issue is the **type system**:

```mlir
// Current: Static shape (compile-time constant)
func.func @gemm_8x16x32(
    %arg0: memref<8x32xf32>,    // ← Shape is part of the TYPE
    %arg1: memref<32x16xf32>,
    %arg2: memref<8x16xf32>
)
```

In MLIR's type system:
- `memref<8x32xf32>` means "8×32 array of float32" - dimensions are **part of the type**
- Different sizes = different types = need different functions!
- `memref<10x20xf32>` and `memref<8x32xf32>` are **incompatible types** (like `int` vs `float`)

### Why Did We Start This Way?

**Pedagogical reasons:**
1. **Simpler to understand** - fixed sizes eliminate one layer of complexity
2. **Easier to debug** - no dynamic dimension logic to worry about
3. **Follows the MLIR Toy Tutorial** - which also starts with static shapes
4. **Avoids bufferization** - tensor-based dynamic shapes require bufferization passes

**This is the right learning path!** Master the basics before adding complexity.

### What Doesn't Work (Common Misconceptions)

❌ **"Just pass the size as a parameter"**
```cpp
// This WON'T work:
void gemm(memref<8x32xf32> A, int rows, int cols) {
    // Can't change the type based on parameters!
}
```
The type is already `memref<8x32xf32>` - you can't change it at runtime.

❌ **"Just remove the validation in bindings.cpp"**
```cpp
// Removing this check would cause crashes:
if (A_buf.shape[0] != 8) { throw ... }  // ← Don't just delete this!
```
The MLIR function still expects 8×32. Passing 10×20 would cause memory corruption.

❌ **"Just pass different sizes to the JIT call"**
```cpp
// This would crash - the compiled function expects specific sizes:
gemm_func(A, A, 0, 10, 20, 20, 1, ...);  // ← Function signature is fixed!
```

### The Correct Solution Path

There are **two approaches** to support flexible shapes:

#### **Approach A: Dynamic Memrefs** (Simpler, what we'll do first)
Use MLIR's dynamic dimension feature:
```mlir
func.func @gemm(
    %arg0: memref<?x?xf32>,    // ← ? means "unknown at compile time"
    %arg1: memref<?x?xf32>,
    %arg2: memref<?x?xf32>
)
```

**Pros:**
- Still uses memrefs (no bufferization needed)
- Natural extension of current approach
- Sizes are runtime parameters in the memref descriptor

**Cons:**
- Less optimizable (compiler doesn't know sizes)
- Still working with mutable memrefs

#### **Approach B: Tensors + Bufferization** (More complex, better long-term)
Use tensors (immutable values) and let MLIR handle memory:
```mlir
func.func @gemm(
    %arg0: tensor<?x?xf32>,    // ← Immutable tensor (value semantics)
    %arg1: tensor<?x?xf32>
) -> tensor<?x?xf32> {          // ← Returns result (functional style)
    %result = linalg.matmul ins(%arg0, %arg1) ...
    return %result
}
```

**Pros:**
- More idiomatic MLIR (value semantics)
- Better optimization opportunities
- Easier to reason about transformations

**Cons:**
- Requires bufferization passes
- More complex pipeline
- Steeper learning curve

### Our Improvement Plan

We'll follow a **gradual progression**:

1. **Step 1: Dynamic memrefs (Chapter 2)** ✅ **COMPLETE!**
   - Modified IR generation to use `memref<?x?xf32>`
   - Updated JIT call to pass runtime dimensions
   - Removed fixed-size validation in bindings
   - **Result:** Works with ANY matrix size!

2. **Step 2: Multiple specializations (Chapter 3)** 📋 *Coming next*
   - Cache compiled functions for different sizes
   - Avoid recompiling on every call

3. **Step 3: Tensors + Bufferization (Chapter 4)** 📋 *Future*
   - Migrate to tensor-based IR
   - Add bufferization passes
   - Understand the full MLIR transformation pipeline

4. **Step 4: Advanced optimizations (Chapter 5)** 📋 *Future*
   - Loop tiling for cache efficiency
   - Vectorization
   - Parallel execution

---

## Chapter 2: ✅ COMPLETE!

**What we achieved:**
- Dynamic shapes using `ShapedType::kDynamic`
- Works with matrices of any compatible size
- Clean, simple API: `gemm(A, B)` just works
- Verified against NumPy: perfect accuracy

**See:** `CHAPTER_2_DYNAMIC_SHAPES.md` for full details

---

## Key Takeaways

### Why the current code uses fixed sizes:
1. **Type system constraint**: `memref<8x32xf32>` has dimensions as part of the type
2. **Compile-time decisions**: LLVM codegen needs to know sizes (or mark them dynamic)
3. **Three coupled locations**: IR generation, validation, and JIT call must all agree

### Why this is actually good for learning:
- Simpler to understand the basic MLIR pipeline
- Fewer moving parts to debug
- Clear separation of concerns

### What we'll learn by removing this limitation:
- Dynamic shapes in MLIR type system
- Runtime vs compile-time information
- Shape inference and propagation
- Eventually: bufferization and tensor semantics

---

**Next:** Chapter 2 will implement dynamic memrefs step-by-step!
