# MLIR-JIT GEMM: Implementation Guide

This comprehensive guide covers both the high-level API design and the low-level bufferization implementation details.

## Table of Contents

### Part I: API Design & Architecture
1. [User-Facing API](#user-facing-api)
2. [Implementation Architecture](#implementation-architecture)
3. [The Out-Parameter Pattern](#the-out-parameter-pattern)
4. [Common Pitfall: Function Typedef Mismatch](#common-pitfall-function-typedef-mismatch)

### Part II: MLIR Bufferization Deep Dive
5. [What is Bufferization?](#what-is-bufferization)
6. [The Challenge](#the-challenge)
7. [The Correct Approach](#the-correct-approach)
8. [Why Other Approaches Failed](#why-other-approaches-failed)
9. [Key Learnings](#key-learnings)

---

# Part I: API Design & Architecture

## User-Facing API

The Python API is clean and user-friendly:

```python
import llvm_example

# Clean API - No manual allocation needed!
C = llvm_example.gemm(A, B)
```

Users **never** need to manually allocate memory for the output matrix. The allocation happens transparently inside the C++ binding code.

### Example Usage

```python
import numpy as np
import llvm_example

# Small matrices
A = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]], dtype=np.float32)
B = np.array([[7.0, 8.0],
              [9.0, 10.0],
              [11.0, 12.0]], dtype=np.float32)

C = llvm_example.gemm(A, B)  # Returns [[58, 64], [139, 154]]

# Large matrices - same API!
A_large = np.random.randn(1000, 1000).astype(np.float32)
B_large = np.random.randn(1000, 1000).astype(np.float32)
C_large = llvm_example.gemm(A_large, B_large)  # Works with any size!
```

### Key Features

- ✅ **No manual allocation**: Output array created automatically
- ✅ **Dynamic shapes**: Works with any matrix dimensions
- ✅ **NumPy integration**: Seamless interop with NumPy arrays
- ✅ **Type safety**: Validates dimensions and data types
- ✅ **Error messages**: Clear error reporting for invalid inputs

## Implementation Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Python User Code                                        │
│   C = llvm_example.gemm(A, B)  ← Clean API!            │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Python Binding (bindings.cpp)                           │
│   • Validates inputs (2D, float32, compatible shapes)   │
│   • Allocates output array C                            │
│   • Calls executeGemm(A, B, C, M, N, K)                │
│   • Returns C to Python                                 │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ JIT Executor (jit.cpp)                                  │
│   • Checks if function is cached                        │
│   • If not cached: Compile once and cache               │
│   • Expands memrefs to 21 parameters                    │
│   • Calls: gemm(A[7], B[7], C[7])                      │
│   • C is out-parameter (void return)                    │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ MLIR Optimization Pipeline (lowering.cpp)              │
│   1. Canonicalization (simplify IR)                     │
│   2. One-Shot Bufferize (tensor → memref)              │
│   3. Buffer-Results-To-Out-Params (return → out-param)  │
│   4. Bufferization-To-MemRef (finalize memrefs)        │
│   5. Linalg to Loops (linalg.matmul → scf.for)         │
│   6. SCF to Control Flow (scf.for → cf.br)             │
│   7. Convert to LLVM Dialect                            │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Generated Machine Code (x86_64 assembly)               │
│   • malloc buffer for result                            │
│   • memset(buffer, 0) - zero initialization            │
│   • Triple-nested loop for matmul                       │
│   • memcpy(buffer, C_out_param) - copy to output       │
└─────────────────────────────────────────────────────────┘
```

### Layer-by-Layer Breakdown

#### **Layer 1: Python Binding (`bindings.cpp`)**

```cpp
py::array_t<float> gemm(py::array_t<float> A, py::array_t<float> B) {
  // Extract dimensions
  int64_t M = A.shape[0];  // Rows in A
  int64_t K = A.shape[1];  // Cols in A / Rows in B
  int64_t N = B.shape[1];  // Cols in B
  
  // Validate compatibility
  if (A.shape[1] != B.shape[0]) {
    throw std::runtime_error("Matrix dimensions incompatible");
  }
  
  // Allocate output array (hidden from user!)
  auto C = py::array_t<float>({M, N});
  
  // Call JIT function with C as out-parameter
  mlir::executeGemm(A.ptr, B.ptr, C.ptr, M, N, K);
  
  return C;  // Return to Python
}
```

**Responsibilities:**
- Input validation
- Output allocation (automatic!)
- Type conversion (NumPy ↔ C++)
- Error handling

#### **Layer 2: JIT Executor (`jit.cpp`)**

```cpp
void executeGemm(float* A, float* B, float* C, int64_t M, int64_t N, int64_t K) {
  // Check cache (compile once, reuse for all shapes!)
  if (!gGemmJIT.isCompiled) {
    auto [jit, func] = compileGemmFunction();
    gGemmJIT.jit = std::move(jit);
    gGemmJIT.funcPtr = func;
    gGemmJIT.isCompiled = true;
  }
  
  // Call cached function
  GemmFnPtr gemm_func = gGemmJIT.funcPtr;
  
  // Expand memrefs to 7 params each (ptr, ptr, offset, sizes[2], strides[2])
  gemm_func(
      A, A, 0, M, K, K, 1,      // A: memref<?x?xf32>
      B, B, 0, K, N, N, 1,      // B: memref<?x?xf32>
      C, C, 0, M, N, N, 1       // C: memref<?x?xf32> (out-param!)
  );
}
```

**Responsibilities:**
- JIT compilation (once per process)
- Function caching
- MemRef descriptor expansion
- Runtime dimension passing

#### **Layer 3: MLIR Lowering Pipeline (`lowering.cpp`)**

Transforms high-level tensor operations into low-level LLVM IR through a series of passes. See [Part II](#part-ii-mlir-bufferization-deep-dive) for details.

## The Out-Parameter Pattern

### Why Out-Parameters?

The JIT-compiled function uses out-parameter style:

```cpp
// Function signature after bufferization
void gemm(
    float* A_ptr, float* A_aligned, int64_t A_offset, 
    int64_t A_size0, int64_t A_size1, int64_t A_stride0, int64_t A_stride1,
    float* B_ptr, float* B_aligned, int64_t B_offset,
    int64_t B_size0, int64_t B_size1, int64_t B_stride0, int64_t B_stride1,
    float* C_ptr, float* C_aligned, int64_t C_offset,
    int64_t C_size0, int64_t C_size1, int64_t C_stride0, int64_t C_stride1
);
```

**Advantages:**

1. **Memory Management**: Caller controls allocation
2. **ABI Compatibility**: Struct returns have complex calling conventions
3. **Performance**: Avoids copying large descriptor structures on stack
4. **Standard C Practice**: Matches C ABI expectations

### MLIR Transformation

The `buffer-results-to-out-params` pass converts:

```mlir
// Before: Return value
func.func @gemm(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) 
    -> memref<?x?xf32>

// After: Out-parameter
func.func @gemm(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                %arg2: memref<?x?xf32>)
```

This happens automatically during the bufferization pipeline.

## Common Pitfall: Function Typedef Mismatch

### The Bug

A common mistake is declaring the function pointer with a return type:

```cpp
// ❌ WRONG: Says function returns MemRefDescriptor
using GemmFnPtr = MemRefDescriptor(*)(
    float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // A
    float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // B
    float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t   // C
);
```

But after `buffer-results-to-out-params`, the actual signature is:

```cpp
// ✅ CORRECT: Void return, C is out-parameter
void gemm(A[7 params], B[7 params], C[7 params]);
```

### Symptoms

When the typedef mismatches the actual function:

- ✅ Compilation succeeds (IR is correct)
- ❌ Runtime produces garbage: `[[0.0, -2.0], [-2.6e11, 4.1e-41]]`
- ❌ Tests show NaN and random floats
- ❌ Memory corruption

**Why?** Calling convention mismatch:
- Caller expects return value in RAX register
- Callee writes to out-parameter instead
- Stack/register layout differs between calling conventions
- Output buffer isn't written correctly

### The Fix

Match the typedef to the lowered signature:

```cpp
// ✅ CORRECT: Void return, C is out-parameter
using GemmFnPtr = void(*)(
    float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // A
    float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // B
    float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t   // C
);
```

### Key Takeaway

**IR correctness ≠ Runtime correctness**

The MLIR IR can be perfectly correct, but if the C++ function pointer typedef doesn't match the actual calling convention, you get runtime corruption.

Always verify:
1. Check what `buffer-results-to-out-params` produces
2. Match your typedef to the lowered signature
3. Test with actual data to catch ABI mismatches

---

# Part II: MLIR Bufferization Deep Dive

---

# Part II: MLIR Bufferization Deep Dive

## What is Bufferization?

**Bufferization** is the process of converting MLIR operations from **tensor semantics** (functional/immutable) to **memref semantics** (imperative/mutable).

### Tensor vs MemRef

| Aspect | Tensor (Functional) | MemRef (Imperative) |
|--------|---------------------|---------------------|
| **Semantics** | Immutable values (SSA) | Mutable memory buffers |
| **Operations** | Return new tensors | Modify buffers in-place |
| **Function Signature** | `func(tensor, tensor) -> tensor` | `func(memref, memref, memref)` |
| **Example** | `%result = linalg.matmul ins(%A, %B : tensor) outs(%C : tensor) -> tensor` | `linalg.matmul ins(%A, %B : memref) outs(%C : memref)` |
| **Optimization Level** | High-level (easier to transform) | Low-level (ready for execution) |

### Why Bufferize?

1. **Execution**: CPUs work with memory (memrefs), not abstract values (tensors)
2. **Performance**: Enables in-place updates, avoiding unnecessary copies
3. **Compilation**: LLVM backend expects memref operations

---

## The Challenge

When migrating from memref-based IR generation to tensor-based IR generation, we need to:

1. ✅ Generate clean tensor IR at the high level (done in `createGemmModuleTensor()`)
2. ⚠️ Convert tensor IR → memref IR during lowering (**the hard part**)
3. ✅ Ensure the resulting memref IR matches JIT expectations

### The Function Signature Problem

Our tensor IR generates:
```mlir
func.func @gemm(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
```

But our JIT expects:
```mlir
func.func @gemm(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>)
```

Notice:
- **Inputs**: 2 tensors → 2 memrefs ✅ (straightforward)
- **Output**: 1 return value → 3rd parameter ⚠️ (requires transformation)

This is the core challenge of bufferization!

---

## The Correct Approach

### Overview

```
Tensor IR
    ↓
[Phase 1: Register Bufferization Interfaces]
    ↓
[Phase 2: One-Shot Bufferize with Function Boundaries]
    ↓
[Phase 3: Convert Return to Out-Parameter]
    ↓
[Phase 4: Lower Bufferization Ops]
    ↓
MemRef IR (ready for LLVM lowering)
```

### Step-by-Step Implementation

#### **Step 1: Register Bufferization Interfaces**

```cpp
// Register bufferization interface implementations
// These tell one-shot-bufferize how to handle tensor ops
DialectRegistry registry;
arith::registerBufferizableOpInterfaceExternalModels(registry);
linalg::registerBufferizableOpInterfaceExternalModels(registry);
tensor::registerBufferizableOpInterfaceExternalModels(registry);
bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
context->appendDialectRegistry(registry);
```

**Why needed?**
- One-Shot Bufferize is extensible via `BufferizableOpInterface`
- Each dialect must register how its ops should be bufferized
- Without these, One-Shot Bufferize doesn't know how to handle `linalg.matmul`, `tensor.empty`, etc.

**What each registration does:**
- `arith`: Handles `arith.constant`, arithmetic ops on tensors
- `linalg`: Handles `linalg.matmul`, `linalg.fill` (the core computation)
- `tensor`: Handles `tensor.empty`, `tensor.dim`, `tensor.extract_slice`
- `func_ext`: Handles function calls and boundaries

#### **Step 2: One-Shot Bufferize with Function Boundaries**

```cpp
// Configure One-Shot Bufferize to handle function boundaries
bufferization::OneShotBufferizationOptions options;
options.bufferizeFunctionBoundaries = true;
options.setFunctionBoundaryTypeConversion(
    bufferization::LayoutMapOption::IdentityLayoutMap);

// "One-Shot Bufferize" converts all tensors to memrefs (including function args/results)
pm.addPass(bufferization::createOneShotBufferizePass(options));
```

**Why needed?**
- **Default behavior**: One-Shot Bufferize only bufferizes function *bodies*, leaving function signatures unchanged
- **Our need**: We need to convert the entire function, including arguments and return types
- **Solution**: Set `bufferizeFunctionBoundaries = true`

**What `bufferizeFunctionBoundaries` does:**
- Converts tensor function arguments → memref function arguments
- Converts tensor return types → memref return types
- Analyzes SSA use-def chains to minimize buffer copies
- Performs destination-passing style optimization

**Layout map option:**
- `IdentityLayoutMap`: Use identity layout maps (no strides/offsets complexity)
- Alternative would be `FullyDynamicLayoutMap` (more flexible but harder to work with)
- For our use case, identity layout is simpler and sufficient

**After this step, we get:**
```mlir
func.func @gemm(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) -> memref<?x?xf32>
```
(Notice: still returns a memref!)

#### **Step 3: Convert Return to Out-Parameter**

```cpp
// Convert memref function results to out-parameters
// This transforms: func(memref, memref) -> memref
// Into: func(memref, memref, memref) with output as 3rd parameter
pm.addPass(bufferization::createBufferResultsToOutParamsPass());
```

**Why needed?**
- JIT-compiled C++ function signature expects output as parameter
- Returning a memref would require memory allocation logic in caller
- Out-parameter style is standard for C ABI (what JIT uses)

**What it does:**
- Finds function return values of memref type
- Adds a new function parameter for each return value
- Replaces `return %result` with a copy to the out-parameter
- Updates call sites (if any)

**After this step, we get:**
```mlir
func.func @gemm(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>)
```
(Perfect! Matches JIT expectations)

#### **Step 4: Lower Bufferization Ops**

```cpp
// Lower remaining bufferization operations to memref  
pm.addPass(createBufferizationToMemRefPass());
pm.addPass(createCanonicalizerPass());  // Clean up
```

**Why needed?**
- One-Shot Bufferize may leave some `bufferization.*` ops in the IR
- These ops are abstractions that need to be lowered to concrete memref ops
- The LLVM conversion doesn't understand bufferization dialect

**What gets lowered:**
- `bufferization.to_tensor` → `memref.load` (if reading)
- `bufferization.to_memref` → `memref.cast` (if type conversion)
- `bufferization.alloc_tensor` → `memref.alloc`
- Any remaining bufferization abstractions → concrete memref operations

**Canonicalization:**
- Removes redundant operations
- Simplifies memref.cast chains
- Folds constants

---

## Why Other Approaches Failed

### ❌ **Attempt 1: Using `func-bufferize` First**

```cpp
// WRONG: This creates to_tensor ops without 'restrict' attribute
pm.addPass(func::createFuncBufferizePass());
pm.addPass(bufferization::createOneShotBufferizePass());
```

**Why it failed:**
- `func-bufferize` converts function signatures: `tensor` → `memref`
- But the function *body* still has tensors
- So it inserts `to_tensor` ops to convert arguments back: `memref` → `tensor`
- One-Shot Bufferize requires `to_tensor` ops to have `restrict` attribute
- `func-bufferize` doesn't add `restrict` attribute
- **Error**: `to_tensor ops without 'restrict' are not supported by One-Shot Analysis`

**Lesson**: Don't manually convert function signatures before One-Shot Bufferize. Let One-Shot Bufferize handle everything with `bufferizeFunctionBoundaries = true`.

### ❌ **Attempt 2: Running `buffer-results-to-out-params` Before Bufferization**

```cpp
// WRONG: Can't convert tensor returns to out-params
pm.addPass(bufferization::createBufferResultsToOutParamsPass());
pm.addPass(bufferization::createOneShotBufferizePass());
```

**Why it failed:**
- `buffer-results-to-out-params` only works on **memref** returns
- Our function has **tensor** returns at this point
- The pass simply does nothing (no-op)
- **Result**: Function signature unchanged

**Lesson**: `buffer-results-to-out-params` must run *after* bufferization, not before.

### ❌ **Attempt 3: Not Registering Bufferization Interfaces**

```cpp
// WRONG: Missing interface registrations
// (Just calling createOneShotBufferizePass without setup)
pm.addPass(bufferization::createOneShotBufferizePass());
```

**Why it failed:**
- One-Shot Bufferize encounters `linalg.matmul`, `tensor.empty`, etc.
- Without registered interfaces, it doesn't know how to bufferize these ops
- **Error**: `op was not bufferized` or `op does not implement BufferizableOpInterface`

**Lesson**: Always register bufferization interfaces for all dialects you're using.

### ❌ **Attempt 4: Not Enabling Function Boundary Bufferization**

```cpp
// WRONG: Default options don't bufferize function boundaries
pm.addPass(bufferization::createOneShotBufferizePass());
```

**Why it failed:**
- Default `bufferizeFunctionBoundaries = false`
- Function signature remains: `func(tensor, tensor) -> tensor`
- Function body is bufferized, but function boundary is not
- Creates `to_memref` and `to_tensor` at boundaries
- These boundary ops can't be lowered properly
- **Error**: `failed to legalize operation 'bufferization.to_memref'`

**Lesson**: Must explicitly set `bufferizeFunctionBoundaries = true` when you want to fully bufferize a function.

### ❌ **Attempt 5: Wrong Pass Ordering**

```cpp
// WRONG: Trying to lower bufferization ops before creating them
pm.addPass(createBufferizationToMemRefPass());
pm.addPass(bufferization::createOneShotBufferizePass());
```

**Why it failed:**
- Bufferization ops don't exist yet
- Pass does nothing
- Then One-Shot Bufferize runs and creates bufferization ops
- But the lowering pass already ran (too early)
- **Result**: Bufferization ops remain in IR, causing errors later

**Lesson**: Pass order matters! Bufferize first, then lower bufferization ops.

---

## Key Learnings

### 1. **Bufferization is a Multi-Step Process**

Don't expect a single pass to do everything. The correct pipeline has 4 distinct steps:
1. Register interfaces (setup)
2. One-Shot Bufferize (tensor → memref transformation)
3. Buffer results to out-params (signature adjustment)
4. Lower bufferization ops (cleanup)

### 2. **One-Shot Bufferize is Powerful but Requires Configuration**

- Default behavior: only bufferizes function bodies
- For full bufferization: set `bufferizeFunctionBoundaries = true`
- Always register dialect interfaces first
- Choose appropriate layout map strategy

### 3. **Function Boundaries are Special**

Converting function signatures is non-trivial:
- Tensor arguments → memref arguments (handled by One-Shot Bufferize)
- Tensor returns → memref out-parameters (needs `buffer-results-to-out-params`)
- Must maintain ABI compatibility with JIT caller

### 4. **Destination-Passing Style is Key**

One-Shot Bufferize works best with DPS ops:
- `linalg.matmul` has `outs` operand (destination)
- One-Shot Bufferize can reuse the `outs` buffer for the result
- Minimizes memory allocations and copies
- Why we use `linalg.fill` to initialize the output

### 5. **Error Messages are Helpful**

MLIR error messages point to the problem:
- `to_tensor ops without 'restrict' are not supported` → Don't use `func-bufferize`
- `op was not bufferized` → Missing interface registration
- `failed to legalize operation` → Missing pass in pipeline

### 6. **The Restrict Attribute Matters**

From the docs:
> The `restrict` attribute indicates that there is no other `to_tensor` or `materialize_in_destination` op with the same or an aliasing MemRef operand.

This gives strong aliasing guarantees to the analysis, enabling aggressive optimization.

### 7. **Modularity is a Feature, Not a Bug**

MLIR's bufferization is split across multiple passes for good reasons:
- **One-Shot Bufferize**: Core tensor→memref transformation
- **Buffer Results to Out-Params**: ABI adjustment
- **Bufferization to MemRef**: Dialect lowering
- Each pass has a single, well-defined responsibility
- Makes debugging easier (can inspect IR between passes)

---

## Verification

To verify bufferization is working correctly:

### 1. **Check Compilation Succeeds**
```bash
cmake --build build/x64-release
```

### 2. **Run Tests**
```bash
python3 test_jit.py
```
Expected: All tests pass with `0.00e+00` error

### 3. **Inspect IR at Each Stage**

Enable IR printing in `lowering.cpp`:
```cpp
pm.enableIRPrinting();
```

Then run and check:
- **After canonicalization**: Should see tensor IR
- **After one-shot-bufferize**: Should see memref args/results
- **After buffer-results-to-out-params**: Should see out-parameter signature
- **After bufferization-to-memref**: Should see no `bufferization.*` ops

---

## Summary: The Correct Recipe

```cpp
// 1. Register interfaces (MUST BE FIRST)
DialectRegistry registry;
arith::registerBufferizableOpInterfaceExternalModels(registry);
linalg::registerBufferizableOpInterfaceExternalModels(registry);
tensor::registerBufferizableOpInterfaceExternalModels(registry);
bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
context->appendDialectRegistry(registry);

// 2. One-Shot Bufferize with function boundaries (CORE TRANSFORMATION)
bufferization::OneShotBufferizationOptions options;
options.bufferizeFunctionBoundaries = true;
options.setFunctionBoundaryTypeConversion(
    bufferization::LayoutMapOption::IdentityLayoutMap);
pm.addPass(bufferization::createOneShotBufferizePass(options));

// 3. Convert returns to out-parameters (ABI ADJUSTMENT)
pm.addPass(bufferization::createBufferResultsToOutParamsPass());

// 4. Lower bufferization ops (CLEANUP)
pm.addPass(createBufferizationToMemRefPass());
pm.addPass(createCanonicalizerPass());
```

This is the **minimum viable bufferization pipeline** for tensor→memref conversion with function boundary handling.

---

## Quick Reference

### Running the Code

```bash
# Build the project
cmake --build build/x64-release

# Run the comprehensive test suite (includes API demo)
python3 test_jit.py

# Quick test
python3 -c "import llvm_example; import numpy as np; \
A = np.ones((2,3), dtype=np.float32); \
B = np.ones((3,2), dtype=np.float32); \
C = llvm_example.gemm(A, B); \
print('Result:', C)"

# Inspect generated IR
python3 -c "import llvm_example; print(llvm_example.test_ir_generation())"
python3 -c "import llvm_example; print(llvm_example.test_optimized_ir())"
```

### File Organization

```
llvm-example/
├── src/
│   ├── ir.cpp           - MLIR IR generation (tensor-based)
│   ├── lowering.cpp     - Optimization pipeline (bufferization!)
│   ├── jit.cpp          - JIT compilation and execution
│   └── bindings.cpp     - Python bindings (clean API)
├── test_jit.py          - Comprehensive test suite
├── BUFFERIZATION_GUIDE.md  - This document
└── README.md            - Project overview
```

### Key Concepts Checklist

When working with MLIR bufferization, remember:

- ✅ **Register interfaces first** - Required for One-Shot Bufferize
- ✅ **Enable function boundaries** - Set `bufferizeFunctionBoundaries = true`
- ✅ **Use buffer-results-to-out-params** - Converts returns to out-parameters
- ✅ **Match C++ typedef to lowered signature** - Avoid calling convention mismatches
- ✅ **Test with actual data** - IR correctness ≠ runtime correctness
- ✅ **Pass ordering matters** - Bufferize → out-params → lower bufferization ops
- ✅ **Clean API for users** - Hide implementation details in bindings

### Debugging Tips

**If compilation fails:**
1. Check IR at each pass stage (`pm.enableIRPrinting()`)
2. Verify all dialect interfaces are registered
3. Ensure passes are in correct order

**If runtime produces garbage:**
1. Check function typedef matches lowered signature (void vs struct return)
2. Verify memref descriptor expansion (7 params per memref)
3. Test parameter passing with debug prints
4. Check calling convention (System V AMD64 ABI on Linux)

**If tests fail with wrong values:**
1. Compare against NumPy with `np.allclose()`
2. Check for dimension mismatches (M, N, K)
3. Verify row-major layout (C-style arrays)
4. Inspect stride calculation (should be [cols, 1] for row-major)

---

## References

- [MLIR Bufferization Docs](https://mlir.llvm.org/docs/Bufferization/)
- [One-Shot Bufferize Paper](https://arxiv.org/abs/2202.03293)
- [BufferizableOpInterface](https://mlir.llvm.org/docs/Dialects/BufferizationOps/)
- [MLIR Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [MemRef Type](https://mlir.llvm.org/docs/Dialects/MemRef/)

---

*Last updated: November 2025*
*This guide reflects the successful implementation of tensor-based IR with full bufferization for JIT-compiled matrix multiplication.*
