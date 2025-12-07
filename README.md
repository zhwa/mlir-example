# MLIR Learn-by-Doing Tutorial

A hands-on tutorial demonstrating MLIR JIT compilation, progressing from basic matrix operations to neural network operators with computation graphs.

## Quick Start

### Prerequisites

```bash
# Ubuntu/WSL2
sudo apt install -y libmlir-18-dev libmlir-18 mlir-18-tools
sudo apt install -y python3-dev python3-numpy ninja-build libzstd-dev
sudo apt install -y clang-18  # or g++

# Fix LLVM header conflict (required for pybind11)
sudo mv /usr/lib/llvm-18/include/cxxabi.h /usr/lib/llvm-18/include/cxxabi.h.backup
```

### Build All Chapters

```bash
# Configure (one time)
cmake --preset x64-release

# Build
cmake --build --preset x64-release
```

Outputs: `build/x64-release/ch.*/ch*_*.so`

### Test a Chapter

```bash
cd ch.1.Fixed-size
python3 test_jit.py
```

Expected output:
```
Using build directory: ../build/x64-release/ch.1.Fixed-size
✓ Success! Result shape: (8, 16)
✓ All values correct!
✓ Results match NumPy (within float32 precision)!
=== All tests complete ===
```

## Chapter Progression

### Chapter 1: Fixed-Size Matrices
**Goal**: Understand MLIR basics with hardcoded 8×32 × 32×16 multiplication

**Key Concepts**:
- MLIR IR generation (`linalg.matmul`)
- Optimization pipeline (affine, linalg, SCF)
- JIT execution with mlir::ExecutionEngine
- Python binding basics

**Documentation**: `ch.1.Fixed-size/README.md`, `MLIR_CODING_PATTERN.md`

### Chapter 2: Dynamic Shapes
**Goal**: Support arbitrary matrix dimensions at runtime

**Key Concepts**:
- Dynamic shape syntax: `tensor<?x?xf32>`
- Runtime dimension passing (21-parameter memref ABI)
- Shape-agnostic IR generation
- Out-parameter bufferization pattern

### Chapter 3: JIT Caching
**Goal**: Optimize performance by caching compiled functions

**Key Concepts**:
- Single compilation for all shapes
- Function pointer caching
- Performance measurement (first call vs cached)
- Shape flexibility without recompilation

### Chapter 4: Tensor Bufferization
**Goal**: Master advanced MLIR bufferization strategies

**Key Concepts**:
- Tensor vs memref semantics
- One-shot bufferization
- Buffer-results-to-out-params pass
- Clean Python API design

**Documentation**: `ch.4.Tensor-bufferization/README.md`, `BUFFERIZATION_GUIDE.md`

### Chapter 5: Vector Operations & SCF Dialect
**Goal**: Learn structured control flow with explicit loops

**Kernel**: SAXPY (Scalar A times X Plus Y): `C[i] = α·A[i] + B[i]`

**Key Concepts**:
- SCF (Structured Control Flow) dialect
- `scf.for` loops with induction variables
- `memref.dim` for dynamic size queries
- Comparison: SCF explicit loops vs Linalg high-level ops

**Documentation**: `ch.5.Vector-ops/README.md`

### Chapter 6: Softmax with Math Dialect
**Goal**: Implement softmax using Math dialect for mathematical functions

**Kernel**: Softmax: `output[i] = exp(input[i] - max) / sum(exp(...))`

**Key Concepts**:
- Math dialect operations (`math.exp`)
- Multi-pass algorithms (find max, compute exp/sum, normalize)
- Numerical stability techniques
- Loop-carried variables in `scf.for`
- Math-to-LLVM and Math-to-Libm lowering strategies

**Documentation**: `ch.6.Softmax/README.md`

### Chapter 7: Neural Operations (Operator Composition)
**Goal**: Build computation graphs with deferred execution before introducing custom dialects

**Operations**: Element-wise (add, mul), matrix multiplication, activations (ReLU, softmax)

**Key Concepts**:
- Computation graph with symbolic operation tracking
- Deferred execution model (build → compile → execute)
- Recursive IR generation from graph structure
- Operation composition for multi-layer networks
- MLIR calling convention for static memrefs (5 params for 1D, 7 for 2D)
- Critical lesson: Math dialect lowering pass order (math-to-llvm THEN math-to-libm)

**Documentation**: `ch.7.Neural-ops/README.md`

### Chapter 8: Universal Bindings with libffi
**Goal**: Implement production-grade universal Python bindings

**Key Concepts**:
- libffi-based marshalling (eliminates per-kernel wrappers)
- Dynamic FFI dispatch for ANY function signature
- ~60% code reduction vs explicit parameter cases
- Python string-based custom dialect workflow
- Comparison with industrial compilers (IREE)

**Documentation**: `ch.8.Custom-dialect/README.md`

### Chapter 9: TableGen Custom Dialect
**Goal**: Build production-grade custom dialect using TableGen/ODS

**Key Concepts**:
- TableGen dialect definition (NNDialect.td, NNOps.td)
- Operation Definition Specification (ODS)
- OpBuilder for programmatic IR construction
- Memref-based operations (nn.add, nn.mul, nn.matmul, nn.relu)
- Flat directory structure (inc/, src/)
- PyTorch-like API (`ch9.forward()`)

**Documentation**: `ch.9.TableGen-dialect/README.md`, `TUTORIAL.md`


## Key Implementation Details

### Python Binding Layer (`bindings.cpp`)

```cpp
py::array_t<float> gemm(py::array_t<float> A, py::array_t<float> B) {
  // Extract dimensions
  int64_t M = A.shape[0];
  int64_t K = A.shape[1];
  int64_t N = B.shape[1];

  // Allocate output array (hidden from user!)
  auto C = py::array_t<float>({M, N});

  // Call JIT function with C as out-parameter
  mlir::executeGemm(A.ptr, B.ptr, C.ptr, M, N, K);

  return C;  // Return to Python
}
```

The Python binding:
1. Accepts two NumPy arrays (A, B)
2. **Automatically allocates** the output array C
3. Calls the JIT function with C as an out-parameter
4. Returns C to Python

This design gives users a clean API while using efficient out-parameter style internally.

### JIT Function Signature (Internal)

After MLIR bufferization and lowering, the compiled function has this signature:

```cpp
// LLVM-level signature (after buffer-results-to-out-params pass)
void gemm(
    float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // A memref
    float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // B memref
    float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t   // C memref
);
```

Each `memref<?x?xf32>` expands to 7 parameters:
- `ptr, ptr, offset, size0, size1, stride0, stride1`

Total: 3 memrefs × 7 params = **21 parameters**

The function takes C as the third memref parameter (out-parameter style) and returns `void`.

### Why Out-Parameters?

MLIR's bufferization pipeline uses the `buffer-results-to-out-params` pass to convert:

```mlir
// Before: Return value
func.func @gemm(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) 
    -> memref<?x?xf32>

// After: Out-parameter
func.func @gemm(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                %arg2: memref<?x?xf32>)
```

This is necessary because:
1. **Memory management**: Returning a memref would require the caller to know where it was allocated
2. **ABI compatibility**: Struct returns (memref descriptors) have complex calling conventions
3. **Performance**: Out-parameters avoid copying large descriptors on the stack

## Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│ Python User Code                                        │
│   C = gemm.gemm(A, B)  ← Clean API!                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Python Binding (bindings.cpp)                           │
│   • Allocates C internally                              │
│   • Calls executeGemm(A, B, C, M, N, K)                 │
│   • Returns C to Python                                 │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ JIT Executor (jit.cpp)                                  │
│   • Expands memrefs to 21 parameters                    │
│   • Calls: gemm(A[7], B[7], C[7])                       │
│   • C is out-parameter (void return)                    │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ MLIR Bufferization Pipeline (lowering.cpp)              │
│   1. One-Shot Bufferize (tensor → memref)               │
│   2. Buffer-Results-To-Out-Params (return → out-param)  │
│   3. Bufferization-To-MemRef (finalize memrefs)         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Generated Machine Code                                  │
│   • malloc buffer                                       │
│   • memset(buffer, 0)                                   │
│   • matmul(A, B, buffer)                                │
│   • memcpy(buffer, C_out_param)                         │
└─────────────────────────────────────────────────────────┘
```

The user sees a clean functional API, but internally it uses efficient imperative memory operations with out-parameters.

## Troubleshooting

### "Could not find LLVM/MLIR"
```bash
sudo apt install libmlir-18-dev mlir-18-tools
```

### "cxxabi.h duplicate declarations"
```bash
sudo mv /usr/lib/llvm-18/include/cxxabi.h /usr/lib/llvm-18/include/cxxabi.h.backup
```

### "Module not found" when running tests
Ensure you're in the chapter directory:
```bash
cd ch.1.Fixed-size  # Not root!
python3 test_jit.py
```

### Build fails with "pybind11 not found"
pybind11 is auto-fetched. If fetch fails:
```bash
rm -rf build/deps
cmake --preset x64-release  # Re-fetch
```

## Python API Examples

All chapters export the same clean API:

```python
import numpy as np

# Chapter 1 (fixed size only)
import ch1_fixed_size
A = np.ones((8, 32), dtype=np.float32)
B = np.ones((32, 16), dtype=np.float32)
C = ch1_fixed_size.gemm(A, B)  # Returns (8, 16)

# Chapters 2-4 (any size)
import ch2_dynamic_size
A = np.random.rand(100, 50).astype(np.float32)
B = np.random.rand(50, 25).astype(np.float32)
C = ch2_dynamic_size.gemm(A, B)  # Returns (100, 25)
```