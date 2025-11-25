# Chapter 4: Tensor-Based IR with Bufferization

This chapter demonstrates MLIR's tensor-based approach with **bufferization** - converting high-level immutable tensor operations into low-level mutable memory operations. This is the preferred approach for production MLIR code as it enables better optimization opportunities.

## What This Chapter Demonstrates

This chapter shows the MLIR compilation stack with **tensor-based IR and bufferization**:

1. **Tensor-Based IR** (`src/ir.cpp`) - Creates MLIR with `linalg.matmul` on tensors (`tensor<?x?xf32>`)
2. **Bufferization** (`src/lowering.cpp`) - Converts tensors to memrefs using One-Shot Bufferize
3. **Optimization Pipeline** - Multi-pass lowering: Tensor → MemRef → Loops → LLVM dialect
4. **JIT Execution** (`src/jit.cpp`) - LLJIT-based compilation with runtime dimensions
5. **Python Bindings** (`src/bindings.cpp`) - NumPy-compatible interface

**Key Learning:** Tensor-based approach with bufferization enables better optimization opportunities compared to direct memref-based IR (Chapter 1-3).

## Key Differences from Previous Chapters

| Aspect | Chapters 1-3 | Chapter 4 |
|--------|-------------|-----------|
| **IR Type** | MemRef (imperative) | Tensor (functional) |
| **Semantics** | Mutable buffers | Immutable values (SSA) |
| **Optimization** | Limited | Better (functional semantics) |
| **Transformation** | Direct lowering | Bufferization required |
| **Complexity** | Simpler | More sophisticated |

## Usage Example

```python
import ch4_tensor_bufferization as gemm
import numpy as np

# Works with any size!
A = np.ones((10, 20), dtype=np.float32)
B = np.ones((20, 15), dtype=np.float32)
C = gemm.gemm(A, B)  # Returns 10×15 matrix
```

## Exploring the IR

```python
import ch4_tensor_bufferization as gemm

# See the high-level tensor-based MLIR
print(gemm.test_ir_generation())
# Output: func.func @gemm(%arg0: tensor<?x?xf32>, ...)

# See the optimized/lowered LLVM dialect (after bufferization)
print(gemm.test_optimized_ir())
```

### Understanding the Pipeline

1. **High-level IR** (`src/ir.cpp`): 
   - `linalg.matmul` operates on `tensor<?x?xf32>` types
   - Functional/immutable semantics
   - Shape-polymorphic (works with any compatible sizes)

2. **Bufferization** (`src/lowering.cpp`):
   - One-Shot Bufferize: tensor → memref transformation
   - Buffer-Results-To-Out-Params: ABI adjustment
   - Bufferization-To-MemRef: dialect lowering

3. **Optimization Pipeline**:
   - Linalg → Loops: Convert to explicit loops
   - SCF → CF: Convert to control flow
   - MemRef/Arith/Func/CF → LLVM: Complete lowering

4. **JIT Execution** (`src/jit.cpp`):
   - LLJIT compiles LLVM dialect to native code
   - Runtime dimensions passed through memref descriptors
   - Direct function pointer invocation

## Where Does Bufferization Fit?

An important insight: **bufferization happens entirely within the MLIR optimization pipeline** and is invisible to the outer layers. The JIT interface and function signatures remain identical between Chapter 2 (direct memref) and Chapter 4 (tensor + bufferization).

### The Complete Picture

```
┌───────────────────────────────────────────────────────────┐
│                    Python Layer                           │
│  gemm.gemm(A, B) → calls C++ bindings                     │
└──────────────────────┬────────────────────────────────────┘
                       │ (Same in both chapters)
                       ▼
┌───────────────────────────────────────────────────────────┐
│                   C++ Bindings                            │
│  Allocates output C, extracts M, N, K                     │
│  Calls: executeGemm(A, B, C, M, N, K)                     │
└──────────────────────┬────────────────────────────────────┘
                       │ (Same in both chapters)
                       ▼
┌───────────────────────────────────────────────────────────┐
│                    JIT Layer                              │
│  Compiles to native function, calls:                      │
│  gemm_func(A[7], B[7], C[7])  ← 21 params, all memrefs    │
└──────────────────────┬────────────────────────────────────┘
                       │ (Same in both chapters!)
                       ▼
┌───────────────────────────────────────────────────────────┐
│          MLIR Optimization Pipeline (DIFFERS!)            │
│                                                           │
│  Chapter 2:                    Chapter 4:                 │
│  ┌──────────────────┐          ┌──────────────────┐       │
│  │ Generate MemRef  │          │ Generate Tensor  │       │
│  │    IR directly   │          │   IR (functional)│       │
│  └────────┬─────────┘          └────────┬─────────┘       │
│           │                             │                 │
│           │                             ▼                 │
│           │                    ┌──────────────────┐       │
│           │                    │  Bufferization   │       │
│           │                    │ Tensor→MemRef    │       │
│           │                    └────────┬─────────┘       │
│           │                             │                 │
│           └──────────┬──────────────────┘                 │
│                      │ Both produce same memref IR        │
│                      ▼                                    │
│           ┌──────────────────┐                            │
│           │ Linalg→Loops→LLVM│                            │
│           └──────────┬───────┘                            │
│                      │                                    │
│                      ▼                                    │
│           ┌──────────────────┐                            │
│           │    memref IR     │                            │
│           │ with out-param   │                            │
│           └──────────────────┘                            │
└───────────────────────────────────────────────────────────┘
                       │ (Same output!)
                       ▼
          [Native x86_64 Machine Code]
```

### Key Insight

**Both chapters produce the exact same final function signature:**
```cpp
void gemm(
    float* A_ptr, float* A_align, int64_t A_offset, 
    int64_t A_M, int64_t A_K, int64_t A_stride0, int64_t A_stride1,
    float* B_ptr, float* B_align, int64_t B_offset, 
    int64_t B_K, int64_t B_N, int64_t B_stride0, int64_t B_stride1,
    float* C_ptr, float* C_align, int64_t C_offset, 
    int64_t C_M, int64_t C_N, int64_t C_stride0, int64_t C_stride1
);
```

The **only** difference is **inside** the optimization pipeline:
- **Chapter 2:** Generates `memref<?x?xf32>` directly → simple lowering
- **Chapter 4:** Generates `tensor<?x?xf32>` → bufferization converts to `memref<?x?xf32>` → same lowering

### Why Use Bufferization Then?

Despite producing the same final output, the tensor-based approach with bufferization offers significant advantages:

1. **Better optimization opportunities** - Tensor IR uses functional (immutable) semantics, making it easier for compilers to reason about data dependencies and apply transformations safely
2. **Production standard** - Real-world MLIR projects (TensorFlow, PyTorch bridges, JAX) use tensor IR as their high-level representation
3. **Composability** - Tensor operations compose better with other high-level dialects and enable more sophisticated program transformations
4. **Future-proofing** - Advanced MLIR features (automatic parallelization, polyhedral optimization, distributed execution) work better with tensor IR

**Bottom line:** Bufferization is an internal compiler technique that lets you write cleaner, more optimizable high-level IR while still generating efficient low-level code.