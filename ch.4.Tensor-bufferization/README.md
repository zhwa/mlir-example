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

## Project Structure

```
ch.4.Tensor-bufferization/
├── src/
│   ├── ir.cpp              # Tensor-based MLIR IR generation
│   ├── lowering.cpp        # Bufferization and optimization pipeline
│   ├── jit.cpp             # JIT compilation and execution
│   └── bindings.cpp        # Python bindings
├── test_jit.py             # Test script
├── BUFFERIZATION_GUIDE.md  # Deep dive into bufferization
└── README.md               # This file
```

## Further Reading

- **BUFFERIZATION_GUIDE.md** - Comprehensive guide to MLIR bufferization, including:
  - API design patterns
  - Bufferization pipeline details
  - Common pitfalls and how to avoid them
  - Troubleshooting guide

## References

- [MLIR Bufferization Docs](https://mlir.llvm.org/docs/Bufferization/)
- [One-Shot Bufferize Paper](https://arxiv.org/abs/2202.03293)
- [Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
