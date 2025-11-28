# Chapter 5: Vector Operations & SCF Dialect

This chapter demonstrates the **SCF (Structured Control Flow)** dialect for explicit looping in MLIR.

## What You'll Learn

- **SCF Dialect**: Structured loops (`scf.for`) vs high-level operations (`linalg.generic`)
- **MemRef Ranks**: Understanding how rank determines the shape
  - **1D (Vector)**: `memref<?xf32>` - one dimension `{ShapedType::kDynamic}`
  - **2D (Matrix)**: `memref<?x?xf32>` - two dimensions `{kDynamic, kDynamic}`
  - **3D+ (Tensors)**: `memref<?x?x?xf32>` - three or more dimensions
- **Dynamic vs Static Shapes**: 
  - `?` (kDynamic) means runtime-determined dimensions
  - Fixed numbers (e.g., `memref<8x32xf32>`) mean compile-time constants
- **MemRef Queries**: Using `memref.dim` to get sizes at runtime
- **Loop Construction**: Building explicit for loops in MLIR IR

## The Kernel: SAXPY

**SAXPY** (Single-Precision A·X Plus Y) is a fundamental operation:

```
C[i] = α · A[i] + B[i]
```

This is simpler than matrix multiplication but teaches important concepts about loops and control flow.

## Key Concepts

### SCF vs Linalg

**Linalg** (Chapter 1-4): High-level, declarative
```mlir
linalg.generic { ... }  // Compiler decides how to loop
```

**SCF** (This chapter): Explicit control flow
```mlir
scf.for %i = %c0 to %size step %c1 {
  // You control the loop body
}
```

### When to Use Each

- **Linalg**: Complex operations where compiler optimizations matter (matmul, convolution)
- **SCF**: Simple operations where you want explicit control (element-wise, reductions)

## Generated MLIR

```mlir
func.func @saxpy(%alpha: f32, 
                 %A: memref<?xf32>,
                 %B: memref<?xf32>, 
                 %C: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %size = memref.dim %A, %c0 : memref<?xf32>
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %size step %c1 {
    %a = memref.load %A[%i] : memref<?xf32>
    %b = memref.load %B[%i] : memref<?xf32>
    %scaled = arith.mulf %alpha, %a : f32
    %result = arith.addf %scaled, %b : f32
    memref.store %result, %C[%i] : memref<?xf32>
  }
  return
}
```

## Lowering Pipeline

```
High-Level MLIR (scf.for)
  ↓ scf-to-cf
Control Flow (cf.br)
  ↓ convert-to-llvm
LLVM Dialect
  ↓ mlir-translate
LLVM IR
  ↓ JIT compile
Native Code
```

## Usage

```python
import numpy as np
import ch5_vector_ops

alpha = 2.0
A = np.array([1, 2, 3, 4], dtype=np.float32)
B = np.array([5, 6, 7, 8], dtype=np.float32)

C = ch5_vector_ops.saxpy(alpha, A, B)
# Result: [7, 10, 13, 16]
```

## Build

From the project root:
```bash
cmake --build build --config Release
```

## Test

```bash
cd ch.5.Vector-ops
python test_jit.py
```

## Comparison with Chapter 1

| Aspect | Chapter 1 (GEMM) | Chapter 5 (SAXPY) |
|--------|------------------|-------------------|
| Operation | Matrix multiply | Vector add |
| Dialect | Linalg | SCF |
| Shapes | Static (8×32) | Dynamic (?xf32) |
| Complexity | High-level | Explicit loops |
| Optimization | Compiler decides | You control |

## Next Steps

- **Chapter 6**: Add `math` dialect for `exp`, `log`, implement softmax
- **Chapter 7**: Define custom dialects in Python with xDSL
- **Chapter 8**: Compare Python (xDSL) vs C++ (TableGen) approaches