# Chapter 5: Tensor-First Architecture with Linalg

This chapter introduces **tensor-first architecture** - the industry-standard approach for modern MLIR compilers. After learning foundational concepts in Chapters 1-4, we now adopt the patterns used in production systems like Torch-MLIR, IREE, and StableHLO.

## What You'll Learn

- **Tensor-First Pattern**: Using tensors at high level, bufferizing late
- **Linalg.generic**: Expressing computations with tensor operations
- **Tensor Dialect**: `tensor.empty`, `tensor.dim` for dynamic shapes
- **Bufferization Pipeline**: Automatic tensor → memref transformation
- **Function Semantics**: Returning tensors (functional) vs out-parameters (imperative)
- **Modern MLIR**: Industry-standard compilation patterns

## Key Architectural Shift

Starting from this chapter, we follow modern MLIR best practices:

**Chapters 1-4 (Foundation):** Direct memref operations to understand execution
**Chapter 5+ (Production):** Tensor operations with automatic bufferization

This shift aligns with how real ML compilers work and enables better optimization opportunities.

## The Kernel: SAXPY

**SAXPY** (Single-Precision A·X Plus Y) is a fundamental operation:

```
C[i] = α · A[i] + B[i]
```

We implement this using tensors and `linalg.generic`, showing the modern approach.

## Generated MLIR (Tensor-Based)

```mlir
#map = affine_map<(d0) -> (d0)>

func.func @saxpy(%alpha: f32,
                 %A: tensor<?xf32>,
                 %B: tensor<?xf32>) -> tensor<?xf32> {
  // Get dynamic size
  %c0 = arith.constant 0 : index
  %size = tensor.dim %A, %c0 : tensor<?xf32>
  
  // Create empty output tensor
  %empty = tensor.empty(%size) : tensor<?xf32>
  
  // Compute with linalg.generic (tensor operations)
  %result = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%A, %B : tensor<?xf32>, tensor<?xf32>)
    outs(%empty : tensor<?xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %scaled = arith.mulf %alpha, %a : f32
    %sum = arith.addf %scaled, %b : f32
    linalg.yield %sum : f32
  } -> tensor<?xf32>
  
  return %result : tensor<?xf32>
}
```

**Key Features:**
- Function **returns** a tensor (functional semantics)
- All operations on **immutable tensors** (no in-place updates)
- `linalg.generic` with tensor types
- Clean, optimizable high-level IR

## Lowering Pipeline (Tensor-First)

The modern MLIR pipeline with bufferization:

```
┌─────────────────────────────────────────────┐
│  High-Level IR (Tensors)                    │
│  • linalg.generic with tensor<?xf32>        │
│  • Functional semantics (immutable)         │
│  • Returns result tensor                    │
└───────────────────┬─────────────────────────┘
                    │ Canonicalize (simplify)
                    ▼
┌─────────────────────────────────────────────┐
│  Bufferization Phase                        │
│  • One-Shot Bufferize: tensor → memref      │
│  • Buffer-Results-To-Out-Params             │
│  • Returns → out-parameters                 │
└───────────────────┬─────────────────────────┘
                    │ Bufferization complete
                    ▼
┌─────────────────────────────────────────────┐
│  MemRef IR (After Bufferization)            │
│  • linalg.generic with memref<?xf32>        │
│  • Out-parameter signature                  │
└───────────────────┬─────────────────────────┘
                    │ Linalg-to-Loops
                    ▼
┌─────────────────────────────────────────────┐
│  Explicit Loops (SCF)                       │
│  • scf.for with memref.load/store           │
│  • Explicit control flow                    │
└───────────────────┬─────────────────────────┘
                    │ SCF-to-CF
                    ▼
┌─────────────────────────────────────────────┐
│  Control Flow (CF)                          │
│  • Branches (cf.br, cf.cond_br)             │
└───────────────────┬─────────────────────────┘
                    │ Convert-to-LLVM
                    ▼
┌─────────────────────────────────────────────┐
│  LLVM Dialect                               │
│  • llvm.* operations                        │
└───────────────────┬─────────────────────────┘
                    │ JIT Compile
                    ▼
              [Native Code]
```

**Key Insight:** Bufferization happens automatically in the middle of the pipeline. You write tensor code, and the compiler handles the transformation to efficient memref code.

## Why Tensor-First?

### 1. Industry Standard
- **Torch-MLIR**: PyTorch → Tensor IR → Bufferization
- **IREE**: All frontends use tensor operations
- **StableHLO**: TensorFlow/JAX use tensor semantics

### 2. Better Optimization
Tensor operations enable:
- **Fusion**: Combine multiple tensor ops → single op
- **Algebraic Simplification**: Rewrite patterns on immutable values
- **Dead Code Elimination**: Easier with functional semantics

### 3. Cleaner Semantics
```mlir
// Tensor (functional): Easy to reason about
%result = linalg.generic {...} ins(%A, %B) -> tensor<?xf32>

// vs Memref (imperative): Requires alias analysis
linalg.generic {...} ins(%A, %B) outs(%C)  // Mutates C in-place
```

### 4. Framework Alignment
ML frameworks (PyTorch, TensorFlow, JAX) think in tensors:
```python
# PyTorch
result = alpha * A + B  # Returns new tensor

# MLIR (tensor-first)
%result = linalg.generic {...} -> tensor<?xf32>  # Returns new tensor
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

## Comparison: Tensor-First vs Direct MemRef

| Aspect | Chapters 1-4 (MemRef) | Chapter 5+ (Tensor) |
|--------|----------------------|---------------------|
| **Types** | `memref<?xf32>` | `tensor<?xf32>` |
| **Semantics** | Imperative (mutable) | Functional (immutable) |
| **Functions** | `void f(memref, memref)` | `tensor f(tensor)` |
| **Optimization** | Limited (aliasing concerns) | Extensive (functional) |
| **Lowering** | Direct to loops | Bufferize → loops |
| **Industry Use** | Low-level codegen | High-level IR |

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

You'll see:
- High-level tensor IR (clean, functional)
- Lowered LLVM IR (after complete pipeline)
- Test results (all should pass)

## Implementation Details

### IR Generation (src/ir.cpp)
- Uses `RankedTensorType::get({ShapedType::kDynamic}, f32)`
- Creates `linalg.generic` with tensor inputs/outputs
- Returns tensor result (functional)

### Lowering Pipeline (src/lowering.cpp)
- Registers bufferization interface implementations
- Configures One-Shot Bufferize with `bufferizeFunctionBoundaries`
- Adds Buffer-Results-To-Out-Params pass
- Complete chain: Canonicalize → Bufferize → Linalg-to-Loops → SCF-to-CF → LLVM

### JIT Execution (src/jit.cpp)
- Transparent: Python bindings allocate output buffer
- Internal: Bufferized function uses out-parameter
- User never sees memref details

## Next Steps

- **Chapter 6**: Softmax with tensor operations and reductions
- **Chapter 7**: Neural network operations (ReLU, Conv2D) with tensors
- **Chapter 8**: Custom dialects defining tensor-based operations
- **Chapter 9**: TableGen for tensor dialect definitions
- **Chapter 10**: Two-level optimization (tensor + memref levels)

**Key Takeaway:** From Chapter 5 onward, all operations use tensor-first architecture. This is the pattern you'll see in production MLIR compilers.