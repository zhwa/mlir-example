# Chapter 10: Compiler Optimizations

## Overview

Demonstrates **backend optimizations** on Chapter 9's NN dialect:
- ✅ Same operations (`nn.matmul`, `nn.add`, `nn.relu`)
- ✅ Same Python API (`forward()`)
- ✅ **New**: Fusion + auto-vectorization

**Key Insight**: Real ML frameworks keep stable APIs while improving backends.

## Optimizations Added

### 1. Linalg Fusion
Merges adjacent operations to reduce memory traffic.

### 2. Loop Invariant Code Motion
Hoists computations out of loops to avoid redundancy.

### 3. Auto-Vectorization Support
Enables LLVM's auto-vectorizer by preparing loops (LICM) and providing vector lowering passes.
Uses MLIR's `vector` dialect for SIMD instructions - converts scalar loops to vector operations that process 8 floats at once (AVX2).

## File Structure

**Chapter 10 reuses Chapter 9's NN dialect entirely**:

- `src/bindings.cpp` - Only file! (465 lines, includes optimized pipeline)
- `CMakeLists.txt` - Links against Chapter 9's libNNDialect.a
- `test_jit.py` - Correctness tests

**vs Chapter 9**: 1 file instead of 9 (dialect code reused)

## Compilation Pipeline

```cpp
// Chapter 9: Basic pipeline
pm.addPass(createConvertNNToStandardPass());
pm.addPass(createCanonicalizerPass());
pm.addPass(createConvertLinalgToLoopsPass());
// ... lower to LLVM

// Chapter 10: Optimized pipeline
pm.addPass(createConvertNNToStandardPass());
pm.addPass(createCanonicalizerPass());

// NEW: Fusion
pm.addPass(createLinalgGeneralizationPass());
pm.addPass(createCanonicalizerPass());
pm.addPass(createLinalgElementwiseOpFusionPass());
pm.addPass(createCanonicalizerPass());

pm.addPass(createConvertLinalgToLoopsPass());

// NEW: Loop optimization
pm.addPass(createLoopInvariantCodeMotionPass());
pm.addPass(createCanonicalizerPass());

// NEW: Vectorization
pm.addPass(createConvertVectorToSCFPass());
pm.addPass(createCanonicalizerPass());
pm.addPass(createConvertVectorToLLVMPass());

// ... rest of lowering
```

## API Usage

```python
import ch10
import numpy as np

# Build computation graph (same as Chapter 9)
A = ch10.Tensor(np.random.randn(128, 256).astype(np.float32))
B = ch10.Tensor(np.random.randn(256, 128).astype(np.float32))
C = ch10.matmul(A, B)

# Execute (optimizations applied automatically)
result = ch10.forward(C)
```

**No API changes** - optimizations are transparent!

## Building

```bash
cmake --build build/x64-release --target ch10
```

## Running Tests

```bash
cd ch.10.Optimizations
python3 test_jit.py
```

Expected output:
```
======================================================================
 Chapter 10: Optimized Compilation
 Fusion + Vectorization with vector dialect
======================================================================

Test 1: Matrix Multiplication (128x256 @ 256x128)
✓ MatMul correctness verified

Test 2: Fused MatMul + ReLU (256x512 @ 512x256)
✓ Fused MatMul+ReLU correctness verified

Test 3: Element-wise Fusion (A+B)*(C+D)
✓ Element-wise fusion correctness verified

======================================================================
 All tests passed!
 Optimizations: Linalg fusion + vectorization (vector dialect)
======================================================================
```

## Learning Outcomes

- ✅ **API stability**: High-level code unchanged, backend evolves
- ✅ **Progressive lowering**: Optimizations at the right abstraction
- ✅ **Operation fusion**: Eliminating intermediate buffers
- ✅ **Explicit vectorization**: Using `vector` dialect for SIMD
- ✅ **Pass ordering**: Why canonicalization matters

## Relation to Production

| Framework | API | Optimizations | Backend |
|-----------|-----|---------------|---------|
| **PyTorch 2.0** | `torch.nn` | Fusion, inlining | Inductor → Triton |
| **JAX** | `jax.numpy` | Fusion, layout | XLA → LLVM/CUDA |
| **TensorFlow** | `tf.keras` | XLA passes | TPU/GPU/CPU |
| **This Chapter** | `ch10.Tensor` | Fusion, vectorization | LLVM |

Same pattern: **stable API + improving backend**.