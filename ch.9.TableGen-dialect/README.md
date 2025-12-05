# Chapter 9: Custom Dialect with TableGen

Production-grade custom MLIR dialect using TableGen (Operation Definition Specification) - the standard approach for industrial MLIR compiler development.

📖 **For detailed comparison with Chapter 8 and in-depth tutorial, see [TUTORIAL.md](TUTORIAL.md)**

## Quick Overview

This chapter demonstrates how real-world MLIR projects define custom dialects using TableGen, automatic C++ code generation, and type-safe IR transformation patterns.

**Key Difference from Chapter 8**:
- Chapter 8: Python string-based prototyping (fast iteration, learning)
- Chapter 9: C++ TableGen production (type safety, industrial scale)

## NN Dialect Operations

The `nn` (Neural Network) dialect provides high-level memref-based operations:

| Operation | Syntax | Description |
|-----------|--------|-------------|
| `nn.add` | `nn.add %a, %b, %out : memref<...>, memref<...>, memref<...>` | Element-wise addition |
| `nn.mul` | `nn.mul %a, %b, %out : memref<...>, memref<...>, memref<...>` | Element-wise multiplication |
| `nn.matmul` | `nn.matmul %a, %b, %out : memref<...>, memref<...>, memref<...>` | Matrix multiplication |
| `nn.relu` | `nn.relu %x, %out : memref<...>, memref<...>` | ReLU activation |

**Note**: All operations use **output-parameter style** (memref-based) to avoid tensor bufferization complexity.

## Project Structure

```
ch.9.TableGen-dialect/
├── include/NN/
│   ├── NNDialect.td       # Dialect definition (TableGen)
│   ├── NNOps.td           # Operation definitions (TableGen)
│   ├── NNDialect.h        # C++ dialect header
│   ├── NNOps.h            # C++ ops header
│   └── NNToStandard.h     # Lowering pass header
├── lib/
│   ├── NN/
│   │   ├── NNDialect.cpp  # Dialect implementation
│   │   └── NNOps.cpp      # Op implementations
│   └── Conversion/
│       └── NNToStandard.cpp # Lowering patterns
├── python/
│   └── bindings.cpp       # Python bindings (pybind11)
├── CMakeLists.txt         # Build configuration with TableGen
├── test_jit.py            # Test suite
├── README.md              # This file
└── TUTORIAL.md            # Detailed tutorial and comparison
```

## Building

The project uses TableGen to generate code during build:

```bash
cd build/x64-release
cmake --build . --target ch9 -j 8
```

CMake automatically:
1. Runs `mlir-tblgen` on `.td` files
2. Generates `.inc` files
3. Compiles C++ sources
4. Links Python module

## Usage

```python
import ch9
import numpy as np

# Create tensors from numpy arrays
a = ch9.Tensor(np.array([1., 2., 3., 4.], dtype=np.float32))
b = ch9.Tensor(np.array([5., 6., 7., 8.], dtype=np.float32))

# Use operator overloading - builds computation graph
c = a + b

# Compile and execute - OpBuilder creates IR directly
result = ch9.compile(c)
print(result)  # [6. 8. 10. 12.]

# Chain operations
d = (a + b) * a
result = ch9.compile(d)

# Matrix operations
A = ch9.Tensor(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
B = ch9.Tensor(np.array([[5., 6.], [7., 8.]], dtype=np.float32))
C = ch9.matmul(A, B)
result = ch9.compile(C)
```

## Compilation Pipeline

The Pythonic API uses **industrial-grade IR construction**:

1. **Graph Building**: Python operations (`a + b`) build computation graph
2. **IR Construction**: OpBuilder creates MLIR IR directly (no string generation!)
3. **Lowering**: NN dialect → Linalg → Loops → LLVM
4. **JIT Compilation**: LLVM IR → native machine code
5. **Execution**: Call generated function with libffi

**Key Difference from Tutorials**: We use `OpBuilder` to construct IR objects directly, just like Torch-MLIR, JAX, and IREE. No MLIR text parsing overhead!

## Testing

```bash
cd ch.9.TableGen-dialect
python3 test.py
```

Expected output:
```
======================================================================
Chapter 9: Custom Dialect with TableGen
======================================================================

### Test 1: Tensor Addition (a + b) ###
✓ [1. 2. 3. 4.] + [5. 6. 7. 8.] = [ 6.  8. 10. 12.]

### Test 2: Tensor Multiplication (a * b) ###
✓ [2. 3. 4. 5.] * [10. 10. 10. 10.] = [20. 30. 40. 50.]

... (more tests)

All tests passed! ✓
```

## Summary

This chapter demonstrates industrial-strength MLIR dialect development with **two Python APIs**:

### Pythonic Tensor API (New!)
- ✅ **Operator overloading**: `a + b`, `a * b` work naturally
- ✅ **Graph building**: Operations build computation graph lazily
- ✅ **Clean syntax**: No MLIR text needed from users
- ✅ **PyTorch-like**: Familiar API for ML practitioners

### Low-level MLIR API (Educational)
- ✅ **Direct MLIR control**: Write MLIR text explicitly
- ✅ **Learning tool**: Understand MLIR syntax and semantics
- ✅ **Debugging**: See exact MLIR being compiled

### Underlying Technology
- ✅ **TableGen/ODS**: Declarative operation definitions (~50 lines → ~1000 lines generated C++)
- ✅ **Type Safety**: Compile-time verification and error checking
- ✅ **Pattern Rewriting**: Transform IR programmatically (vs string manipulation)
- ✅ **Production Patterns**: How real MLIR projects (Torch-MLIR, IREE) are built
- ✅ **Full Verification**: MLIR verifier catches errors at every transformation step

**Key Achievement**: Production-grade custom dialect with elegant Python interface!

---

## Next Steps

See **[TUTORIAL.md](TUTORIAL.md)** for:

- **Deep comparison with Chapter 8**: Operation-by-operation walkthrough showing Python vs C++/TableGen approach
- **TableGen basics**: How to define operations declaratively
- **Pattern rewriting**: C++ patterns for lowering custom ops
- **Step-by-step guide**: Phase-by-phase reading order for the code
- **Common patterns**: Reusable recipes for element-wise, unary, and matrix operations
- **Cheat sheet**: Quick reference for adding new operations
- **When to use which approach**: Decision guide for prototyping vs production