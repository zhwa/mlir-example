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

# Write MLIR using NN dialect
mlir_code = """
module {
  func.func @add(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>) {
    nn.add %arg0, %arg1, %arg2 : memref<4xf32>, memref<4xf32>, memref<4xf32>
    return
  }
}
"""

# Execute (automatic lowering and compilation)
a = np.array([1, 2, 3, 4], dtype=np.float32)
b = np.array([5, 6, 7, 8], dtype=np.float32)
result = ch9.execute(mlir_code, "add", [a, b], (4,))
print(result)  # [6. 8. 10. 12.]
```

## Compilation Pipeline

1. **Parse**: MLIR text with NN dialect → IR
2. **Lower NN → Standard**: `nn.add` → `linalg.generic` + `arith.addf`
3. **Lower to Loops**: `linalg` → `scf`
4. **Lower to LLVM**: Standard dialects → LLVM dialect
5. **JIT**: LLVM IR → native code → execute

## Testing

```bash
cd ch.9.TableGen-dialect
python3 test_jit.py
```

Expected output:
```
Chapter 9: Custom Dialect with TableGen
========================================

### Test 1: NN Add ###
✓ [1. 2. 3. 4.] + [5. 6. 7. 8.] = [ 6.  8. 10. 12.]

### Test 2: NN Mul ###
✓ [2. 3. 4. 5.] * [10. 10. 10. 10.] = [20. 30. 40. 50.]

### Test 3: NN MatMul ###
✓ MatMul: (2, 3) @ (3, 4) = (2, 4)
  Result: [1. 2. 3. 6.]

### Test 4: NN ReLU ###
✓ Input:  [-1.  2. -3.  4.]
  Output: [0. 2. 0. 4.]

All tests passed! ✓
```

## Next Steps

See **[TUTORIAL.md](TUTORIAL.md)** for:

- **Deep comparison with Chapter 8**: Operation-by-operation walkthrough showing Python vs C++/TableGen approach
- **TableGen basics**: How to define operations declaratively
- **Pattern rewriting**: C++ patterns for lowering custom ops
- **Step-by-step guide**: Phase-by-phase reading order for the code
- **Common patterns**: Reusable recipes for element-wise, unary, and matrix operations
- **Cheat sheet**: Quick reference for adding new operations
- **When to use which approach**: Decision guide for prototyping vs production