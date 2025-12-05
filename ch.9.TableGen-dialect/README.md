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

## Usage

```python
import ch9
import numpy as np

# Create tensors from numpy arrays
a = ch9.Tensor(np.array([1., 2., 3., 4.], dtype=np.float32))
b = ch9.Tensor(np.array([5., 6., 7., 8.], dtype=np.float32))

# Use operator overloading - builds computation graph
c = a + b

# Forward pass: compile and execute - OpBuilder creates IR directly
result = ch9.forward(c)
print(result)  # [6. 8. 10. 12.]

# Chain operations
d = (a + b) * a
result = ch9.forward(d)

# Matrix operations
A = ch9.Tensor(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
B = ch9.Tensor(np.array([[5., 6.], [7., 8.]], dtype=np.float32))
C = ch9.matmul(A, B)
result = ch9.forward(C)
```

## Key Features

- **TableGen/ODS**: Declarative operation definitions (~50 lines → ~1000 lines generated C++)
- **OpBuilder**: Industrial-grade IR construction (same as Torch-MLIR, JAX, IREE)
- **Pattern Rewriting**: Type-safe IR transformations in C++
- **Pythonic API**: `ch9.forward()` follows PyTorch conventions
- **Graph Building**: Operations build computation graph lazily

## Learn More

See **[TUTORIAL.md](TUTORIAL.md)** for detailed comparison with Chapter 8, TableGen basics, pattern rewriting, and step-by-step guide.