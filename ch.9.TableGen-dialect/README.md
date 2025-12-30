# Chapter 9: Custom Dialect with TableGen

Production-grade custom MLIR dialect using TableGen (Operation Definition Specification) - the standard approach for industrial MLIR compiler development.

📖 **For detailed comparison with Chapter 8 and in-depth tutorial, see [TUTORIAL.md](TUTORIAL.md)**

## Quick Overview

This chapter demonstrates how real-world MLIR projects define custom dialects using TableGen, automatic C++ code generation, and type-safe IR transformation patterns.

**Key Difference from Chapter 8**:
- Chapter 8: Python string-based prototyping (fast iteration, learning)
- Chapter 9: C++ TableGen production (type safety, industrial scale)

## NN Dialect Operations

The `nn` (Neural Network) dialect provides high-level tensor-based operations:

| Operation | Syntax | Description |
|-----------|--------|-------------|
| `nn.add` | `%result = nn.add %a, %b : tensor<...>` | Element-wise addition |
| `nn.mul` | `%result = nn.mul %a, %b : tensor<...>` | Element-wise multiplication |
| `nn.matmul` | `%result = nn.matmul %a, %b : tensor<...>` | Matrix multiplication |
| `nn.relu` | `%result = nn.relu %x : tensor<...>` | ReLU activation |

**Note**: All operations use **functional tensor-based style** (returning values) for consistency with Chapters 5-8. The tensor→memref conversion is handled automatically by the OneShotBufferize pass.

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
- **Tensor-First Design**: Modern MLIR approach with automatic bufferization
- **Pythonic API**: `ch9.forward()` follows PyTorch conventions
- **Graph Building**: Operations build computation graph lazily

## Compilation Pipeline

```
NN Dialect (tensors)
   ↓ createConvertNNToStandardPass()
Linalg + Tensor (tensors)
   ↓ OneShotBufferizePass()
Linalg + MemRef (memrefs)
   ↓ ConvertLinalgToLoopsPass()
SCF + MemRef
   ↓ ConvertToLLVM
LLVM IR
```

The tensor→memref conversion is handled automatically using OneShotBufferize with proper interface registrations.

## Learn More

See **[TUTORIAL.md](TUTORIAL.md)** for detailed comparison with Chapter 8, TableGen basics, pattern rewriting, and step-by-step guide.