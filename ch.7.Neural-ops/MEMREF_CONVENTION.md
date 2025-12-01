## MLIR Memref Calling Convention

### The Challenge

When MLIR compiles a function like:
```mlir
func.func @add(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>)
```

You might expect a simple C signature like `void add(float* a, float* b, float* out)`, but MLIR actually generates 15 parameters (5 per memref)!

### Why Multiple Parameters Per Memref?

Each memref is a **descriptor** with metadata:

**For 1D memref** `memref<Nxf32>` (5 parameters):
1. allocated_ptr, 2. aligned_ptr, 3. offset, 4. size, 5. stride

**For 2D memref** `memref<MxNxf32>` (7 parameters):

1. allocated_ptr - base pointer to allocated memory
2. aligned_ptr - aligned pointer for actual data access
3. offset - offset from base (usually 0)
4. size[0] - first dimension size (M)
5. size[1] - second dimension size (N)
6. stride[0] - row stride (usually N)
7. stride[1] - column stride (usually 1)

This enables dynamic shapes, strided access, memory alignment, and type safety.

### Handling in Bindings

We provide a **generic C++ helper** in `bindings.cpp`:
- `execute_generic(fn, inputs, output_shape)` - handles arbitrary number of inputs/outputs with 1D or 2D shapes

This single function replaces all shape-specific helpers through runtime shape introspection.

For reference, the old approach required separate helpers:
- ❌ `execute_binary_1d(fn, lhs, rhs)` - 2 1D inputs → 1 1D output
- ❌ `execute_matmul(fn, lhs, rhs)` - 2 2D inputs → 1 2D output  
- ❌ `execute_3inputs_2d(fn, in1, in2, in3)` - 3 2D inputs → 1 2D output

## Concrete Example: Three Approaches

Let's execute a compiled multi-layer network function with signature:
```mlir
func.func @mlp(%x: memref<2x3xf32>, %W1: memref<3x4xf32>, %W2: memref<4x2xf32>, %out: memref<2x2xf32>)
```

This expands to **28 parameters** at the LLVM level (7 params × 4 memrefs).

### Approach 1: Generic C++ Helper (Recommended)

```python
import ch7_neural_ops as ch7
import numpy as np

# Compile function
g = ch7.Graph()
x = g.variable([2, 3])
W1 = g.variable([3, 4])
W2 = g.variable([4, 2])
h = g.matmul(x, W1)
h_relu = g.relu(h)
y = g.matmul(h_relu, W2)
fn = g.compile(y, "mlp")

# Execute with generic API - automatically handles all 28 parameters
x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
W1_data = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]], dtype=np.float32)
W2_data = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], dtype=np.float32)

result = ch7.execute_generic(fn, [x_data, W1_data, W2_data], (2, 2))
# Done! Runtime shape introspection handles the complexity.
```

**Advantages**: Clean API, automatic shape handling, type-safe, extensible.

### Approach 2: Python with ctypes (Educational/Manual)

```python
import ctypes
import numpy as np

def execute_3inputs_2d_ctypes(fn_ptr, input1, input2, input3):
    """Execute function with 3 2D inputs using ctypes"""
    out = np.zeros((input1.shape[0], input3.shape[1]), dtype=np.float32)

    # Define the function signature with ALL 28 parameters
    fn_type = ctypes.CFUNCTYPE(
        None,  # Return type (void)
        # input1: memref<2x3xf32> → 7 parameters
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        # input2: memref<3x4xf32> → 7 parameters
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        # input3: memref<4x2xf32> → 7 parameters
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        # output: memref<2x2xf32> → 7 parameters
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
    )

    # Cast function pointer
    fn = fn_type(fn_ptr)

    # Call with ALL 28 arguments
    fn(
        # input1 descriptor (7 params)
        input1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        input1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        0, input1.shape[0], input1.shape[1], input1.shape[1], 1,
        # input2 descriptor (7 params)
        input2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        input2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        0, input2.shape[0], input2.shape[1], input2.shape[1], 1,
        # input3 descriptor (7 params)
        input3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        input3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        0, input3.shape[0], input3.shape[1], input3.shape[1], 1,
        # output descriptor (7 params)
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        0, out.shape[0], out.shape[1], out.shape[1], 1
    )
    return out

# Usage
fn_ptr = ctypes.cast(fn, ctypes.c_void_p).value
result = execute_3inputs_2d_ctypes(fn_ptr, x_data, W1_data, W2_data)
```

**Lines of code**: ~40-50 lines for this function alone

**Pros**: 
- No C++ compilation needed
- Flexible - can handle any signature
- Good for quick prototyping

**Cons**:
- Extremely verbose and error-prone
- Have to count all 28 parameters manually
- Easy to get parameter order wrong
- No type checking at definition time
- Hard to maintain

### Approach 3: Old C++ Shape-Specific Helper (Obsolete)

This approach is **no longer used** but shown for historical reference.

**Old Pattern** (`execute_3inputs_2d` - removed):
```cpp
// OLD: Separate function for 3 2D inputs
py::array_t<float> execute_3inputs_2d(uintptr_t fnPtr, 
                                      py::array_t<float> input1,
                                      py::array_t<float> input2, 
                                      py::array_t<float> input3) {
    // ... manually handle 28 parameters ...
}

    return output;
}

// Expose to Python
PYBIND11_MODULE(ch7_neural_ops, m) {
    // ...
    m.def("execute_3inputs_2d", &execute_3inputs_2d, 
          "Execute function with 3 2D inputs");
}
```

**Python side** (usage):
```python
import ch7_neural_ops as ch7

# Clean, simple call - all complexity hidden!
result = ch7.execute_3inputs_2d(fn, x_data, W1_data, W2_data)
**Problem with shape-specific approach**:
- Need separate C++ function for every pattern:
  - `execute_binary_1d` - 2 1D inputs
  - `execute_matmul` - 2 2D inputs
  - `execute_3inputs_2d` - 3 2D inputs
  - `execute_4inputs_mixed` - 4 mixed inputs... endless combinations!

**Solution**: Generic binding with runtime introspection (see Approach 1).

### Comparison Summary

| Aspect | Generic Helper (Now) | ctypes Manual | Shape-Specific (Old) |
|--------|---------------------|---------------|---------------------|
| **Python Code** | 1 line per call | 40-50 lines | 1 line per call |
| **C++ Code** | ~110 lines (handles all) | 0 lines | ~30 lines per pattern |
| **Type Safety** | Compile-time | Runtime only | Compile-time |
| **Error Messages** | Clear exceptions | Cryptic segfaults | Clear exceptions |
| **Maintainability** | High (one function) | Low (repetitive) | Medium (many functions) |
| **Flexibility** | High (any shape combo) | High (any signature) | Low (one pattern each) |
| **Best For** | Production & learning | Quick experiments | Obsolete |

### Recommendation

✅ **Use `execute_generic()`**: Handles all patterns automatically through runtime introspection.

🔧 **Use ctypes**: Only for learning or debugging unusual calling conventions.

❌ **Don't write shape-specific helpers**: Generic approach supersedes them.

### Key Achievement

The generic binding layer (Approach 1) combines the best of both worlds:
- **Clean API** like C++ helpers (1 line of Python)
- **Flexibility** like ctypes (works with any shape combination)
- **Maintainability** - single implementation handles everything
- **Extensibility** - easy to add support for new cases

No more combinatorial explosion of binding functions! 🎉