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

We provide C++ helpers in `bindings.cpp`:
- `execute_binary_1d(fn, lhs, rhs)` - 2 1D inputs → 1 1D output
- `execute_matmul(fn, lhs, rhs)` - 2 2D inputs → 1 2D output  
- `execute_3inputs_2d(fn, in1, in2, in3)` - 3 2D inputs → 1 2D output

For custom patterns, add new helpers or use ctypes.

## Concrete Example: Two Approaches

Let's execute a compiled multi-layer network function with signature:
```mlir
func.func @mlp(%x: memref<2x3xf32>, %W1: memref<3x4xf32>, %W2: memref<4x2xf32>, %out: memref<2x2xf32>)
```

This expands to **28 parameters** at the LLVM level (7 params × 4 memrefs).

### Approach 1: Python with ctypes (Verbose but Flexible)

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

### Approach 2: C++ Helper in bindings.cpp (Clean and Reusable)

**C++ side** (`bindings.cpp`):
```cpp
py::array_t<float> execute_3inputs_2d(uintptr_t fnPtr, 
                                      py::array_t<float> input1,
                                      py::array_t<float> input2, 
                                      py::array_t<float> input3) {
    auto in1Buf = input1.request();
    auto in2Buf = input2.request();
    auto in3Buf = input3.request();

    if (in1Buf.ndim != 2 || in2Buf.ndim != 2 || in3Buf.ndim != 2) {
        throw std::runtime_error("All inputs must be 2D");
    }

    // Allocate output
    py::array_t<float> output({in1Buf.shape[0], in3Buf.shape[1]});
    auto outputBuf = output.request();

    // Type-safe function pointer
    using FnType = void(*)(        
        float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // input1
        float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // input2
        float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // input3
        float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t   // output
    );
    auto fn = reinterpret_cast<FnType>(fnPtr);

    // Call with proper memref descriptors
    fn(static_cast<float*>(in1Buf.ptr),
       static_cast<float*>(in1Buf.ptr),
       0, in1Buf.shape[0], in1Buf.shape[1], in1Buf.shape[1], 1,
       static_cast<float*>(in2Buf.ptr),
       static_cast<float*>(in2Buf.ptr),
       0, in2Buf.shape[0], in2Buf.shape[1], in2Buf.shape[1], 1,
       static_cast<float*>(in3Buf.ptr),
       static_cast<float*>(in3Buf.ptr),
       0, in3Buf.shape[0], in3Buf.shape[1], in3Buf.shape[1], 1,
       static_cast<float*>(outputBuf.ptr),
       static_cast<float*>(outputBuf.ptr),
       0, outputBuf.shape[0], outputBuf.shape[1], outputBuf.shape[1], 1);

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
```

**Lines of code**: 1 line in Python, ~30 lines of C++ (reusable)

**Pros**:
- Clean Python API (1 line!)
- Type-safe C++ implementation
- Compile-time error checking
- Reusable across all tests
- Proper error messages
- Self-documenting code

**Cons**:
- Requires C++ compilation
- Need to write a new C++ function for every input/output combination (2 inputs, 3 inputs, 4 inputs, mixed 1D/2D, etc.). The ctypes approach is more flexible for prototyping, even if verbose
- Less flexible than ctypes

### Comparison Summary

| Aspect | ctypes Approach | C++ Helper Approach |
|--------|----------------|---------------------|
| **Python Code** | 40-50 lines per call site | 1 line per call site |
| **C++ Code** | 0 lines | ~30 lines (one-time) |
| **Type Safety** | Runtime only | Compile-time |
| **Error Messages** | Cryptic segfaults | Clear exceptions |
| **Maintainability** | Low (repetitive) | High (reusable) |
| **Flexibility** | High (any signature) | Medium (per-pattern helpers) |
| **Best For** | Quick experiments | Production code |

### Recommendation

- **Use ctypes**: For one-off experiments or when testing unusual signatures
- **Use C++ helpers**: For any operation you'll use more than once
- **Future direction**: Build a generic executor that handles arbitrary signatures automatically

### Future Work

Production systems need **generic executors** that introspect function signatures and handle arbitrary input/output counts automatically. This represents the direction for future chapters on building production-ready ML systems.