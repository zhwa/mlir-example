# Chapter 7: Operator Composition with Computation Graphs

This chapter introduces **operator composition** using computation graphs. It demonstrates how to build complex ML operations by composing simpler ones using a deferred execution model, generating MLIR with tensors and linalg operations.

## Learning Objectives

- Build a computation graph that tracks operations symbolically
- Implement deferred execution (build graph first, compile later)
- Compose multiple operations (add, mul, matmul, relu, softmax)
- Generate MLIR with tensors and linalg operations from high-level graphs
- Understand the bridge between imperative Python code and declarative MLIR

## Key Concepts

### Computation Graph

A computation graph is a directed acyclic graph (DAG) where:
- **Nodes** represent operations (add, matmul, relu, etc.)
- **Edges** represent data dependencies between operations
- **Execution is deferred** until `compile()` is called

```python
g = ch7.Graph()
x = g.variable([4])       # Create variable/placeholder
y = g.variable([4])       # Create another variable
z = g.add(x, y)           # Add operation (not executed yet!)
fn = g.compile(z, "add")  # NOW we generate MLIR and compile
result = ch7.execute_binary_1d(fn, a_data, b_data)
```

### Why Computation Graphs?

1. **Optimization**: Build entire computation before execution, enabling whole-program optimization
2. **Portability**: Same graph can target CPU, GPU, or specialized accelerators
3. **Debugging**: Inspect the full computation before running
4. **Composition**: Complex operations built from simple primitives

### Deferred Execution

Unlike eager execution (NumPy, PyTorch with default settings), we:
1. **Build** the graph symbolically (returns operation IDs, not values)
2. **Compile** the entire graph to MLIR → LLVM → machine code
3. **Execute** the compiled function with actual data

This is similar to:
- TensorFlow 1.x static graphs
- JAX's `jit` compilation
- PyTorch's `torch.jit.script`

## Implemented Operations

### Element-wise Operations

- **Add**: `z = g.add(x, y)` - Element-wise addition with `linalg.generic`
- **Mul**: `z = g.mul(x, y)` - Element-wise multiplication with `linalg.generic`

Generated MLIR:
```mlir
%result = linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>, ...],
  iterator_types = ["parallel"]
} ins(%lhs, %rhs) outs(%empty) {
^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
  %sum = arith.addf %arg0, %arg1 : f32
  linalg.yield %sum : f32
}
```

### Matrix Operations

- **MatMul**: `z = g.matmul(A, B)` - Matrix multiplication with `linalg.matmul`

Generated MLIR:
```mlir
%init = linalg.fill ins(%zero : f32) outs(%empty)
%result = linalg.matmul ins(%A, %B) outs(%init) -> tensor<2x4xf32>
```

### Activation Functions

- **ReLU**: `y = g.relu(x)` - Rectified Linear Unit using `linalg.generic`
- **Softmax**: `y = g.softmax(x)` - Softmax normalization using `linalg.reduce` + `linalg.generic`

ReLU example:
```mlir
%relu = linalg.generic {
  iterator_types = ["parallel"]
} ins(%input) outs(%empty) {
^bb0(%arg0: f32, %arg1: f32):
  %max = arith.maximumf %arg0, %zero : f32
  linalg.yield %max : f32
}
```

## Implementation Details

### Architecture

```
Python API (bindings.cpp)
    ↓
ComputationGraph (ir.cpp)
    ├─ Track operations symbolically
    ├─ Store operation types and dependencies
    └─ Generate MLIR with tensors and linalg ops
        ↓
Bufferization (lowering.cpp)
    ├─ Tensor → MemRef
    ├─ Linalg → Loops
    └─ Math → Libm/LLVM
        ↓
JIT Compilation (jit.cpp)
    ├─ MLIR → LLVM IR
    └─ mlir::ExecutionEngine → Machine Code
```

### IR Generation Strategy

The `ComputationGraph::generateMLIR()` method:

1. **Collect variables**: Find all `Variable` operations
2. **Build function signature**: Create func.func with tensor arguments and return type
3. **Recursive generation**: Build operations depth-first, memoizing results
4. **Return tensor result**: Function returns computed tensor

### Operation Building with Linalg

#### Element-wise Operations
```cpp
// Build linalg.generic for element-wise ops
auto genericOp = builder.create<linalg::GenericOp>(
    loc, lhsType, ValueRange{lhs, rhs}, ValueRange{empty},
    indexingMaps, iteratorTypes,
    [&](OpBuilder& b, Location l, ValueRange args) {
        Value result = b.create<arith::AddFOp>(l, args[0], args[1]);
        b.create<linalg::YieldOp>(l, result);
    }
);
```

#### Matrix Multiplication
```cpp
// Use linalg.matmul directly
Value init = builder.create<linalg::FillOp>(loc, zero, empty);
auto matmulOp = builder.create<linalg::MatmulOp>(
    loc, resultType, ValueRange{lhs, rhs}, ValueRange{init}
);
```

## Pipeline

```
High-Level Graph                                        (Python/C++)
  ├─ Operations stored as DAG nodes
  ├─ Symbolic execution (no values yet)
  └─ Dependencies tracked
          ↓ [generateMLIR]
Tensor IR                                               (Functional)
  ├─ func.func with tensor arguments/results
  ├─ linalg.generic (element-wise)
  ├─ linalg.matmul (matrix multiply)
  ├─ linalg.reduce (reductions)
  └─ math.exp (mathematical functions)
          ↓ [Canonicalize + Bufferize]
MemRef IR                                               (Imperative)
  ├─ func.func with memref out-parameters
  ├─ scf.for loops
  └─ memref.load/store
          ↓ [Lower to LLVM]
LLVM Dialect
  └─ llvm.* instructions
          ↓ [JIT Compile]
Machine Code                                            (Executable)
```

## MLIR Calling Convention

### Static 1D Memrefs

For `memref<Nxf32>`, MLIR uses 5 parameters:
```cpp
void function(float* allocated, float* aligned, 
              int64_t offset, int64_t size, int64_t stride)
```

### Static 2D Memrefs

For `memref<MxNxf32>`, MLIR uses 7 parameters:
```cpp
void function(float* allocated, float* aligned, int64_t offset,
              int64_t size0, int64_t size1, 
              int64_t stride1, int64_t stride0)
```

**Note**: Strides are in reverse order! `stride1` comes before `stride0`.

## Testing

Run all tests:
```bash
cd ch.7.Neural-ops
PYTHONPATH=../build/x64-release/ch.7.Neural-ops python3 test_jit.py
```

### Test Coverage

- ✅ Element-wise addition
- ✅ Element-wise multiplication
- ✅ Matrix multiplication (2D)
- ✅ ReLU activation
- ✅ Softmax activation (math.exp lowered to libm)
- ✅ Multi-layer network (2-layer MLP structure)

**All 6 tests pass!**

## Example: Two-Layer Neural Network

```python
g = ch7.Graph()
x = g.variable([2, 3])    # Input: batch_size=2, features=3
W1 = g.variable([3, 4])   # Layer 1 weights
W2 = g.variable([4, 2])   # Layer 2 weights

# Layer 1: x @ W1
h = g.matmul(x, W1)    # → [2, 4]

# Activation
h_relu = g.relu(h)

# Layer 2: h @ W2
y = g.matmul(h_relu, W2)  # → [2, 2]

# Compile the entire computation
mlir_code = g.get_mlir(y, "mlp")
print(mlir_code)  # See the full MLIR before compilation!
```

## Known Limitations

1. **Shape inference**: Currently requires explicit shapes for all operations
2. **Dynamic shapes**: All shapes must be known at graph construction time
3. **Memory management**: Intermediate buffers are allocated with memref.alloc (no pooling yet)
4. **2D operations**: Softmax currently only works with 1D inputs (2D requires flattening or per-row operation)

## Troubleshooting

### "failed to legalize operation 'math.exp'"

**Problem**: Math operations like `math.exp` need to be lowered in the correct order.

**Solution**: The pass pipeline must apply math lowering in this specific order:
```cpp
pm.addPass(createConvertMathToLLVMPass());   // FIRST: converts to LLVM intrinsics
pm.addPass(createConvertMathToLibmPass());   // SECOND: remaining ops to libm calls
```

This two-tier strategy ensures all math operations get properly lowered.

### Incorrect Results

**Symptom**: Output shows garbage or zeros

**Likely Cause**: MLIR calling convention mismatch

**Fix**: Each memref expands to multiple parameters (5 for 1D, 7 for 2D), not a simple pointer. See `MEMREF_CONVENTION.md` for detailed explanation and examples of handling this with C++ helpers vs ctypes.

**Quick reference**:
- **Use**: `execute_generic(fn, [inputs...], output_shape)` - single function handles all patterns
- Runtime shape introspection automatically marshals memref parameters
- For deep dive into memref calling convention, see `MEMREF_CONVENTION.md`

**Key improvement**: Generic binding layer eliminates all shape-specific helper functions. One clean API for everything!

## See Also

- **`MEMREF_CONVENTION.md`**: Deep dive into MLIR's memref calling convention with concrete ctypes vs C++ examples
- **`test_jit.py`**: Working examples of all operations and execution patterns
- **Chapter 6 (`ch.6.Softmax`)**: Reference for softmax implementation details