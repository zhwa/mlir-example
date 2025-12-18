# Chapter 12: Transformer Block

**Goal**: Build a complete transformer block with clean Tensor API for end-to-end inference

## What We're Building

A full transformer block with proper abstraction:

```
TransformerBlock(x):
  # Pre-norm architecture
  x = LayerNorm(x + MultiHeadAttention(x))
  x = LayerNorm(x + FeedForward(x))
  return x
```

**Components:**
1. **Layer Normalization**: Normalize across embedding dimension
2. **Multi-Head Attention**: Scaled dot-product with Q/K/V projections
3. **Feed-Forward Network**: Linear → GELU → Linear
4. **Residual Connections**: Skip connections around each sub-layer

## Running Tests

```bash
cd ch.12.Transformer
python3 test_jit.py
```

### Design Decisions

1. **Pure MLIR JIT Compilation**: Follows Chapter 9's pattern
   - Tensor API → Computation Graph → MLIR IR → Native Code
   - IRBuilder translates graph nodes to MLIR operations
   - ExecutionEngine compiles MLIR to LLVM IR to machine code
   - libffi handles calling convention for memref descriptors

2. **Minimalist Approach**: Focused on core transformer functionality
   - No causal masking (Chapter 13)
   - No KV cache (Chapter 13)
   - No RoPE (Chapter 13)
   - Clean foundation for advanced features

3. **Architecture** (606 lines total):
   - TransformerCompiler: MLIR context, lowering pipeline, JIT compilation
   - IRBuilder: Graph → MLIR IR translation
   - Tensor API: High-level Python interface with operator overloading
   - forward(): JIT entry point with topological graph traversal


## Technical Implementation

### JIT Execution Flow
1. **Graph Construction**: Tensor operations build computation graph (OpType::LayerNorm, etc.)
2. **Topological Traversal**: Collect inputs and parameters (gamma, beta, weight, bias)
3. **IR Generation**: IRBuilder translates graph nodes to MLIR operations
4. **Function Signature**: `func.func @compute(inputs..., parameters..., output) -> ()`
5. **JIT Compilation**: ExecutionEngine compiles MLIR → LLVM IR → Native code
6. **libffi Call**: Proper memref descriptor marshalling (7 args per 2D memref)

### Key Fixes
- **Segfault Resolution**: Used libffi instead of `void(*)(void**)` casting
- **Parameter Passing**: Gamma/beta/weight/bias as function arguments (not allocations in IR)
- **Pass Ordering**: MathToLLVM before MathToLibm for proper math.rsqrt lowering

## Debugging NaN Issues

When implementing Pure MLIR JIT, we encountered NaN (Not a Number) results in complex operations. Here's how to debug and fix such issues:

### What Causes NaN?

1. **Uninitialized Memory**: Allocating buffers without initializing them
   - **Example**: `Value scale = createAlloc({1})` without storing a value
   - **Effect**: Reading garbage memory leads to NaN in arithmetic operations

2. **Wrong Loop Bounds**: Iterating with incorrect dimensions
   - **Example**: Transpose iterating over input dims instead of output dims
   - **Effect**: Reading/writing out of bounds or leaving memory uninitialized

3. **Missing Operations**: Forgetting to implement parts of the computation
   - **Example**: Creating a scale buffer but not storing the scale_factor value
   - **Effect**: Operations use default (garbage) values, propagating NaN

### Debugging Strategy

**Step 1: Isolate the Problem**
```python
# Test operations individually, from simple to complex
test_add()       # ✓ Works
test_layernorm() # ✓ Works
test_linear()    # ✓ Works
test_attention() # ✗ NaN! Problem is in attention or its dependencies
```

**Step 2: Test Dependencies**
```python
# Attention uses: linear, matmul, transpose, scale, softmax
test_matmul()    # ✓ Works
test_transpose() # ✗ Wrong values! Found bug #1
test_softmax()   # ✓ Works
test_scale()     # ✗ NaN! Found bug #2
```

**Step 3: Add Numerical Validation**
```python
# Don't just check shapes - validate values!
result = ch12.forward(output)
expected = compute_reference_numpy(inputs)
np.testing.assert_allclose(result, expected, rtol=1e-4)
```

### Bugs Found & Fixed

#### Bug #1: Transpose Loop Bounds
**Problem**: Loop used input dimensions instead of output dimensions
```cpp
// WRONG: Input is (dim0, dim1), but output is (dim1, dim0)
for i in 0..dim0:  // Should be 0..dim1
  for j in 0..dim1:  // Should be 0..dim0
    output[i,j] = input[j,i]
```
**Effect**: Only (dim0, dim1) elements written, leaving rest uninitialized
- Input: (2, 3) → Output: (3, 2) but only 2×3=6 elements written, last row had garbage

**Fix**: Use output dimensions for loop bounds
```cpp
// CORRECT: Iterate over output dimensions
for i in 0..inputDim1:  // Output's first dimension
  for j in 0..inputDim0:  // Output's second dimension
    output[i,j] = input[j,i]
```

#### Bug #2: Scale Factor Not Initialized
**Problem**: Allocated memref for scale but never stored the value
```cpp
// WRONG: Buffer allocated but never initialized
Value scale = createAlloc({1});
builder.create<ScaleOp>(input, scale, output);  // scale[0] is garbage!
```
**Effect**: ScaleOp reads uninitialized memory, multiplies by garbage → NaN

**Fix**: Store the scale_factor constant into the buffer
```cpp
// CORRECT: Initialize the scale buffer
Value scale = createAlloc({1});
Value zeroIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
Value scaleConst = builder.create<arith::ConstantFloatOp>(
    loc, llvm::APFloat(node->scale_factor), builder.getF32Type());
builder.create<memref::StoreOp>(loc, scaleConst, scale, ValueRange{zeroIdx});
builder.create<ScaleOp>(input, scale, output);  // Now scale[0] has correct value
```

### Lessons Learned

1. **Test incrementally**: Don't wait until the full transformer is built
2. **Validate numerically**: Shape checks aren't enough - compare actual values
3. **Check loop bounds**: Especially for transpose/reshape operations
4. **Initialize all memory**: Every allocated buffer must be initialized before reading
5. **Use reference implementations**: NumPy helps validate correctness

### Test Coverage

The test suite ([test_jit.py](test_jit.py)) validates all 11 operations with numerical checks:
- 8 primitive operations: Add, LayerNorm, Linear, GELU, Matmul, Transpose, Softmax, Scale
- 3 compositions: FFN, Multi-Head Attention, Transformer Block

Each test compares MLIR JIT output against NumPy reference implementation with `rtol=1e-4`.