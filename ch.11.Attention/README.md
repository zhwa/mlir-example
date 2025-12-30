# Chapter 11: Attention Mechanism (Pure MLIR JIT)

**Goal**: Implement attention mechanism building blocks using pure MLIR JIT compilation with libffi

## Quick Start

```bash
# Build
cmake --build build/x64-release --target ch11

# Test
cd ch.11.Attention
PYTHONPATH=../build/x64-release/ch.11.Attention python3 test_jit.py
```

## What We're Building

Attention mechanism using pure MLIR JIT compilation:

```
Q, K, V = inputs
scores = Q @ K^T
scaled_scores = scores * (1/√d_k)
attn_weights = softmax(scaled_scores)
output = attn_weights @ V
```

## Architecture

**Tensor-First MLIR Compilation** using the Tensor API with Bufferization:

1. **Tensor API** - Build computation graphs with Python-like operations
2. **GraphNode** - Internal representation of operations and their dependencies
3. **IRBuilder** - Converts computation graph to MLIR tensor operations
4. **Bufferization Pipeline** - Three-pass transformation (tensor → memref)
   - `OneShotBufferize` - Converts tensor operations to memref operations
   - `BufferResultsToOutParams` - Transforms results to output parameters
   - `ConvertBufferizationToMemRef` - Removes remaining tensor artifacts
5. **JIT Compilation** - MLIR → LLVM IR → native code
6. **libffi Execution** - Proper calling convention for memref descriptors

**Custom Transformer Dialect** operations (all tensor-based):
- `transformer.matmul` - 2D matrix multiplication (tensor inputs/result)
- `transformer.add` - Element-wise addition (tensor inputs/result)
- `transformer.mul` - Element-wise multiplication (tensor inputs/result)
- `transformer.softmax` - Numerically stable softmax (tensor input/result)
- `transformer.transpose` - 2D transpose (tensor input/result)
- Higher-level: `attention(Q, K, V)` composed from primitives

## Key Concepts

### Pure MLIR JIT Pipeline (Tensor-First)

1. **Python API** → **Computation Graph**
   ```python
   Q, K, V = ch11.Tensor(...), ch11.Tensor(...), ch11.Tensor(...)
   result = ch11.attention(Q, K, V)  # Builds GraphNode tree
   ```

2. **Computation Graph** → **MLIR IR (Tensor-Based)**
   ```
   IRBuilder traverses graph, generates tensor-based MLIR operations:
   - Input nodes → func arguments (tensor<?x?xf32>)
   - Operation nodes → transformer.* operations (return tensors)
   Example: %result = transformer.matmul %lhs, %rhs : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
   ```

3. **Bufferization Pipeline** → **Memref-Based IR**
   ```
   Three-pass bufferization (func_ext registration required):
   - OneShotBufferize: tensor ops → memref ops (with function boundaries)
   - BufferResultsToOutParams: results → output parameters
   - ConvertBufferizationToMemRef: cleanup remaining tensor artifacts
   Result: %result = memref.alloc() + transformer ops update memrefs in-place
   ```

4. **MLIR IR** → **Native Code**
   ```
   PassManager pipeline:
   - Transformer dialect → Linalg (tensor-based)
   - Bufferization passes (tensor → memref)
   - Linalg (memref) → SCF loops
   - Standard dialects → LLVM IR
   - LLVM IR → native machine code
   ```

5. **Native Code** → **Execution**
   ```
   libffi calls JIT-compiled function with memref descriptors
   ```

### Lowering Strategy (Tensor-First → Linalg → Bufferization → Loops)

Operations lower through multiple stages:

**Stage 1: Transformer Dialect (Tensor) → Linalg (Tensor)**
- `MatmulOp` → `tensor::EmptyOp` + `linalg::FillOp` + `linalg::MatmulOp` (all tensors)
- `AddOp` → `tensor::EmptyOp` + `linalg::AddOp` (tensors)
- `MulOp` → `tensor::EmptyOp` + `linalg::MulOp` (tensors)
- `SoftmaxOp` → `tensor::EmptyOp` + `linalg::Reduce` + `linalg::Generic` (4-pass tensor algorithm)
- `TransposeOp` → `tensor::EmptyOp` + `linalg::TransposeOp` (tensors)

**Stage 2: Bufferization (Tensor → Memref)**
- `OneShotBufferize`: Converts all tensor operations to memref operations
- `BufferResultsToOutParams`: Adds output parameters to function signatures
- `ConvertBufferizationToMemRef`: Removes tensor.empty → memref.alloc

**Stage 3: Linalg (Memref) → SCF Loops**
- `linalg.matmul` → nested `scf.for` loops with `arith.mulf` and `arith.addf`
- `linalg.add`, `linalg.mul` → rank-generic loops with arithmetic ops
- `linalg.reduce` + `linalg.generic` → multi-pass loops (max → exp → normalize)
- `linalg.transpose` → dimension permutation via `memref.load`/`store`

This multi-stage approach:
- **Cleaner semantics** at high level (pure functions with tensor results)
- **Optimization opportunities** before bufferization (tensor-level passes)
- **Explicit memory management** after bufferization (memref allocations)
- **Consistency** with Chapters 5-10 architecture

### Debugging NaN Issues

**Common causes of NaN in MLIR JIT**:
1. **Uninitialized memory** - Forgot to store values after allocation
2. **Wrong loop bounds** - Using input dims when output dims needed
3. **Division by zero** - Softmax sum = 0 (needs numerical stability)
4. **Inf propagation** - Overflow in exp() without max subtraction

**Debugging strategy** (learned from Chapter 12):
1. **Test incrementally** - Test each operation independently first
2. **Add numerical validation** - Compare against NumPy reference with `rtol=1e-4`
3. **Check intermediate values** - Print outputs at each stage
4. **Simplify inputs** - Use small known inputs (e.g., identity matrices)
5. **Verify MLIR IR** - Dump IR to check operation patterns

**Bugs fixed**:
- **Scale initialization**: Must create and fill constant memref before MulOp
- **Transpose loops**: Must iterate over **output shape**, not input shape

### LLVM 19 Compatibility
Fixed several API changes:
- `createLinalgGeneralizationPass()` → `createLinalgGeneralizeNamedOpsPass()`
- `.cast<T>()` → `mlir::cast<T>()`
- `ExecutionEngine::defaultTransformer` removed → use `ExecutionEngineOptions`
- **Pass nesting**: Use `addNestedPass<func::FuncOp>()` for function-level passes

## API

**Tensor-based API** (pure MLIR JIT):

```python
import numpy as np
import ch11

# Create tensors from NumPy arrays
Q_data = np.random.randn(4, 8).astype(np.float32)
K_data = np.random.randn(4, 8).astype(np.float32)
V_data = np.random.randn(4, 8).astype(np.float32)

Q = ch11.Tensor(Q_data)
K = ch11.Tensor(K_data)
V = ch11.Tensor(V_data)

# Build computation graph
result_tensor = ch11.attention(Q, K, V)

# Execute with JIT compilation
output = ch11.forward(result_tensor)

# Individual operations
a = ch11.Tensor(np.random.randn(3, 4).astype(np.float32))
b = ch11.Tensor(np.random.randn(3, 4).astype(np.float32))

c = ch11.forward(a + b)                      # Element-wise addition
d = ch11.forward(ch11.matmul(a, b.T))        # Matrix multiplication
e = ch11.forward(ch11.transpose(a))          # Transpose
f = ch11.forward(ch11.softmax(a))            # Softmax
g = ch11.forward(ch11.scale(a, 0.5))         # Scale by constant
```