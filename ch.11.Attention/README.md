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

**Pure MLIR JIT Compilation** using the Tensor API:

1. **Tensor API** - Build computation graphs with Python-like operations
2. **GraphNode** - Internal representation of operations and their dependencies
3. **IRBuilder** - Converts computation graph to MLIR operations
4. **JIT Compilation** - MLIR → LLVM IR → native code
5. **libffi Execution** - Proper calling convention for variable memref descriptors

**Custom Transformer Dialect** operations:
- `transformer.matmul` - 2D matrix multiplication
- `transformer.add` - Element-wise addition  
- `transformer.mul` - Element-wise multiplication (for scaling)
- `transformer.softmax` - Numerically stable softmax
- `transformer.transpose` - 2D transpose
- Higher-level: `attention(Q, K, V)` composed from primitives

## Key Concepts

### Pure MLIR JIT Pipeline

1. **Python API** → **Computation Graph**
   ```python
   Q, K, V = ch11.Tensor(...), ch11.Tensor(...), ch11.Tensor(...)
   result = ch11.attention(Q, K, V)  # Builds GraphNode tree
   ```

2. **Computation Graph** → **MLIR IR**
   ```
   IRBuilder traverses graph, generates MLIR operations:
   - Input nodes → func arguments
   - Operation nodes → transformer.* operations
   ```

3. **MLIR IR** → **Native Code**
   ```
   PassManager pipeline:
   - Transformer dialect → Standard/Arith/MemRef/SCF
   - Standard dialects → LLVM IR
   - LLVM IR → native machine code
   ```

4. **Native Code** → **Execution**
   ```
   libffi calls JIT-compiled function with memref descriptors
   ```

### Lowering Strategy
Operations lower to standard MLIR dialects:
- `MatmulOp` → nested `scf.for` loops with `arith.mulf` and `arith.addf`
- `AddOp`, `MulOp` → rank-generic loops, `arith.addf`/`arith.mulf`
- `SoftmaxOp` → three-pass algorithm (find max → exp → normalize)
- `TransposeOp` → dimension permutation via `memref.load`/`store`
- `attention()` → composition of primitives (matmul, transpose, scale, softmax)

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