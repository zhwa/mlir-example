# Chapter 11: Attention Mechanism

**Goal**: Implement multi-head scaled dot-product attention with custom Transformer dialect

## Quick Start

```bash
# Build
cmake --build build/x64-release --target ch11

# Test
cd ch.11.Attention
PYTHONPATH=../build/x64-release/ch.11.Attention python3 test_jit.py
```

## What We're Building

Multi-head scaled dot-product attention with Q/K/V/O projections:

```
Q = input @ W_q^T
K = input @ W_k^T
V = input @ W_v^T

For each head h:
  scores_h = (Q_h @ K_h^T) / √head_dim
  attn_h = softmax(scores_h)
  context_h = attn_h @ V_h

output = concat(context_1, ..., context_h) @ W_o^T
```

## Architecture

**Custom Transformer Dialect** with TableGen-defined operations:
- `transformer.matmul` - 2D/3D matrix multiplication
- `transformer.add` - Element-wise addition
- `transformer.mul` - Element-wise multiplication (for scaling)
- `transformer.softmax` - Numerically stable softmax
- `transformer.transpose` - Transpose last two dimensions
- `transformer.attention` - Multi-head attention (placeholder for future MLIR implementation)

**Current Implementation**: C++ reference implementation for validation

## Key Concepts

### TableGen Dialect Definition
- **No attributes on operations** - LLVM 19 automatically adds BytecodeOpInterface for ops with attributes, which doesn't exist in LLVM 19 API
- **Minimal includes** - Only `mlir/IR/OpBase.td` to avoid unwanted interface generation
- **Memref-based operations** - Direct memory operations for efficiency

### LLVM 19 Compatibility
Fixed several API changes:
- `createLinalgGeneralizationPass()` → `createLinalgGeneralizeNamedOpsPass()`
- `.cast<T>()` → `mlir::cast<T>()`
- `ExecutionEngine::defaultTransformer` removed → use `ExecutionEngineOptions`

### Lowering Strategy
Operations lower to standard MLIR dialects:
- `MatmulOp` → nested `scf.for` loops with `memref.load`/`store`
- `AddOp`, `MulOp` → dynamic loops based on rank
- `SoftmaxOp` → three-pass algorithm (find max → exp → normalize)
- `TransposeOp` → dimension permutation via load/store
- `AttentionOp` → currently returns failure (uses C++ fallback)

## File Structure

```
ch.11.Attention/
├── README.md                      # This file
├── CMakeLists.txt                 # Build with TableGen integration
├── test_jit.py                    # Test suite (NumPy-based)
├── inc/
│   ├── TransformerDialect.td      # Dialect definition
│   ├── TransformerOps.td          # Operation definitions
│   ├── TransformerDialect.h       # Generated dialect header
│   ├── TransformerOps.h           # Generated ops header
│   └── TransformerToStandard.h    # Lowering pass interface
└── src/
    ├── TransformerDialect.cpp     # Dialect implementation
    ├── TransformerOps.cpp         # Operation implementations
    ├── TransformerToStandard.cpp  # Lowering patterns
    └── bindings.cpp               # Python bindings + C++ reference

Generated (build/):
├── TransformerOps.h.inc           # TableGen generated
├── TransformerOps.cpp.inc
└── TransformerDialect.h.inc
```

## API

```python
import numpy as np
import ch11

# Input: [seq_len, d_model]
input = np.random.randn(4, 8).astype(np.float32)

# Projection weights: [d_model, d_model]
w_q = np.random.randn(8, 8).astype(np.float32) * 0.1
w_k = np.random.randn(8, 8).astype(np.float32) * 0.1
w_v = np.random.randn(8, 8).astype(np.float32) * 0.1
w_o = np.random.randn(8, 8).astype(np.float32) * 0.1

# Multi-head attention (Pythonic API - returns output)
num_heads = 2
head_dim = 4
output = ch11.attention(input, w_q.T, w_k.T, w_v.T, w_o.T, num_heads, head_dim)
```

**Note**: Weights must be transposed (`.T`) because C++ implementation does `input @ W` while convention is `input @ W^T`.

## Progress

- [x] Directory structure
- [x] Custom Transformer dialect with TableGen
- [x] 6 operations defined (matmul, add, mul, softmax, transpose, attention)
- [x] Lowering pass for 5 operations (attention placeholder)
- [x] C++ reference implementation
- [x] Python bindings
- [x] Test suite with 3 passing tests
  - Single-head self-attention
  - Multi-head attention (2 heads)
  - Attention with Q/K/V/O projections
- [x] LLVM 19 compatibility fixes
- [ ] Future: Implement AttentionOp lowering using other dialect ops