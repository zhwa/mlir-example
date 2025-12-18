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