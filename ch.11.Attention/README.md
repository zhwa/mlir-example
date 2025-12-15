# Chapter 11: Attention Mechanism

**Goal**: Implement scaled dot-product attention - the core of transformers

## Quick Start

```bash
# Build
cmake --build build/x64-release --target ch11

# Test
cd ch.11.Attention
python3 test_attention.py
```

## What We're Building

Single-head scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

For now: Q = K = V = input (self-attention)

## Implementation Strategy

**Phase 1** (Current): Use existing Chapter 9 operations
- Reuse `nn.matmul`, `nn.add`, `nn.mul` from Chapter 9
- Implement transpose/reshape in C++ helper functions
- Build attention graph with OpBuilder

**Phase 2** (Later): Add new operations if needed
- `nn.transpose` for dimension permutation
- `nn.reshape` for view changes
- Only if C++ helpers become too complex

## Why This Approach?

1. **Minimalist**: Maximum reuse of existing infrastructure
2. **Educational**: Shows how to compose operations
3. **Incremental**: Build complexity gradually
4. **Practical**: Attention works without new dialect ops

## File Structure

```
ch.11.Attention/
├── README.md              # This file
├── CMakeLists.txt         # Build configuration
├── test_attention.py      # Test suite
└── src/
    └── bindings.cpp       # Attention implementation
```

## Progress

- [x] Directory structure
- [x] Test framework
- [ ] Basic attention computation graph
- [ ] Single-head attention test passing
- [ ] Multi-head attention
