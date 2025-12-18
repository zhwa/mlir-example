# Chapter 13: Minimal GPT with RoPE

**Goal**: Implement a minimal educational GPT model with causal attention, RoPE, and autoregressive generation

## Quick Start

```bash
# Build
cmake --build build/x64-release --target ch13

# Test
cd ch.13.GPT
PYTHONPATH=../build/x64-release/ch.13.GPT python3 test.py

# Generate text
PYTHONPATH=../build/x64-release/ch.13.GPT python3 demo.py
```

## Model Architecture

**Minimal Configuration** (educational scale):
```python
vocab_size = 256       # Byte-level tokens
d_model = 64           # Hidden dimension
num_heads = 4          # Attention heads
head_dim = 16          # d_model / num_heads
num_layers = 2         # Transformer layers
max_seq_len = 32       # Maximum sequence length
```

**Components**:
1. Token embeddings (lookup table)
2. Rotary Position Embeddings (RoPE)
3. Causal multi-head attention
4. Feed-forward networks
5. Layer normalization
6. Autoregressive generation

## Key Features

### 1. Causal Masking
Prevents attention from looking at future tokens:
```
Mask (4x4):
[ 0   -inf -inf -inf]
[ 0    0   -inf -inf]
[ 0    0    0   -inf]
[ 0    0    0    0  ]
```

### 2. Rotary Position Embeddings (RoPE)
Modern positional encoding that:
- Applies rotation to Q and K (not V)
- Better length extrapolation than learned embeddings
- Used in GPT-NeoX, LLaMA, PaLM

**RoPE Formula**:
```
For each position pos and dimension pair (2d, 2d+1):
θ_d = 10000^(-2d/head_dim)
angle = pos * θ_d

[x_2d    ]   [cos(angle)  -sin(angle)] [x_2d    ]
[x_2d+1  ] = [sin(angle)   cos(angle)] [x_2d+1  ]
```

### 3. Autoregressive Generation
```python
tokens = [prompt_tokens]
for _ in range(max_new_tokens):
    logits = model(tokens)           # Forward pass
    next_token = sample(logits[-1])  # Sample from last position
    tokens.append(next_token)
```

## API

### Basic Usage
```python
import ch13
import numpy as np

# Initialize model weights (random for demo)
model_config = {
    'vocab_size': 256,
    'd_model': 64,
    'num_heads': 4,
    'num_layers': 2,
    'max_seq_len': 32
}

weights = ch13.init_random_weights(model_config)

# Forward pass
tokens = np.array([1, 2, 3, 4], dtype=np.int32)
logits = ch13.forward(ch13.gpt_model(tokens, weights))

# Shape: [seq_len, vocab_size]
print(logits.shape)  # (4, 256)
```

### Text Generation
```python
# Character-level generation
prompt = "Hello"
prompt_tokens = [ord(c) for c in prompt]

output_tokens = ch13.generate(
    weights, 
    prompt_tokens, 
    max_new_tokens=50,
    temperature=0.8
)

output_text = ''.join([chr(t) for t in output_tokens])
print(output_text)
```

## Operations

### New Operations (Chapter 13)
- `transformer.embedding` - Token lookup
- `transformer.rope` - Rotary position embeddings
- `transformer.masked_softmax` - Softmax with causal mask

### Reused from Chapter 12
- `transformer.layer_norm` - Layer normalization
- `transformer.linear` - Linear transformation
- `transformer.gelu` - GELU activation
- `transformer.add` - Element-wise addition
- `transformer.matmul` - Matrix multiplication
- `transformer.transpose` - Transpose
- `transformer.softmax` - Softmax
- `transformer.scale` - Scalar multiplication

---

## Implementation Phases

**Phase 1: Setup & Foundation** ✅ **COMPLETE**
- Directory structure, CMake configuration
- Module builds and imports successfully
- Tests: Module import ✓

**Phase 2: Embedding Layer** ✅ **COMPLETE**
- Operation: `transformer.embedding`
- Lowering: Nested loops with int32→index conversion
- C++ helper: `embedding(indices, table)`
- Tests: Basic lookup ✓, GPT-scale dimensions ✓
- All tests passing with numerical validation

**Phase 3: Causal Masking** ✅ **COMPLETE**
- Operations: `transformer.create_causal_mask`, `transformer.masked_softmax`
- Lowering: Lower triangular mask generation, masked softmax with -inf for future positions
- C++ helpers: `create_causal_mask(seq_len)`, `masked_softmax(logits, mask)`
- 3D memref support added for attention tensors
- Tests: Mask structure ✓, Uniform logits ✓, Varied logits ✓
- All tests passing with numerical validation

**Phase 4: Rotary Position Embeddings (RoPE)** ✅ **COMPLETE**
- Operation: `transformer.rope`
- Lowering: Pairwise rotation with position-dependent angles (θ = 10000^(-2j/d))
- Formula: output[i,2j] = x[i,2j]*cos(i*θ) - x[i,2j+1]*sin(i*θ)
- C++ helper: `rope(input)`
- Tests: Basic transformation ✓, NumPy reference ✓, Norm preservation ✓, Position differentiation ✓
- All tests passing with numerical validation

**Phase 5: GPT Model Composition** ✅ **COMPLETE**
- C++ functions: `gpt_attention`, `gpt_block`, `gpt_forward`
- Full pipeline: Token IDs → Embedding → N transformer blocks → LayerNorm → Hidden states
- Pre-norm architecture with residual connections
- Bug fixes: MaskedSoftmaxOp 2D/3D support, parameter indexing for mixed int32/float params
- Tests: GPT attention ✓, Single block ✓, Full forward (1 layer) ✓, Full forward (2 layers) ✓
- All tests passing with numerical validation

### Phase 5 Debugging Deep Dive

This phase revealed two subtle but critical bugs that are worth documenting for future reference.

#### Bug #1: MaskedSoftmaxOp Dimension Mismatch

**Symptom**: `Assertion 'Index < Length && "Invalid index!"' failed` in llvm::ArrayRef

**Root Cause**: The original `MaskedSoftmaxOpLowering` assumed 3D input tensors `[batch, seq_len, seq_len]` for batch attention. However, in GPT attention, when processing a single sequence, `matmul(Q_rope, K_T)` produces 2D scores `[seq_len, seq_len]` (no batch dimension).

**Debugging Process**:
1. Traced crash to MaskedSoftmaxOp lowering accessing out-of-bounds indices
2. Identified that attention scores are 2D but lowering expected 3D
3. Confirmed by checking Phase 1-4 tests (which use 3D batched tensors) vs Phase 5 (single sequence)

**Solution**: Modified `MaskedSoftmaxOpLowering` (TransformerPasses.cpp ~lines 640-760) to detect dimensionality:
```cpp
size_t ndim = shape.size();
bool is2D = (ndim == 2);

// For 2D: treat as batch=1, use indices [j, k]
// For 3D: use indices [i, j, k]
Value logit = is2D 
    ? builder.create<memref::LoadOp>(loc, input, ValueRange{j, k})
    : builder.create<memref::LoadOp>(loc, input, ValueRange{i, j, k});
```

Applied conditional indexing to all three loop bodies (max computation, exp sum, softmax output).

#### Bug #2: Parameter Indexing with Mixed int32/float Types

**Symptom**: 
- First error: `'arith.index_cast' op operand #0 must be signless-integer-like but got 'f32'`
- After partial fix: `'memref.load' op incorrect number of indices, expected 1 but got 2`
- After second fix: `'arith.mulf' op operand #1 must be floating-point-like but got 'i32'`

**Root Cause**: The compilation pipeline uses a single `paramIndex` map to track both int32 parameters (embedding indices) and float parameters (weights, biases, tables). Function arguments are laid out as:
```
[inputs..., int32_params..., float_params..., output]
```

However, during parameter collection (DFS graph traversal), we computed indices using:
- Float params: `paramIndex[ptr] = int32_parameters.size() + parameters.size()`
- Int32 params: `paramIndex[ptr] = int32_parameters.size()`

The bug: When collecting float params **before** int32 params (e.g., LayerNorm's gamma/beta before Embedding's indices), `int32_parameters.size()` is still 0, causing float params to overlap with int32 params at index 0, 1, 2...

**Debugging Process**:
1. Initial fix: Separated int32 param indexing from float param indexing
2. Realized float params still calculated with `int32_parameters.size()` during collection
3. Traced parameter collection order: LayerNorm (gamma, beta) → Embedding (indices, table)
4. Discovered the race condition: gamma gets index 0, but indices also gets index 0!
5. Created isolated test: `embedding + layer_norm` (no Input nodes) to reproduce

**Solution** (bindings.cpp ~lines 738-790):
1. **During collection**: Use only local array sizes (no cross-contamination)
   ```cpp
   // Float params
   paramIndex[ptr] = parameters.size();  // Don't add int32_parameters.size() yet

   // Int32 params  
   paramIndex[ptr] = int32_parameters.size();
   ```

2. **After collection**: Adjust all float param indices by the final int32 count
   ```cpp
   int int32_count = static_cast<int>(int32_parameters.size());
   if (int32_count > 0) {
     for (auto& [ptr, idx] : paramIndex) {
       bool is_int32 = /* check if ptr is in int32_parameters */;
       if (!is_int32) {
         idx += int32_count;  // Shift float params after int32 params
       }
     }
   }
   ```

This ensures:
- Int32 params: indices 0, 1, 2, ... (unchanged)
- Float params: indices N, N+1, N+2, ... (where N = int32_parameters.size())
- Matches the function signature layout perfectly

**Key Lesson**: When mixing different parameter types in a single index namespace, **timing matters**. Compute indices relative to local arrays during collection, then apply global offsets in a post-processing pass.

**Phase 6: Autoregressive Generation** ✅ **COMPLETE**
- Python module: `generation.py` with `generate()` and `generate_greedy()` functions
- Sampling strategies: Greedy (argmax), temperature sampling, top-k filtering
- Generation loop: Forward pass → extract last position → project to vocab → sample → append token
- Output projection: Use embedding table as tied weights (embedding.T = output projection)
- Tests: Deterministic generation ✓, Different prompts ✓, Softmax sampling ✓, Full-scale generation ✓
- Demo script: `generation.py` with "Hello" prompt example
- Note: Model is randomly initialized, so output is random. A trained model would generate coherent text.

**Phase 7: Demo & Documentation** ✅ COMPLETE
- Interactive `demo.py` demo script with 3 demonstrations
- Complete API reference and usage examples
- Educational explanations about model architecture
- Results: All demos working ✓, Documentation complete ✓

---

## Getting Started

### Build the Project

```bash
cd ch.13.GPT
mkdir -p build && cd build
cmake .. -GNinja
ninja
cd ..
```

### Run Tests

```bash
python3 test.py
```

Expected output:
```
✓ Phase 1: Module import test passed!
✓ Phase 2: All embedding tests passed!
✓ Phase 3: All causal masking tests passed!
✓ Phase 4: All RoPE tests passed!
✓ Phase 5: All GPT model composition tests passed!
✓ Phase 6: All autoregressive generation tests passed!
Test suite complete - Phases 1-6 passing!
```

### Run Interactive Demo

```bash
python3 demo.py
```

This runs an interactive demo with:
- **Demo 1**: Basic greedy generation
- **Demo 2**: Temperature comparison (0.5, 1.0, 1.5)
- **Demo 3**: Multiple different prompts

Example output:
```
Minimal GPT - Text Generation Demo
...
Prompt: 'Hello'
Generating 20 tokens (greedy)...
Output: 'Hellooooooooooooooooooooo'
```

⚠️ **Note**: Model has random weights (not trained), so output demonstrates the generation mechanism but is not coherent text.

---

## API Reference

### C++ Functions (via Python bindings)

**Embedding Operations**
```python
# Token embedding
embedded = ch13.embedding(indices, embedding_table)  # [seq_len, d_model]
```

**Positional Encoding**
```python
# Rotary Position Embedding (RoPE)
q_rot = ch13.rope(q, dim=0)  # Apply RoPE to queries
k_rot = ch13.rope(k, dim=0)  # Apply RoPE to keys
```

**Attention Operations**
```python
# Create causal mask [seq_len, seq_len]
mask = ch13.create_causal_mask(seq_len)

# Masked softmax (prevents attention to future tokens)
attn = ch13.masked_softmax(scores, mask)  # [batch, seq_len, seq_len]
```

**Normalization**
```python
# Layer normalization
normalized = ch13.layer_norm(x, gamma, beta)  # [*, d_model]
```

**Activation**
```python
# GELU activation
activated = ch13.gelu(x)  # [*, d_model]
```

**GPT Model**
```python
# Full GPT forward pass
hidden_states = ch13.gpt_forward(
    indices,           # [seq_len] int32
    embedding_table,   # [vocab_size, d_model]
    all_weights,       # List of layer weights
    final_gamma,       # [d_model]
    final_beta         # [d_model]
)  # Returns: [seq_len, d_model]
```

### Python Generation Functions

**Autoregressive Generation**
```python
from generation import generate, generate_greedy

# Greedy generation (deterministic)
output = generate_greedy(
    prompt_tokens,      # np.array [seq_len] int32
    embedding_table,    # np.array [vocab_size, d_model]
    all_weights,        # List of transformer weights
    final_gamma,        # np.array [d_model]
    final_beta,         # np.array [d_model]
    max_new_tokens=20,  # Number of tokens to generate
    max_seq_len=32      # Maximum sequence length
)

# Stochastic generation with temperature and top-k
output = generate(
    prompt_tokens,
    embedding_table,
    all_weights,
    final_gamma,
    final_beta,
    max_new_tokens=20,
    temperature=1.0,    # Higher = more random (default: 1.0)
    top_k=50,          # Top-k filtering (default: None)
    max_seq_len=32
)
```

---

## Model Architecture Details

### Configuration
- **Vocabulary**: 256 tokens (byte-level)
- **Hidden dimension**: 64
- **Feed-forward dimension**: 256 (4× hidden)
- **Number of layers**: 2
- **Attention heads**: 4
- **Maximum sequence length**: 32

### Transformer Block Structure
```
Input
  ↓
LayerNorm
  ↓
Multi-Head Attention (with RoPE and causal masking)
  ↓
Residual Connection
  ↓
LayerNorm
  ↓
Feed-Forward Network (Linear → GELU → Linear)
  ↓
Residual Connection
  ↓
Output
```

### Parameter Count
Per layer:
- Q, K, V, O projections: 4 × (64×64 + 64) = 16,640
- FFN: (64×256 + 256) + (256×64 + 64) = 16,832 + 16,448 = 33,280
- Layer norms: 4 × 64 = 256

Total per layer: 50,176 parameters
**Total model**: ~100K parameters (2 layers + embedding table)

---

## Troubleshooting

### Build Issues

**Missing MLIR/LLVM**
```bash
# Make sure LLVM is built with MLIR enabled
cmake -DLLVM_ENABLE_PROJECTS=mlir ...
```

**CMake can't find MLIR**
```bash
# Set MLIR_DIR to your LLVM build
export MLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
```

### Runtime Issues

**Module import fails**
```python
# Make sure you're importing from the build directory
import sys
sys.path.insert(0, '../build/x64-release/ch.13.GPT')
```

**Shape mismatch errors**
- Check that your model config matches test expectations (vocab_size=256, d_model=64)
- Verify weight dimensions: Q/K/V/O should be [d_model, d_model]

**Generation produces empty output**
- Ensure prompt is not empty
- Check max_seq_len is not exceeded
- Verify embedding_table shape is [vocab_size, d_model]

### Test Failures

Run individual phase tests:
```python
# In test.py, comment out phases to isolate issues
# Example: test only Phase 3
print("=== Phase 3: Causal Masking ===")
test_causal_mask()
test_masked_softmax_2d()
test_masked_softmax_3d()
```