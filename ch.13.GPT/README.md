# Chapter 13: Minimal GPT with RoPE

**Goal**: Implement a minimal educational GPT model with causal attention, RoPE, and autoregressive generation

## Quick Start

```bash
# Build
cmake --build build/x64-release --target ch13

# Test
cd ch.13.GPT
python3 test.py

# Generate text
python3 demo.py
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

Chapter 13 uses 12 operations: **8 common operations with linalg-based lowering** (shared architectural foundation with Chapters 11-12), and **4 GPT-specific operations with manual loop lowering** (domain-specific implementations).

### Common Operations (Linalg-based)
These 8 operations use `linalg` dialect operations for structured computation:

- **`transformer.layer_norm`** - Layer normalization using `linalg.reduce`, `linalg.fill`, `linalg.generic`
- **`transformer.linear`** - Linear transformation using `linalg.matmul`, `linalg.generic` (bias broadcast)
  - Note: Ch13 convention: weights are `(in_features, out_features)` (no transpose needed)
- **`transformer.gelu`** - GELU activation using `linalg.generic` (element-wise polynomial approximation)
- **`transformer.add`** - Element-wise addition using `linalg.generic`
- **`transformer.matmul`** - Matrix multiplication using `linalg.matmul`
- **`transformer.transpose`** - Matrix transpose using `linalg.transpose`
- **`transformer.softmax`** - Softmax using `linalg.reduce`, `linalg.fill`, `linalg.generic`
- **`transformer.scale`** - Scalar multiplication using `linalg.generic`

### GPT-Specific Operations (Manual loops)
These 4 operations use nested `scf.for` loops for domain-specific logic:

- **`transformer.embedding`** - Token lookup (integer indexing, not suitable for linalg)
- **`transformer.create_causal_mask`** - Generate lower-triangular mask (conditional logic)
- **`transformer.masked_softmax`** - Softmax with mask addition (supports 2D/3D tensors)
- **`transformer.rope`** - Rotary position embeddings (pairwise dimension rotation)

---

## Lowering Implementation

Chapter 13's lowering strategy achieves architectural consistency with Chapters 11-12 by using **linalg-based patterns** for the 8 common operations, while preserving **manual loop implementations** for the 4 GPT-specific operations that require specialized logic.

### Linalg-Based Lowering Patterns

The 8 common operations lower to structured `linalg` operations, which then convert to loops via `createConvertLinalgToLoopsPass()`:

#### 1. LayerNorm → linalg.reduce + linalg.generic
```cpp
// Step 1: Compute mean using linalg.reduce (parallel across batch, reduce across d_model)
linalg::ReduceOp meanReduce = rewriter.create<linalg::ReduceOp>(
    loc, ValueRange{input}, ValueRange{meanBuffer}, dimensions,
    [&](OpBuilder &b, Location loc, ValueRange args) {
      Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
      b.create<linalg::YieldOp>(loc, sum);
    }
);

// Step 2: Compute variance using linalg.reduce (sum of squared differences)
// Step 3: Normalize using linalg.generic (element-wise: (x - mean) * rsqrt(variance + eps))
// Step 4: Apply scale/shift using linalg.generic (element-wise: normalized * gamma + beta)
```

#### 2. Linear → linalg.matmul + linalg.generic
```cpp
// Ch13 uses PyTorch format: weight is (out_features, in_features)
// Step 1: Transpose weight to (in_features, out_features), then initialize output and perform matmul
rewriter.create<linalg::FillOp>(loc, zero, output);
rewriter.create<linalg::MatmulOp>(loc, ValueRange{input, weight}, ValueRange{output});

// Step 2: Broadcast bias across batch dimension
SmallVector<AffineExpr> biasExprs;
biasExprs.push_back(rewriter.getAffineDimExpr(1));  // Only use second dimension
AffineMap biasMap = AffineMap::get(rank, 0, biasExprs, rewriter.getContext());

rewriter.create<linalg::GenericOp>(
    loc, TypeRange{}, ValueRange{bias}, ValueRange{output},
    indexingMaps, iteratorTypes,
    [&](OpBuilder &b, Location loc, ValueRange args) {
      Value sum = b.create<arith::AddFOp>(loc, args[1], args[0]);
      b.create<linalg::YieldOp>(loc, sum);
    }
);
```

**Note**: Chapter 13's `LinearOp` differs from Chapter 12's implementation:
- **Ch12**: Weights are `(out_features, in_features)`, requires `linalg.transpose` before matmul
- **Ch13**: Weights are `(out_features, in_features)` (PyTorch format), requires `linalg.transpose` before matmul
- Both chapters use the same PyTorch weight convention; Ch13 also transposes weights during lowering

#### 3. GELU → linalg.generic (element-wise)
```cpp
// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
rewriter.create<linalg::GenericOp>(
    loc, TypeRange{}, ValueRange{input}, ValueRange{output},
    indexingMaps, iteratorTypes,
    [&](OpBuilder &b, Location loc, ValueRange args) {
      Value x = args[0];
      // Compute x^3
      Value x2 = b.create<arith::MulFOp>(loc, x, x);
      Value x3 = b.create<arith::MulFOp>(loc, x2, x);
      // ... (polynomial approximation)
      Value result = /* computed result */;
      b.create<linalg::YieldOp>(loc, result);
    }
);
```

#### 4. Matmul → linalg.matmul
```cpp
// Initialize output to zero, then perform matmul
rewriter.create<linalg::FillOp>(loc, zero, output);
rewriter.create<linalg::MatmulOp>(loc, ValueRange{lhs, rhs}, ValueRange{output});
```

#### 5. Transpose → linalg.transpose
```cpp
// Swap dimensions 0 and 1
rewriter.create<linalg::TransposeOp>(
    loc, input, output,
    SmallVector<int64_t>{1, 0}  // permutation
);
```

#### 6. Softmax → linalg.reduce + linalg.generic
```cpp
// Step 1: Find max using linalg.reduce (for numerical stability)
linalg::ReduceOp maxReduce = rewriter.create<linalg::ReduceOp>(
    loc, ValueRange{input}, ValueRange{maxBuffer}, dimensions,
    [&](OpBuilder &b, Location loc, ValueRange args) {
      Value max = b.create<arith::MaximumFOp>(loc, args[0], args[1]);
      b.create<linalg::YieldOp>(loc, max);
    }
);

// Step 2: Compute exp(x - max) and sum using linalg.generic + linalg.reduce
// Step 3: Normalize: exp(x - max) / sum
```

#### 7. Add → linalg.generic
```cpp
// Element-wise addition with broadcasting support
rewriter.create<linalg::GenericOp>(
    loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{output},
    indexingMaps, iteratorTypes,
    [&](OpBuilder &b, Location loc, ValueRange args) {
      Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
      b.create<linalg::YieldOp>(loc, sum);
    }
);
```

#### 8. Scale → linalg.generic
```cpp
// Scalar multiplication: output[i,j] = input[i,j] * scalar
rewriter.create<linalg::GenericOp>(
    loc, TypeRange{}, ValueRange{input, scalar}, ValueRange{output},
    indexingMaps, iteratorTypes,
    [&](OpBuilder &b, Location loc, ValueRange args) {
      Value product = b.create<arith::MulFOp>(loc, args[0], args[1]);
      b.create<linalg::YieldOp>(loc, product);
    }
);
```

### Manual Loop Lowering (GPT-Specific Operations)

The 4 GPT-specific operations use nested `scf.for` loops because they require logic not expressible in linalg's structured iteration model:

#### 1. Embedding → scf.for (integer indexing)
```cpp
// Token lookup requires integer index → memref access
rewriter.create<scf::ForOp>(
    loc, zeroIdx, seqLenVal, oneIdx, ValueRange{},
    [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
      // Load token ID, convert int32 → index
      Value tokenId = builder.create<memref::LoadOp>(loc, indices, ValueRange{i});
      Value tokenIdx = builder.create<arith::IndexCastOp>(loc, indexType, tokenId);
      
      // Copy embedding vector: output[i,:] = table[tokenIdx,:]
      builder.create<scf::ForOp>(/* nested loop for d_model dimension */);
    }
);
```

#### 2. CreateCausalMask → scf.for (conditional logic)
```cpp
// Lower triangular mask: mask[i,j] = (j > i) ? -inf : 0.0
rewriter.create<scf::ForOp>(
    loc, zeroIdx, seqLenVal, oneIdx, ValueRange{},
    [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
      builder.create<scf::ForOp>(
          loc, zeroIdx, seqLenVal, oneIdx, ValueRange{},
          [&](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {
            Value iIndex = builder.create<arith::IndexCastOp>(loc, i64Type, i);
            Value jIndex = builder.create<arith::IndexCastOp>(loc, i64Type, j);
            Value isUpperTri = builder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::sgt, jIndex, iIndex);
            Value maskVal = builder.create<arith::SelectOp>(
                loc, isUpperTri, negInf, zero);
            builder.create<memref::StoreOp>(loc, maskVal, output, ValueRange{i, j});
          });
    });
```

#### 3. MaskedSoftmax → scf.for (2D/3D flexibility)
```cpp
// Softmax with mask addition, supports both 2D and 3D inputs
size_t ndim = shape.size();
bool is2D = (ndim == 2);

// Step 1: Find max(logits + mask) for numerical stability
// Step 2: Compute sum of exp(logits[i] + mask[i] - max)
// Step 3: Normalize: output[i] = exp(logits[i] + mask[i] - max) / sum

// Conditional indexing for 2D vs 3D:
Value logit = is2D 
    ? builder.create<memref::LoadOp>(loc, input, ValueRange{j, k})
    : builder.create<memref::LoadOp>(loc, input, ValueRange{i, j, k});
```

#### 4. RoPE → scf.for (pairwise rotation)
```cpp
// Rotary position embeddings: rotate each pair of dimensions
// For position i, dimension pair (2j, 2j+1):
//   θ = 10000^(-2j/d) * i
//   output[i, 2j]   = input[i, 2j]   * cos(θ) - input[i, 2j+1] * sin(θ)
//   output[i, 2j+1] = input[i, 2j+1] * cos(θ) + input[i, 2j]   * sin(θ)

rewriter.create<scf::ForOp>(
    loc, zeroIdx, seqLenVal, oneIdx, ValueRange{},
    [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
      Value posFloat = builder.create<arith::IndexCastOp>(loc, f32Type, i);
      
      builder.create<scf::ForOp>(/* loop over dimension pairs, apply rotation */);
    }
);
```

### Lowering Pipeline

The complete lowering pipeline in `bindings.cpp`:

```cpp
// 1. Custom dialect → Standard + Linalg dialects
pm.addPass(createTransformerToStandardPass());

// 2. Linalg → Loops (converts all 8 linalg operations to scf.for)
pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());

// 3. Standard dialects → LLVM IR
pm.addPass(createSCFToControlFlowPass());
pm.addPass(createFinalizeMemRefToLLVMConversionPass());
pm.addPass(createConvertFuncToLLVMPass());
pm.addPass(createReconcileUnrealizedCastsPass());
```

After lowering, all operations (both linalg-based and manual loops) exist as `scf.for` loops, which then convert uniformly to LLVM IR.

### Benefits of Hybrid Approach

This hybrid lowering strategy provides:

1. **Architectural Consistency**: 8 common operations share implementation across Chapters 11-13
2. **Expressiveness**: 4 GPT-specific operations retain manual control for complex logic
3. **Optimization Opportunities**: Linalg operations benefit from structured transformation passes
4. **Maintainability**: Changes to common operations propagate automatically
5. **Educational Clarity**: Linalg patterns demonstrate structured compilation; manual loops demonstrate direct control

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