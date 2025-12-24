# Chapter 13: GPT Architecture (Part 2)

Chapter 13 Part 1 built the complete GPT forward pass: token embeddings, causal masking, stacked transformer blocks, and output projection to vocabulary logits. Given input token IDs, the model computes next-token predictions. Part 2 completes the GPT implementation by adding **Rotary Position Embeddings (RoPE)**—modern positional encoding enabling better length extrapolation—and **autoregressive generation**—the iterative process of producing text one token at a time. We'll implement sampling strategies (temperature scaling, top-k filtering, nucleus sampling) and demonstrate end-to-end text generation.

RoPE represents a significant improvement over earlier positional encoding methods (learned embeddings, sinusoidal functions). Used in GPT-NeoX, LLaMA, PaLM, and many modern models, RoPE applies rotations to query and key vectors based on position, encoding relative positional information directly into attention scores. This approach generalizes better to longer sequences than the model saw during training—a critical property for production language models serving diverse context lengths.

Autoregressive generation transforms the forward pass (batch prediction over known sequences) into iterative token production. Starting from a prompt, we predict the next token, append it to the sequence, predict again, and repeat. Sampling strategies control this process: **temperature** adjusts confidence, **top-k** restricts candidates to most probable tokens, **nucleus (top-p)** samples from a cumulative probability mass. These techniques balance creativity and coherence, enabling diverse text generation from the same prompt.

By the end of Part 2, you'll have a complete GPT implementation capable of generating text—small-scale but architecturally identical to production systems. Chapter 14 will then optimize this architecture for production serving with KV caching, FlashAttention fusion, and batching strategies.

## 13.8 Rotary Position Embeddings (RoPE)

Transformers lack inherent position awareness—self-attention is permutation-invariant, treating `[A, B, C]` identically to `[C, A, B]` without positional information. Position embeddings address this by encoding token positions into the model. RoPE represents the state-of-the-art approach, offering superior length extrapolation and mathematical elegance compared to earlier methods. This section explains RoPE's geometric intuition, implements the rotation operation in MLIR, and demonstrates integration into attention.

**The Position Encoding Problem**. Consider the sentence "The cat sat on the mat" vs "The mat sat on the cat." Without position information, attention cannot distinguish these—both contain identical tokens, just reordered. Position embeddings must:

1. **Inject position information**: Distinguish token positions 0, 1, 2, ...
2. **Preserve relative distances**: "cat" and "sat" are adjacent in one sentence, distant in another
3. **Extrapolate**: Generalize to sequence lengths beyond training (e.g., trained on 512 tokens, inference on 2048)

Earlier approaches struggled with requirement 3. **Learned position embeddings** (GPT-2) learn separate vectors for each position but can't handle positions beyond the maximum training length. **Sinusoidal embeddings** (original Transformer) extrapolate better but don't explicitly encode relative positions in attention.

**RoPE's Core Idea**. RoPE rotates query and key vectors by angles proportional to their positions. For position `m` and dimension pair `(2d, 2d+1)`:

```
Rotation matrix R_m:
R_m = [cos(m·θ_d)  -sin(m·θ_d)]
      [sin(m·θ_d)   cos(m·θ_d)]

Where θ_d = 10000^(-2d/d_model)
```

This rotation is applied to query and key vectors **before** computing attention scores. The key insight: when queries and keys are rotated by position-dependent angles, their dot product naturally encodes relative position.

**Why Rotation Encodes Relative Position**. Consider two positions `m` and `n`. After rotating their query and key vectors:

```
Q_rotated[m] = R_m @ Q[m]
K_rotated[n] = R_n @ K[n]

Attention score:
Q_rotated[m]^T @ K_rotated[n] = Q[m]^T @ R_m^T @ R_n @ K[n]
                               = Q[m]^T @ R_{n-m} @ K[n]
```

The matrices `R_m^T @ R_n` simplify to `R_{n-m}`—a rotation by the **relative position** `n - m`. This means attention scores depend only on relative distance between tokens, not absolute positions. This property is desirable: in many languages, relationships between words depend on their relative distance (adjacent, nearby, distant) rather than absolute positions.

**RoPE Implementation**. For a query or key vector `x` of dimension `d_model` at position `pos`:

```python
def apply_rope(x, pos, d_model):
    """
    Apply rotary position embedding to vector x.
    
    Args:
        x: [d_model] query or key vector
        pos: integer position
        d_model: model dimension (must be even)
    
    Returns:
        rotated: [d_model] position-encoded vector
    """
    assert d_model % 2 == 0, "d_model must be even for RoPE"
    
    rotated = np.zeros_like(x)
    
    for d in range(0, d_model, 2):
        # Compute rotation angle
        theta_d = 10000 ** (-2 * d / d_model)
        angle = pos * theta_d
        
        # Apply rotation to dimension pair (2d, 2d+1)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        x_even = x[d]
        x_odd = x[d + 1]
        
        rotated[d] = cos_angle * x_even - sin_angle * x_odd
        rotated[d + 1] = sin_angle * x_even + cos_angle * x_odd
    
    return rotated
```

This rotates each consecutive pair of dimensions by position-dependent angles. Lower dimensions rotate faster (larger θ), higher dimensions rotate slower—creating a spectrum of positional frequencies.

**Batched RoPE for Sequences**. In practice, we apply RoPE to entire sequences at once:

```python
def apply_rope_batch(Q, positions):
    """
    Apply RoPE to query matrix.
    
    Args:
        Q: [seq_len, d_model] query vectors
        positions: [seq_len] integer positions
    
    Returns:
        Q_rotated: [seq_len, d_model]
    """
    seq_len, d_model = Q.shape
    Q_rotated = np.zeros_like(Q)
    
    for i in range(seq_len):
        Q_rotated[i] = apply_rope(Q[i], positions[i], d_model)
    
    return Q_rotated
```

The same rotation applies to key vectors. Value vectors are **not** rotated—RoPE only affects attention score computation (Q @ K^T), not the value aggregation.

**MLIR Operation Definition**. We define a RoPE operation in the Transformer dialect:

```tablegen
// inc/TransformerOps.td
def Transformer_RoPEOp : Transformer_Op<"rope", [Pure]> {
  let summary = "Rotary position embedding";
  let description = [{
    Applies rotary position embeddings to query or key vectors:
    
    For each position m and dimension pair (2d, 2d+1):
      θ_d = 10000^(-2d/d_model)
      angle = m * θ_d
      
      output[m, 2d]   = cos(angle) * input[m, 2d] - sin(angle) * input[m, 2d+1]
      output[m, 2d+1] = sin(angle) * input[m, 2d] + cos(angle) * input[m, 2d+1]
    
    This encodes relative positional information into attention scores.
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,      // [seq_len, d_model]
    AnyRankedTensor:$positions   // [seq_len], int32 positions
  );
  
  let results = (outs AnyRankedTensor:$output);  // [seq_len, d_model]
  
  let assemblyFormat = [{
    $input `,` $positions attr-dict `:` 
    type($input) `,` type($positions) `->` type($output)
  }];
}
```

The operation takes input vectors and position indices, returning rotated vectors with encoded positional information.

**Lowering RoPE to Standard Dialects**. The lowering generates loops computing rotations:

```cpp
// src/TransformerToStandard.cpp
struct RoPEOpLowering : public OpRewritePattern<RoPEOp> {
  using OpRewritePattern::OpRewritePattern;
  
  LogicalResult matchAndRewrite(RoPEOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value positions = op.getPositions();
    
    auto inputType = input.getType().cast<RankedTensorType>();
    int64_t seqLen = inputType.getShape()[0];
    int64_t dModel = inputType.getShape()[1];
    
    // Constants
    Value baseConst = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF32FloatAttr(10000.0f)
    );
    Value dModelFloat = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF32FloatAttr(static_cast<float>(dModel))
    );
    
    // Generate rotation for each position and dimension pair
    Value output = rewriter.create<linalg::GenericOp>(
      loc, inputType, ValueRange{input, positions},
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Get indices
        Value seqIdx = b.create<linalg::IndexOp>(loc, 0);  // Position in sequence
        Value dimIdx = b.create<linalg::IndexOp>(loc, 1);  // Dimension
        
        // Determine if this is even or odd dimension
        Value two = b.create<arith::ConstantIndexOp>(loc, 2);
        Value dimParity = b.create<arith::RemUIOp>(loc, dimIdx, two);
        Value isEven = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, dimParity, 
          b.create<arith::ConstantIndexOp>(loc, 0)
        );
        
        // Compute base dimension (floor to even)
        Value baseDim = b.create<arith::DivUIOp>(loc, dimIdx, two);
        Value baseDimFloat = b.create<arith::IndexCastOp>(loc, b.getF32Type(), baseDim);
        
        // Compute θ_d = 10000^(-2d/d_model)
        Value exponent = b.create<arith::MulFOp>(loc, baseDimFloat, 
          b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(-2.0f))
        );
        exponent = b.create<arith::DivFOp>(loc, exponent, dModelFloat);
        Value theta = b.create<math::PowFOp>(loc, baseConst, exponent);
        
        // Get position and compute angle
        Value pos = args[1];  // positions[seqIdx]
        Value posFloat = b.create<arith::SIToFPOp>(loc, b.getF32Type(), pos);
        Value angle = b.create<arith::MulFOp>(loc, posFloat, theta);
        
        // Compute cos and sin
        Value cosAngle = b.create<math::CosOp>(loc, angle);
        Value sinAngle = b.create<math::SinOp>(loc, angle);
        
        // Get paired values (even/odd dimensions)
        Value evenDimIdx = b.create<arith::MulIOp>(loc, baseDim, two);
        Value oddDimIdx = b.create<arith::AddIOp>(loc, evenDimIdx, 
          b.create<arith::ConstantIndexOp>(loc, 1)
        );
        
        Value xEven = b.create<tensor::ExtractOp>(loc, input, ValueRange{seqIdx, evenDimIdx});
        Value xOdd = b.create<tensor::ExtractOp>(loc, input, ValueRange{seqIdx, oddDimIdx});
        
        // Compute rotation
        Value cosXeven = b.create<arith::MulFOp>(loc, cosAngle, xEven);
        Value sinXodd = b.create<arith::MulFOp>(loc, sinAngle, xOdd);
        Value sinXeven = b.create<arith::MulFOp>(loc, sinAngle, xEven);
        Value cosXodd = b.create<arith::MulFOp>(loc, cosAngle, xOdd);
        
        Value rotatedEven = b.create<arith::SubFOp>(loc, cosXeven, sinXodd);
        Value rotatedOdd = b.create<arith::AddFOp>(loc, sinXeven, cosXodd);
        
        // Select based on dimension parity
        Value result = b.create<arith::SelectOp>(loc, isEven, rotatedEven, rotatedOdd);
        b.create<linalg::YieldOp>(loc, result);
      }
    );
    
    rewriter.replaceOp(op, output);
    return success();
  }
};
```

This generates efficient code: the trigonometric functions (`cos`, `sin`, `pow`) compile to LLVM intrinsics that map to hardware instructions or optimized library calls.

**Integrating RoPE into Attention**. RoPE modifies the attention computation:

```python
def rope_attention(Q, K, V, positions):
    """
    Attention with rotary position embeddings.
    
    Args:
        Q, K, V: [seq_len, d_model] query, key, value matrices
        positions: [seq_len] position indices
    
    Returns:
        output: [seq_len, d_model] attention output
    """
    # Apply RoPE to queries and keys (not values!)
    Q_rot = apply_rope_batch(Q, positions)
    K_rot = apply_rope_batch(K, positions)
    
    # Standard scaled dot-product attention
    d_k = Q.shape[-1]
    scores = (Q_rot @ K_rot.T) / np.sqrt(d_k)
    
    # Causal mask (Part 1)
    mask = np.triu(np.ones_like(scores), k=1) * -np.inf
    masked_scores = scores + mask
    
    attention_weights = softmax(masked_scores)
    output = attention_weights @ V  # V is NOT rotated
    
    return output
```

RoPE only modifies Q and K, leaving V unchanged. This asymmetry is intentional: position information affects attention scores (which positions to attend to) but not value aggregation (what information to extract).

**Testing RoPE**. Validation focuses on relative position invariance:

```python
def test_rope_relative_invariance():
    """Test that RoPE encodes relative, not absolute, positions."""
    d_model = 32
    Q = np.random.randn(4, d_model).astype(np.float32)
    K = np.random.randn(4, d_model).astype(np.float32)
    
    # Positions [0, 1, 2, 3]
    positions1 = np.array([0, 1, 2, 3], dtype=np.int32)
    Q_rot1 = gpt.rope(Q, positions1)
    K_rot1 = gpt.rope(K, positions1)
    scores1 = Q_rot1 @ K_rot1.T
    
    # Positions [10, 11, 12, 13] - shifted by 10
    positions2 = np.array([10, 11, 12, 13], dtype=np.int32)
    Q_rot2 = gpt.rope(Q, positions2)
    K_rot2 = gpt.rope(K, positions2)
    scores2 = Q_rot2 @ K_rot2.T
    
    # Attention scores should be similar (relative positions are the same)
    # scores[i, j] depends on |i - j|, not absolute i or j
    np.testing.assert_allclose(scores1, scores2, rtol=0.1)
    print("✓ RoPE relative invariance test passed")

def test_rope_numerical():
    """Test RoPE against reference implementation."""
    Q = np.random.randn(8, 64).astype(np.float32)
    positions = np.arange(8, dtype=np.int32)
    
    Q_rot_mlir = gpt.rope(Q, positions)
    Q_rot_ref = reference_rope(Q, positions)
    
    np.testing.assert_allclose(Q_rot_mlir, Q_rot_ref, rtol=1e-4)
    print("✓ RoPE numerical test passed")
```

The relative invariance test verifies RoPE's key property: shifting all positions by a constant doesn't change attention patterns.

**RoPE vs Learned Embeddings**. Advantages of RoPE:

1. **Length Extrapolation**: Works on sequences longer than training (within limits—10× longer is often fine)
2. **Parameter Efficiency**: No learned parameters (sinusoidal functions are fixed)
3. **Relative Position Encoding**: Naturally captures relative distances
4. **Hardware Efficiency**: Rotation is compute-light compared to embedding lookups and additions

Disadvantages:
- Slightly more complex to implement
- Requires even `d_model` (or special handling for odd dimensions)

Modern models (LLaMA, GPT-NeoX, PaLM) overwhelmingly prefer RoPE for its extrapolation properties—critical for serving models on diverse context lengths.

## 13.9 Autoregressive Text Generation

With RoPE-augmented attention complete, we turn to **generation**—producing text one token at a time. Autoregressive generation transforms batch prediction (forward pass over complete sequences) into iterative token production, conditioning each new token on all previous tokens. This section implements the generation loop, explores sampling strategies, and demonstrates how hyperparameters affect output diversity.

**The Generation Loop**. Starting from a prompt, generate `max_new_tokens` additional tokens:

```python
def generate(prompt_tokens, model_weights, max_new_tokens=50):
    """
    Generate text autoregressively.
    
    Args:
        prompt_tokens: [prompt_len] initial context
        model_weights: GPT model parameters
        max_new_tokens: number of tokens to generate
    
    Returns:
        generated_tokens: [prompt_len + max_new_tokens]
    """
    tokens = list(prompt_tokens)
    
    for _ in range(max_new_tokens):
        # Forward pass: get logits for all positions
        logits = gpt_forward(np.array(tokens), model_weights)  # [current_len, vocab_size]
        
        # Extract last position logits (next token prediction)
        next_token_logits = logits[-1, :]  # [vocab_size]
        
        # Sample next token
        next_token = sample(next_token_logits)
        
        # Append to sequence
        tokens.append(next_token)
    
    return np.array(tokens, dtype=np.int32)
```

Each iteration:
1. Run forward pass on current sequence (includes prompt + all generated tokens so far)
2. Extract logits for the last position
3. Sample a token from the logit distribution
4. Append to sequence and repeat

The forward pass reprocesses the entire sequence each iteration—this is **inefficient** but simple. Chapter 14 optimizes this with **KV caching**, reusing previously computed attention keys and values.

**Greedy vs Sampling Decoding**. Two fundamental approaches:

**Greedy Decoding**: Always pick the most probable token:

```python
def sample_greedy(logits):
    """Always select highest-probability token."""
    return np.argmax(logits)
```

Greedy decoding is deterministic—same prompt always produces identical output. It's appropriate for tasks requiring factual accuracy (translation, summarization) but produces repetitive, boring text for creative generation.

**Sampling**: Randomly select from the probability distribution:

```python
def sample_random(logits):
    """Sample from probability distribution."""
    probs = softmax(logits)
    return np.random.choice(len(probs), p=probs)
```

Sampling introduces randomness—same prompt can produce different outputs. This enables creative, diverse generation but risks incoherent or unlikely words.

**Temperature Scaling**. Control sampling randomness by scaling logits:

```python
def sample_with_temperature(logits, temperature=1.0):
    """
    Sample with temperature scaling.
    
    Args:
        logits: [vocab_size] raw model outputs
        temperature: float > 0
            - temperature = 1.0: unchanged distribution
            - temperature < 1.0: sharper distribution (more confident)
            - temperature > 1.0: flatter distribution (more random)
    
    Returns:
        token_id: sampled token
    """
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits)
    return np.random.choice(len(probs), p=probs)
```

Temperature affects distribution sharpness:

```
Original logits: [2.0, 1.0, 0.5]
Probabilities: [0.66, 0.24, 0.10]

Temperature 0.5 (confident):
  Scaled logits: [4.0, 2.0, 1.0]
  Probabilities: [0.84, 0.14, 0.02]  # Heavily favors best token

Temperature 2.0 (random):
  Scaled logits: [1.0, 0.5, 0.25]
  Probabilities: [0.52, 0.28, 0.20]  # More uniform distribution
```

Low temperature (0.5-0.8): Coherent but potentially repetitive. High temperature (1.2-1.5): Creative but sometimes incoherent. Temperature 1.0: Unmodified model distribution.

**Top-k Sampling**. Restrict sampling to the k most probable tokens:

```python
def sample_top_k(logits, k=50):
    """
    Sample from top-k most probable tokens.
    
    Args:
        logits: [vocab_size]
        k: number of top tokens to consider
    
    Returns:
        token_id: sampled token from top-k
    """
    # Get top k indices
    top_k_indices = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_indices]
    
    # Compute probabilities over top-k only
    top_k_probs = softmax(top_k_logits)
    
    # Sample from top-k
    sampled_idx = np.random.choice(k, p=top_k_probs)
    return top_k_indices[sampled_idx]
```

Top-k prevents sampling very low-probability tokens that might be nonsensical. Typical values: k=40-50 for diverse output, k=10 for more focused output.

**Nucleus (Top-p) Sampling**. Sample from the smallest set of tokens whose cumulative probability exceeds threshold p:

```python
def sample_nucleus(logits, p=0.9):
    """
    Nucleus (top-p) sampling.
    
    Args:
        logits: [vocab_size]
        p: cumulative probability threshold (0 < p < 1)
    
    Returns:
        token_id: sampled token
    """
    # Sort by probability (descending)
    probs = softmax(logits)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Compute cumulative probabilities
    cumsum_probs = np.cumsum(sorted_probs)
    
    # Find cutoff: smallest set with cumsum >= p
    cutoff_idx = np.argmax(cumsum_probs >= p) + 1
    
    # Sample from nucleus
    nucleus_indices = sorted_indices[:cutoff_idx]
    nucleus_probs = sorted_probs[:cutoff_idx]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()  # Renormalize
    
    sampled_idx = np.random.choice(len(nucleus_indices), p=nucleus_probs)
    return nucleus_indices[sampled_idx]
```

Nucleus sampling adapts to distribution shape: when the model is confident (one token has high probability), the nucleus is small. When uncertain (many tokens have similar probabilities), the nucleus is large. Typical values: p=0.9-0.95.

**Combining Strategies**. Production systems often combine techniques:

```python
def sample_combined(logits, temperature=0.8, top_k=50, top_p=0.9):
    """
    Combined sampling: temperature + top-k + nucleus.
    
    Order of operations:
    1. Apply temperature scaling
    2. Filter to top-k tokens
    3. Apply nucleus sampling
    """
    # Temperature scaling
    scaled_logits = logits / temperature
    
    # Top-k filtering
    top_k_indices = np.argsort(scaled_logits)[-top_k:]
    filtered_logits = np.full_like(scaled_logits, -np.inf)
    filtered_logits[top_k_indices] = scaled_logits[top_k_indices]
    
    # Nucleus sampling on filtered distribution
    return sample_nucleus(filtered_logits, p=top_p)
```

This provides fine-grained control: temperature adjusts overall randomness, top-k eliminates tail, nucleus adapts to confidence.

**Implementation in Python**. Complete generation function:

```python
# generation.py
def generate(prompt_tokens, embedding_table, block_weights, 
             final_gamma, final_beta, max_new_tokens=20, 
             temperature=1.0, top_k=None, top_p=None):
    """
    Generate text autoregressively from a prompt.
    
    Args:
        prompt_tokens: np.array of int32, shape [prompt_len]
        embedding_table: np.array of float32, shape [vocab_size, d_model]
        block_weights: list of weight arrays for transformer blocks
        final_gamma: np.array of float32, shape [d_model]
        final_beta: np.array of float32, shape [d_model]
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature (default 1.0)
        top_k: if set, only sample from top k tokens
        top_p: if set, nucleus sampling threshold
    
    Returns:
        np.array of int32, shape [prompt_len + max_new_tokens]
    """
    tokens = list(prompt_tokens)
    
    for _ in range(max_new_tokens):
        # Forward pass
        current_tokens = np.array(tokens, dtype=np.int32)
        logits = gpt_forward(current_tokens, embedding_table, 
                             block_weights, final_gamma, final_beta)
        
        # Extract last position
        last_logits = logits[-1, :]
        
        # Sample next token
        if temperature != 1.0:
            last_logits = last_logits / temperature
        
        if top_k is not None:
            next_token = sample_top_k(last_logits, k=top_k)
        elif top_p is not None:
            next_token = sample_nucleus(last_logits, p=top_p)
        else:
            probs = softmax(last_logits)
            next_token = np.random.choice(len(probs), p=probs)
        
        tokens.append(int(next_token))
    
    return np.array(tokens, dtype=np.int32)
```

Users call this function with their desired sampling strategy, and it handles the generation loop.

**Example Usage**:

```python
# Initialize model
vocab_size, d_model, num_blocks = 256, 64, 2
embedding_table, block_weights, final_gamma, final_beta = initialize_random_model()

# Prompt: "Hello"
prompt = "Hello"
prompt_tokens = np.array([ord(c) for c in prompt], dtype=np.int32)

# Generate with different strategies
print("Greedy:")
tokens_greedy = generate(prompt_tokens, embedding_table, block_weights, 
                         final_gamma, final_beta, temperature=0.1)
print(''.join([chr(t) for t in tokens_greedy]))

print("\nCreative (high temp):")
tokens_creative = generate(prompt_tokens, embedding_table, block_weights, 
                           final_gamma, final_beta, temperature=1.5)
print(''.join([chr(t) for t in tokens_creative]))

print("\nTop-k:")
tokens_topk = generate(prompt_tokens, embedding_table, block_weights, 
                       final_gamma, final_beta, temperature=0.8, top_k=40)
print(''.join([chr(t) for t in tokens_topk]))
```

With random weights, outputs are gibberish—but the generation mechanism works correctly. With trained weights (GPT-2, LLaMA), outputs are coherent text.

## 13.10 End-to-End Text Generation Demo

Theory and implementation details matter, but seeing the system work end-to-end cements understanding. This section demonstrates complete text generation: initializing a model, providing prompts, generating continuations with different sampling strategies, and analyzing outputs. While we use random weights (untrained model), the architecture and generation process are identical to production systems.

**Model Initialization**. Create a minimal GPT with byte-level tokenization:

```python
# demo.py
import numpy as np
from generation import generate

def initialize_model(vocab_size=256, d_model=64, num_blocks=2, seed=42):
    """
    Initialize GPT model with random weights.
    
    Args:
        vocab_size: size of vocabulary (256 for byte-level)
        d_model: model dimension
        num_blocks: number of transformer blocks
        seed: random seed for reproducibility
    
    Returns:
        embedding_table, block_weights, final_gamma, final_beta
    """
    np.random.seed(seed)
    d_ff = d_model * 4
    
    # Embedding table
    embedding_table = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02
    
    # Transformer blocks
    block_weights = []
    for _ in range(num_blocks):
        block = {
            # Attention weights
            'wq': np.random.randn(d_model, d_model).astype(np.float32) * 0.02,
            'wk': np.random.randn(d_model, d_model).astype(np.float32) * 0.02,
            'wv': np.random.randn(d_model, d_model).astype(np.float32) * 0.02,
            'wo': np.random.randn(d_model, d_model).astype(np.float32) * 0.02,
            # FFN weights
            'w1': np.random.randn(d_model, d_ff).astype(np.float32) * 0.02,
            'b1': np.zeros(d_ff, dtype=np.float32),
            'w2': np.random.randn(d_ff, d_model).astype(np.float32) * 0.02,
            'b2': np.zeros(d_model, dtype=np.float32),
            # Layer norm parameters
            'ln1_gamma': np.ones(d_model, dtype=np.float32),
            'ln1_beta': np.zeros(d_model, dtype=np.float32),
            'ln2_gamma': np.ones(d_model, dtype=np.float32),
            'ln2_beta': np.zeros(d_model, dtype=np.float32),
        }
        block_weights.append(block)
    
    # Final layer norm
    final_gamma = np.ones(d_model, dtype=np.float32)
    final_beta = np.zeros(d_model, dtype=np.float32)
    
    return embedding_table, block_weights, final_gamma, final_beta

# Initialize model
embedding_table, block_weights, final_gamma, final_beta = initialize_model()
print(f"Model initialized:")
print(f"  Embedding table: {embedding_table.shape}")
print(f"  Transformer blocks: {len(block_weights)}")
print(f"  Total parameters: ~{estimate_params(embedding_table, block_weights)}")
```

For `vocab_size=256`, `d_model=64`, `num_blocks=2`, total parameters ≈ 50K—tiny compared to production models (GPT-2 small: 117M, GPT-3: 175B) but sufficient to demonstrate architecture.

**Text Generation Examples**:

```python
def text_to_tokens(text):
    """Convert text to byte-level token IDs."""
    return np.array([ord(c) for c in text], dtype=np.int32)

def tokens_to_text(tokens):
    """Convert token IDs to text."""
    return ''.join([chr(t) for t in tokens if 0 <= t < 128])  # ASCII only

# Example 1: Greedy decoding (deterministic)
print("\n=== Greedy Decoding ===")
prompt = "The quick brown fox"
prompt_tokens = text_to_tokens(prompt)

generated = generate(
    prompt_tokens, embedding_table, block_weights, 
    final_gamma, final_beta,
    max_new_tokens=30,
    temperature=0.1  # Near-greedy
)

print(f"Prompt: {prompt}")
print(f"Generated: {tokens_to_text(generated)}")

# Example 2: Creative sampling (high temperature)
print("\n=== Creative Sampling ===")
generated = generate(
    prompt_tokens, embedding_table, block_weights, 
    final_gamma, final_beta,
    max_new_tokens=30,
    temperature=1.5  # More random
)

print(f"Prompt: {prompt}")
print(f"Generated: {tokens_to_text(generated)}")

# Example 3: Nucleus sampling (adaptive)
print("\n=== Nucleus Sampling ===")
generated = generate(
    prompt_tokens, embedding_table, block_weights, 
    final_gamma, final_beta,
    max_new_tokens=30,
    temperature=0.8,
    top_p=0.9
)

print(f"Prompt: {prompt}")
print(f"Generated: {tokens_to_text(generated)}")
```

**Expected Output** (with random weights):

```
=== Greedy Decoding ===
Prompt: The quick brown fox
Generated: The quick brown fox█▓▒░▓▒░▓█▒░▓...
(deterministic but nonsensical)

=== Creative Sampling ===
Prompt: The quick brown fox
Generated: The quick brown foxЖ♠◄►☺☻♥♦♣...
(random and incoherent)

=== Nucleus Sampling ===
Prompt: The quick brown fox
Generated: The quick brown fox jump over lazy...
(random but slightly more structured)
```

Random weights produce gibberish because the model hasn't learned language patterns. With trained weights (loaded from GPT-2, for instance), outputs would be coherent English continuations.

**Comparing Sampling Strategies**. Generate multiple samples with different settings:

```python
print("\n=== Comparing Sampling Strategies ===")
strategies = [
    ("Greedy", {"temperature": 0.1}),
    ("Moderate", {"temperature": 0.8}),
    ("Creative", {"temperature": 1.5}),
    ("Top-k=20", {"temperature": 0.8, "top_k": 20}),
    ("Top-k=50", {"temperature": 0.8, "top_k": 50}),
    ("Nucleus p=0.9", {"temperature": 0.8, "top_p": 0.9}),
    ("Nucleus p=0.95", {"temperature": 0.8, "top_p": 0.95}),
]

prompt = "Once upon a time"
prompt_tokens = text_to_tokens(prompt)

for name, kwargs in strategies:
    generated = generate(
        prompt_tokens, embedding_table, block_weights, 
        final_gamma, final_beta,
        max_new_tokens=20,
        **kwargs
    )
    output = tokens_to_text(generated)
    print(f"{name:20s}: {output}")
```

This demonstrates how hyperparameters affect output characteristics—essential knowledge for tuning generation quality in production.

**Prompt Engineering**. Different prompts elicit different behaviors:

```python
prompts = [
    "The capital of France is",      # Factual completion
    "To build a transformer, you",   # Technical explanation
    "Once upon a time,",             # Story beginning
    "def fibonacci(n):",             # Code completion
]

for prompt in prompts:
    tokens = text_to_tokens(prompt)
    generated = generate(
        tokens, embedding_table, block_weights, 
        final_gamma, final_beta,
        max_new_tokens=30,
        temperature=0.8,
        top_p=0.9
    )
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {tokens_to_text(generated)}")
```

Trained models condition strongly on prompts—factual prompts elicit factual continuations, creative prompts elicit stories, code prompts elicit more code. Untrained models don't show this behavior (they haven't learned these patterns), but the generation mechanism handles all prompt types identically.

**Performance Analysis**. Measure generation speed:

```python
import time

prompt_tokens = text_to_tokens("Hello")
warmup_runs = 5
timed_runs = 20

# Warmup
for _ in range(warmup_runs):
    _ = generate(prompt_tokens, embedding_table, block_weights, 
                 final_gamma, final_beta, max_new_tokens=10)

# Timed runs
times = []
for _ in range(timed_runs):
    start = time.perf_counter()
    _ = generate(prompt_tokens, embedding_table, block_weights, 
                 final_gamma, final_beta, max_new_tokens=10)
    elapsed = time.perf_counter() - start
    times.append(elapsed)

mean_time = np.mean(times)
tokens_generated = 10
throughput = tokens_generated / mean_time

print(f"\nGeneration Performance:")
print(f"  Mean time: {mean_time*1000:.2f} ms for {tokens_generated} tokens")
print(f"  Throughput: {throughput:.1f} tokens/sec")
print(f"  Time per token: {mean_time/tokens_generated*1000:.2f} ms")
```

For the minimal model (d_model=64, 2 blocks), expect ~50-100 tokens/sec on a modern CPU. Production models (d_model=1024-12288, 24-96 blocks) are much slower without optimization—Chapter 14 addresses this with KV caching, batch processing, and kernel fusion.

**Interactive Generation**. For a more engaging demo:

```python
def interactive_demo():
    """Interactive text generation demo."""
    print("GPT Text Generator")
    print("==================")
    print("Enter prompts to generate continuations (Ctrl+C to exit)\n")
    
    while True:
        try:
            prompt = input("Prompt: ")
            if not prompt:
                continue
            
            prompt_tokens = text_to_tokens(prompt)
            generated = generate(
                prompt_tokens, embedding_table, block_weights, 
                final_gamma, final_beta,
                max_new_tokens=50,
                temperature=0.8,
                top_p=0.9
            )
            
            output = tokens_to_text(generated)
            print(f"Generated: {output}\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    interactive_demo()
```

This allows experimenting with different prompts and observing generation behavior interactively.

## 13.11 Summary

Chapter 13 constructed complete GPT architecture across two parts. Part 1 built the core forward pass (embeddings, causal masking, stacked blocks, output projection). Part 2 added modern positional encoding (RoPE) and complete text generation (autoregressive loop, sampling strategies, interactive demo).

Key insights from Part 2:

- **Rotary Position Embeddings**: Encode relative positions through geometric rotations, enabling length extrapolation beyond training sequences
- **Autoregressive Generation**: Iterative token production by conditioning on all previous tokens, naturally satisfying causal constraints
- **Sampling Strategies**: Temperature, top-k, and nucleus sampling control generation diversity and quality
- **Production Architecture**: Small-scale implementation mirrors production systems (GPT-2, GPT-3, LLaMA)—same architecture, different scale

The complete GPT implementation demonstrates MLIR's scalability: the same compilation techniques from earlier chapters handle complex models with millions of parameters. Token embeddings lower to memory lookups, RoPE lowers to trigonometric operations, generation composes forward passes with sampling—all compiled through MLIR to efficient native code.

**Current Limitations**. Chapter 13's implementation works correctly but inefficiently:

1. **Recomputation**: Each generation step reprocesses the entire sequence (O(n²) cost)
2. **No Batching**: Processes one sequence at a time (GPUs idle during sequential generation)
3. **Naive Attention**: Full O(seq_len²) attention without optimizations
4. **CPU-Only**: No GPU acceleration for parallel matrix operations

**Looking Ahead**. Chapter 14 addresses these limitations with production optimizations:

- **KV Caching**: Store computed keys and values, avoiding recomputation (O(n) → O(1) per token)
- **FlashAttention**: Fused attention kernel reducing memory bandwidth by orders of magnitude
- **Batched Inference**: Process multiple sequences simultaneously for higher throughput
- **Operator Fusion**: Merge operations (layer norm + matmul, matmul + bias + activation) eliminating intermediate buffers
- **Quantization**: Reduce precision (FP32 → INT8) for faster computation and reduced memory

These techniques transform Chapter 13's functional but slow implementation into production-ready serving infrastructure capable of real-time text generation at scale. The architecture remains identical—optimizations happen transparently through MLIR's compilation pipeline.

Chapter 13 completes the GPT implementation journey: from individual operations (attention, feedforward) to complete architecture (stacked blocks, generation). You now have a working language model—ready for optimization in Chapter 14 and GPU acceleration in Chapter 15.
