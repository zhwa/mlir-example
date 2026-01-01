# Chapter 13: GPT Architecture

Chapter 12 built complete transformer blocks—self-attention, feedforward networks, layer normalization, and residual connections composed into reusable units. A single transformer block processes sequences while maintaining structure, but modern language models like GPT-2, GPT-3, and LLaMA require dozens of these blocks stacked sequentially, plus input/output layers for token processing. This chapter constructs complete GPT (Generative Pre-trained Transformer) architecture by stacking transformer blocks, adding token embeddings, implementing causal masking for autoregressive generation, Rotary Position Embeddings (RoPE), and connecting all components into an end-to-end text generation system.

GPT's architecture follows a straightforward pattern: convert input tokens to embeddings, process through stacked transformer blocks with causal attention, and project hidden states back to vocabulary logits. The key innovation enabling autoregressive text generation is **causal masking**—preventing each token from attending to future tokens during self-attention. This constraint allows the model to generate text one token at a time, conditioning each new token only on previous context. RoPE enhances this by encoding relative positional information through geometric rotations, enabling better length extrapolation compared to earlier positional encoding methods.

The chapter demonstrates MLIR's scalability: the same compilation techniques from Chapters 9-12 (dialect operations, progressive lowering, optimization passes) handle GPT's complexity without architectural changes. We'll see how token embeddings lower to memory lookups, how causal masks integrate into attention, how RoPE applies trigonometric operations, and how parameter management scales to models with millions of parameters. By the end, you'll have a complete GPT implementation capable of generating text—small-scale but architecturally identical to production systems.

## 13.1 GPT Architecture Overview

GPT models predict the next token given previous context, stacking transformer blocks (from Chapter 12) with token embeddings and causal masking. The forward pass has five stages:

```python
def gpt_forward(token_ids, weights):
    """Complete GPT forward pass: tokens → logits"""
    x = embedding_lookup(token_ids, weights['token_embedding'])  # [seq_len, d_model]

    for block_weights in weights['transformer_blocks']:
        x = transformer_block(x, block_weights)  # Process through stacked blocks

    x = layernorm(x, weights['final_gamma'], weights['final_beta'])
    logits = x @ weights['token_embedding'].T  # [seq_len, vocab_size]
    return logits
```

**Key architectural elements**:

- **Token Embeddings**: Maps token IDs to vectors via table lookup `[vocab_size, d_model]`
- **Transformer Blocks**: Stacked attention + FFN layers (12 blocks for GPT-2 small, 96 for GPT-3)
- **Causal Masking**: Attention at position `i` only sees `[0..i]`, enabling autoregressive generation
- **Rotary Position Embeddings (RoPE)**: Geometric rotations encoding relative positions for better length extrapolation
- **Tied Embeddings**: Output projection reuses input embedding table to reduce parameters

**Implementation Architecture**. Chapter 13 inherits Chapter 12's linalg-based lowering patterns for the 8 common transformer operations (LayerNorm, Linear, GELU, Add, Matmul, Transpose, Softmax, Scale). Operations take tensor inputs and return tensor results (functional style), enabling automatic bufferization while maintaining clean, composable IR.

These operations lower to structured `linalg` dialect operations, enabling optimization passes and portable compilation. Chapter 13 adds GPT-specific operations (Embedding, MaskedSoftmax, RoPE) that lower directly to SCF loops for domain-specific logic not expressible in linalg's structured iteration model. This hybrid approach—linalg for regular computations, manual loops for specialized operations—balances optimization opportunities with implementation flexibility.

**Bufferization Pipeline**. After IR is generated, the bufferization pipeline automatically converts functional tensor operations to efficient memref code:

1. **OneShotBufferize**: Converts tensor operations to bufferization dialect
2. **BufferResultsToOutParams**: Transforms function signatures from `(tensor) -> tensor` to `(memref, memref) -> ()`
3. **ConvertBufferizationToMemRef**: Finalizes conversion to memref dialect
4. **Lower to LLVM**: Standard memref/scf/arith to LLVM conversion

This approach provides the best of both worlds: clean tensor IR for operations and composition, efficient memref code for execution.

The common operations work identically in Chapters 11 to 14, providing architectural consistency. Changes to these operations (e.g., adding tiling optimizations or vectorization) propagate automatically across all chapters. The GPT-specific operations remain independent, allowing Chapters 13-14 to implement unique functionality (integer token indexing, conditional masking, pairwise dimension rotation) without constraining Chapter 12's design. This design ensures that LayerNorm and Linear use linalg.ReduceOp, linalg.GenericOp, linalg.MatmulOp, and linalg.FillOp rather than manual SCF loops—providing Transform dialect hooks for advanced optimizations in Chapter 14.

## 13.2 Token Embeddings: From IDs to Vectors

Language models operate on continuous vector spaces, but text consists of discrete tokens (words, subwords, or characters). Token embeddings bridge this gap by mapping each token ID to a learned vector representation. This section implements embedding lookup as an MLIR operation, demonstrates efficient lowering to memory access, and explores the role of embeddings in the model.

**Embedding as Lookup Table**. Conceptually, an embedding layer is a matrix `E` with shape `[vocab_size, d_model]`. Each row corresponds to one token's embedding vector. Given input token IDs `[t_1, t_2, ..., t_n]`, embedding lookup retrieves rows:

```
embedded[i] = E[token_ids[i]]  # Retrieve row token_ids[i] from E
```

For vocabulary size `V = 50,000` and `d_model = 768`, the embedding table contains 38.4 million parameters. During training, these embeddings learn to capture semantic relationships: similar words get similar embeddings, enabling the model to generalize across related tokens.

**Operation Definition**. We define an embedding operation in the Transformer dialect:

```tablegen
// inc/TransformerOps.td
def Transformer_EmbeddingOp : Transformer_Op<"embedding"> {
  let summary = "Token embedding lookup";
  let description = [{
    Performs embedding lookup for token IDs.

    For each token ID in the input sequence, retrieves the corresponding
    embedding vector from the embedding table.

    Example:
    ```mlir
    %result = transformer.embedding %indices, %table
      : tensor<?xi32>, tensor<?x?xf32> -> tensor<?x?xf32>
    ```

    Indices shape: [seq_len] (int32)
    Table shape: [vocab_size, d_model] (float32)
    Output shape: [seq_len, d_model] (float32)

    Equivalent to:
    for i in 0..seq_len:
      token_id = indices[i]
      for j in 0..d_model:
        output[i][j] = table[token_id][j]
  }];

  let arguments = (ins 
    AnyTensor:$indices,  // Token IDs [seq_len]
    AnyTensor:$table     // Embedding table [vocab_size, d_model]
  );

  let results = (outs AnyTensor:$result);  // Output embeddings [seq_len, d_model]

  let assemblyFormat = [{
    $indices `,` $table
    attr-dict `:` type($indices) `,` type($table) `->` type($result)
  }];
}
```

**Lowering to Standard Dialects**. The embedding lowering pattern (from `src/TransformerPasses.cpp`) converts `transformer.embedding` to nested loops that perform indexed lookups:

**Step 1: Extract input shapes and create output tensor**
```cpp
int64_t seqLen = indicesType.getShape()[0];    // e.g., 10 tokens
int64_t dModel = tableType.getShape()[1];      // e.g., 768 dims
auto resultType = RankedTensorType::get({seqLen, dModel}, f32);
Value empty = rewriter.create<tensor::EmptyOp>(loc, resultType);
```

**Step 2: Generate outer loop over sequence positions**
```cpp
Value result = rewriter.create<scf::ForOp>(loc, 0, seqLen, 1, ValueRange{empty},
  [&](OpBuilder &b, Location loc, Value i, ValueRange iterArgs) {
    // Extract token ID at position i
    Value tokenId32 = b.create<tensor::ExtractOp>(loc, indices, ValueRange{i});
    Value tokenIdx = b.create<arith::IndexCastOp>(loc, indexType, tokenId32);
    // ... (inner loop copies embedding vector)
```

**Step 3: Inner loop copies the d_model-dimensional embedding**
```cpp
    // For each dimension j in [0, d_model):
    //   output[i, j] = table[tokenIdx, j]
    Value updatedTensor = b.create<scf::ForOp>(loc, 0, dModel, 1, ...
      [&](OpBuilder &b2, Location loc, Value j, ValueRange args2) {
        Value embVal = b2.create<tensor::ExtractOp>(loc, table, {tokenIdx, j});
        Value updated = b2.create<tensor::InsertOp>(loc, embVal, tensor, {i, j});
        return updated;
```

The lowering generates **nested SCF loops on tensors**: outer loop iterates sequence positions, inner loop copies each embedding dimension. We use manual loops rather than linalg because embedding lookup requires random access via token IDs (irregular indexing pattern).

**Bufferization Pipeline**. After lowering, the bufferization pipeline automatically converts tensor operations to efficient memref code:

```cpp
// bindings.cpp - Bufferization pipeline
bufferization::OneShotBufferizePassOptions bufferizeOptions;
bufferizeOptions.bufferizeFunctionBoundaries = true;
pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));
pm.addPass(createConvertTensorToLinalgPass());
pm.addPass(bufferization::createBufferResultsToOutParamsPass());
pm.addPass(createConvertBufferizationToMemRefPass());
```

This transforms:
1. `tensor.empty` → `memref.alloc`
2. `tensor.extract/insert` → `memref.load/store`
3. Function signatures: `(tensor params) -> tensor` becomes `(memref params, memref out) -> ()`
4. Scf.for tensor loop-carried values → memref in-place updates

The result is efficient memref code with nested loops and direct memory access, identical to hand-written memref lowering but with cleaner tensor-level IR.

**Implementation Challenges**. Several subtleties arise:

1. **Integer Indexing**: Token IDs are integers, but MLIR's indexing operations (`tensor.extract`) require `index` type. The lowering must insert `arith.index_cast` operations to convert `int32 → index`.

2. **Bounds Checking**: What if `token_ids[i] >= vocab_size`? Invalid indices cause undefined behavior (reading beyond array bounds). Production implementations either:
   - Assert valid indices (panic on violation)
   - Clamp indices to valid range (silent correction)
   - Return zero vectors for invalid indices (robustness)

   Chapter 13's implementation assumes valid inputs (typical for trained models with proper tokenization).

3. **Memory Layout**: The embedding table is row-major: embedding for token `t` occupies contiguous memory `E[t, 0], E[t, 1], ..., E[t, d_model-1]`. Accessing `E[t, :]` has good cache locality (sequential access), making embedding lookup memory-bound but efficient.

## 13.3 Causal Masking: Preventing Future Attention

Autoregressive language models predict token `t_i` using only `[t_1, ..., t_{i-1}]`. **Causal masking** enforces this by zeroing attention weights where `j > i`, creating a lower-triangular attention pattern:

```
Bidirectional (BERT):          Causal (GPT):
   k0  k1  k2  k3                 k0  k1  k2  k3
q0 [w00 w01 w02 w03]           q0 [w00  0   0   0 ]
q1 [w10 w11 w12 w13]    →      q1 [w10 w11  0   0 ]
q2 [w20 w21 w22 w23]           q2 [w20 w21 w22  0 ]
q3 [w30 w31 w32 w33]           q3 [w30 w31 w32 w33]
```

**Implementing Causal Masking**. The standard approach applies a mask before softmax:

```python
def causal_attention(Q, K, V):
    """Scaled dot-product attention with causal masking."""
    d_k = Q.shape[-1]

    # Compute similarity scores
    scores = (Q @ K.T) / np.sqrt(d_k)  # [seq_len, seq_len]

    # Create causal mask (upper triangle = -inf)
    seq_len = scores.shape[0]
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -np.inf
    # mask = [[0, -inf, -inf, -inf],
    #         [0,   0,  -inf, -inf],
    #         [0,   0,    0,  -inf],
    #         [0,   0,    0,    0]]

    # Apply mask: set future positions to -inf
    masked_scores = scores + mask

    # Softmax converts -inf to 0
    attention_weights = softmax(masked_scores)  # Future positions have weight 0

    # Weighted sum of values
    output = attention_weights @ V

    return output
```

The mask uses `-inf` (negative infinity) for future positions. When computing `softmax(scores)`, the `-inf` values become `exp(-inf) = 0`, zeroing their contribution. This is numerically stable: softmax's normalization handles the infinities without overflow.

**Operation Definition for Masked Softmax**. We extend the Transformer dialect with a masked softmax operation:

```tablegen
// inc/TransformerOps.td
def Transformer_MaskedSoftmaxOp : Transformer_Op<"masked_softmax"> {
  let summary = "Softmax with causal mask applied";
  let description = [{
    Applies softmax along the last dimension with an additive mask.
    The mask is added to logits before exponential:

    masked_logits = logits + mask
    softmax(masked_logits)[i] = exp(masked_logits[i]) / sum(exp(masked_logits))

    For causal attention:
    - mask[i][j] = 0.0 for allowed positions
    - mask[i][j] = -inf for masked positions
    - After softmax, masked positions have probability 0

    Broadcasting: mask shape [seq_len, seq_len] broadcasts to logits [batch, seq_len, seq_len]

    Example:
    ```mlir
    %result = transformer.masked_softmax %logits, %mask
      : tensor<2x4x4xf32>, tensor<4x4xf32> -> tensor<2x4x4xf32>
    ```
  }];

  let arguments = (ins AnyTensor:$input, AnyTensor:$mask);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $mask
    attr-dict `:` type($input) `,` type($mask) `->` type($result)
  }];
}
```

Unlike the conceptual description earlier, the actual operation takes the mask as an explicit parameter. This design separates mask creation (via `create_causal_mask`) from application, allowing flexibility: the same mask can be reused across multiple attention heads or layers.

**Lowering Masked Softmax**. The lowering applies the mask and computes softmax in four steps using nested SCF loops:

**Step 1: Add mask to logits** (masking future positions)
```cpp
// masked_logits[i,j] = logits[i,j] + mask[i,j]
// where mask[i,j] = 0 if j <= i, else -inf
Value maskedLogits = rewriter.create<scf::ForOp>(loc, ...,
  [&](OpBuilder &b, Location loc, Value i, Value j, ValueRange args) {
    Value logit = b.create<tensor::ExtractOp>(loc, input, {i, j});
    Value maskVal = b.create<tensor::ExtractOp>(loc, mask, {i, j});
    Value masked = b.create<arith::AddFOp>(loc, logit, maskVal);
    // insert back into tensor
```

**Step 2: Find max value per row** (numerical stability)
```cpp
// max_val = max(masked_logits[i, :])
Value maxVal = rewriter.create<scf::ForOp>(loc, 0, seqLen, 1, init_negInf,
  [&](OpBuilder &b, Location loc, Value j, ValueRange args) {
    Value current = b.create<tensor::ExtractOp>(loc, maskedLogits, {i, j});
    Value prevMax = args[0];
    Value newMax = b.create<arith::MaximumFOp>(loc, current, prevMax);
    return newMax;
```

**Step 3: Compute exp and sum**
```cpp
// exp_sum = sum(exp(masked_logits[i,j] - max_val))
Value expSum = rewriter.create<scf::ForOp>(loc, 0, seqLen, 1, init_zero,
  [&](OpBuilder &b, Location loc, Value j, ValueRange args) {
    Value val = b.create<tensor::ExtractOp>(loc, maskedLogits, {i, j});
    Value shifted = b.create<arith::SubFOp>(loc, val, maxVal);
    Value expVal = b.create<math::ExpOp>(loc, shifted);
    Value sum = b.create<arith::AddFOp>(loc, expVal, args[0]);
    return sum;
```

**Step 4: Normalize by sum**
```cpp
// output[i,j] = exp(masked_logits[i,j] - max_val) / exp_sum
Value normalized = b.create<arith::DivFOp>(loc, expVal, expSum);
Value result = b.create<tensor::InsertOp>(loc, normalized, output, {i, j});
```

After bufferization, these tensor operations become efficient memref operations with in-place updates. The mask is applied during the element-wise addition without requiring separate memory allocation.

**Memory and Computation Trade-offs**. Generating the mask on-the-fly saves memory bandwidth by avoiding storage and retrieval of a `seq_len × seq_len` matrix. While a 2048×2048 float32 mask would require 16MB (our implementation uses float32 for simplicity), the comparison adds negligible overhead—typically hidden by instruction pipelining in the attention mechanism's dot products. Modern production systems often use boolean (i1) or 8-bit integer masks (reducing to 4MB or 512KB), but the bandwidth savings remain the key benefit. In MLIR, generating masks on-the-fly enables better kernel fusion through the linalg dialect, keeping mask logic in registers and reducing DRAM traffic.

Advanced implementations use **block-sparse attention** techniques that avoid materializing the full attention matrix, computing masked attention in smaller blocks with improved memory efficiency. Production systems like vLLM and SGLang use optimized attention kernels (FlashAttention, Flash-Decoding) for this purpose.

**Causal Masking in Multi-Head Attention**. Chapter 12's multi-head attention splits Q, K, V into multiple heads. Causal masking applies identically to each head—all heads see the same causal structure:

```python
def multi_head_causal_attention(x, W_q, W_k, W_v, W_o, num_heads):
    # Project and split into heads
    Q = split_heads(x @ W_q, num_heads)  # [num_heads, seq_len, d_k]
    K = split_heads(x @ W_k, num_heads)
    V = split_heads(x @ W_v, num_heads)

    # Apply causal attention to each head
    head_outputs = []
    for h in range(num_heads):
        head_out = causal_attention(Q[h], K[h], V[h])  # Mask applied here
        head_outputs.append(head_out)

    # Concatenate and project
    output = concatenate(head_outputs) @ W_o
    return output
```

The mask structure is identical across heads—only the learned projections (W_q, W_k, W_v) differ per head.

Causal masking is the key ingredient enabling autoregressive generation. With this mechanism in place, we can stack transformer blocks and build the complete GPT forward pass.

## 13.4 Stacking Transformer Blocks

GPT's depth comes from stacking multiple transformer blocks sequentially. Each block processes the entire sequence, transforming representations through self-attention and feedforward networks. This section demonstrates how to compose blocks in MLIR, manage parameters across layers, and understand information flow through deep transformer stacks.

**Sequential Block Composition**. A GPT model with `num_blocks` transformer blocks processes input sequentially:

```python
def gpt_forward(embeddings, block_weights, num_blocks):
    """
    Process embeddings through stacked transformer blocks.

    Args:
        embeddings: [seq_len, d_model] token embeddings
        block_weights: list of length num_blocks, each containing block parameters
        num_blocks: number of transformer blocks

    Returns:
        hidden_states: [seq_len, d_model] after all blocks
    """
    x = embeddings

    for i in range(num_blocks):
        x = transformer_block(x, block_weights[i])

    return x
```

Each block's output becomes the next block's input. The model dimension `d_model` remains constant throughout—every block expects input `[seq_len, d_model]` and produces output with the same shape.

**Parameter Organization**. Each transformer block requires ~3.15M parameters (see Chapter 12.4's parameter calculation). For a 12-block GPT:

```python
# Parameters for one transformer block
block_params = {
    'ln1_gamma': np.ones(d_model),           # [d_model]
    'ln1_beta': np.zeros(d_model),           # [d_model]
    'attn_wq': np.random.randn(d_model, d_model) * 0.02,  # [d_model, d_model]
    'attn_wk': np.random.randn(d_model, d_model) * 0.02,  # [d_model, d_model]
    'attn_wv': np.random.randn(d_model, d_model) * 0.02,  # [d_model, d_model]
    'attn_wo': np.random.randn(d_model, d_model) * 0.02,  # [d_model, d_model]
    'ln2_gamma': np.ones(d_model),           # [d_model]
    'ln2_beta': np.zeros(d_model),           # [d_model]
    'ffn_w1': np.random.randn(d_ff, d_model) * 0.02,      # [d_ff, d_model] (PyTorch format)
    'ffn_b1': np.zeros(d_ff),                # [d_ff]
    'ffn_w2': np.random.randn(d_model, d_ff) * 0.02,      # [d_model, d_ff] (PyTorch format)
    'ffn_b2': np.zeros(d_model),             # [d_model]
}

# Create separate parameters for each block
all_block_weights = [
    {key: val.copy() for key, val in block_params.items()}
    for _ in range(num_blocks)
]
```

**Implementing the Stack - Python API Composition**. Like Chapter 12's `ffn()` and `transformer_block()`, the multi-layer transformer stack is implemented as a **Python API composition function** rather than a dialect operation. This follows the architectural pattern established in Chapter 12: primitive operations in the dialect, higher-level compositions in the bindings.

```cpp
// src/bindings.cpp - Multi-layer transformer composition
Tensor gpt_forward(py::array_t<int32_t> indices,
                   py::array_t<float> embedding_table,
                   const std::vector<py::array_t<float>>& all_weights,
                   py::array_t<float> final_gamma,
                   py::array_t<float> final_beta) {
  if (all_weights.size() % 16 != 0) {
    throw std::runtime_error("Expected 16 weights per layer");
  }

  int num_layers = all_weights.size() / 16;

  // Token embeddings
  Tensor hidden = embedding(indices, embedding_table);

  // Apply transformer blocks
  for (int layer = 0; layer < num_layers; layer++) {
    int base = layer * 16;
    hidden = gpt_block(
        hidden,
        all_weights[base + 0],  all_weights[base + 1],   // Q weight, bias
        all_weights[base + 2],  all_weights[base + 3],   // K weight, bias
        all_weights[base + 4],  all_weights[base + 5],   // V weight, bias
        all_weights[base + 6],  all_weights[base + 7],   // O weight, bias
        all_weights[base + 8],  all_weights[base + 9],   // FFN W1, b1
        all_weights[base + 10], all_weights[base + 11],  // FFN W2, b2
        all_weights[base + 12], all_weights[base + 13],  // LN1 gamma, beta
        all_weights[base + 14], all_weights[base + 15]   // LN2 gamma, beta
    );
  }

  // Final layer norm
  return layer_norm(hidden, final_gamma, final_beta);
}
```

**Note**: Chapter 13 includes biases for attention projections (16 weights total), whereas Chapter 12 omitted attention biases for simplicity. This architectural choice adds ~8K parameters per block but can improve model quality.

**Why Composition Functions Instead of Dialect Operations?** As discussed in Chapter 12, this design offers several advantages:

1. **Flexibility**: The Python layer can easily adjust control flow (number of layers, conditional blocks) without modifying the dialect
2. **Simplicity**: The dialect remains focused on primitive operations that benefit from custom lowering
3. **Optimization**: MLIR's optimizer sees the expanded graph of primitive operations, enabling cross-operation optimization

The `gpt_forward()` function builds a computation graph by calling primitive operations (`embedding`, `gpt_block`, `layer_norm`), then compiles and executes the entire graph as a single MLIR module.

**Memory Consumption**. During forward pass, we must store activations for each block (needed for gradient computation during training). For a model with:
- Batch size: 8
- Sequence length: 512
- Model dimension: 768
- Number of blocks: 12

Activation memory per block:
```
Input/output: batch × seq_len × d_model × 4 bytes = 8 × 512 × 768 × 4 ≈ 12 MB
Attention scores: batch × num_heads × seq_len × seq_len × 4 = 8 × 12 × 512 × 512 × 4 ≈ 100 MB
FFN intermediate: batch × seq_len × d_ff × 4 = 8 × 512 × 3072 × 4 ≈ 50 MB
──────────────────────────────────────────────────────────────────────────────────
Total per block: ~162 MB
Total for 12 blocks: ~1.9 GB
```

This explains why large-batch training requires substantial GPU memory. Techniques like gradient checkpointing trade computation for memory by recomputing activations during backward pass rather than storing them.

## 13.5 Rotary Position Embeddings (RoPE)

**Rotary Position Embeddings (RoPE)** encode positions by rotating query and key vectors before attention. For position `m` at dimension pair `(2d, 2d+1)`, RoPE applies rotation:

```
θ_d = 10000^(-2d/d_model)
angle = m · θ_d

rotated[2d]   = cos(angle) · x[2d] - sin(angle) · x[2d+1]
rotated[2d+1] = sin(angle) · x[2d] + cos(angle) · x[2d+1]
```

**Key property**: After rotation, `Q[m]^T @ K[n] = Q[m]^T @ R_{n-m} @ K[n]`, making attention scores depend on **relative position** `n-m` rather than absolute positions. This enables better length extrapolation compared to learned or sinusoidal embeddings.

**MLIR Operation Definition**. We define a RoPE operation in the Transformer dialect:

```tablegen
// inc/TransformerOps.td
def Transformer_RoPEOp : Transformer_Op<"rope"> {
  let summary = "Rotary position embedding";
  let description = [{
    Applies rotary position embeddings to query or key vectors.
    Positions are implicit: 0, 1, 2, ..., seq_len-1.

    For each position i and dimension pair (2j, 2j+1):
      θ_j = 10000^(-2j/d_model)
      angle = i * θ_j

      output[i, 2j]   = cos(angle) * input[i, 2j] - sin(angle) * input[i, 2j+1]
      output[i, 2j+1] = sin(angle) * input[i, 2j] + cos(angle) * input[i, 2j+1]

    This encodes relative positional information into attention scores.

    Example:
    ```mlir
    %result = transformer.rope %input : tensor<32x64xf32> -> tensor<32x64xf32>
    ```

    References:
    - RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
    - Used in LLaMA, GPT-NeoX, and other modern LLMs
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}
```

**Lowering RoPE to Standard Dialects**. The lowering generates tensor operations with nested SCF loops computing rotations in four stages:

**Step 1: Set up outer loop over sequence positions**
```cpp
// Outer loop: iterate positions 0..seq_len-1 (pos = implicit position)
Value result = rewriter.create<scf::ForOp>(
  loc, zeroIdx, seqLenVal, oneIdx, ValueRange{empty},
  [&](OpBuilder &b, Location loc, Value pos, ValueRange iterArgs) {
    Value posFloat = b.create<arith::SIToFPOp>(loc, b.getF32Type(), 
                       b.create<arith::IndexCastOp>(loc, b.getI64Type(), pos));
```

**Step 2: Inner loop processes dimension pairs with stride 2**
```cpp
    // Process pairs: (0,1), (2,3), (4,5), ...
    Value updatedTensor = b.create<scf::ForOp>(
      loc, zeroIdx, dModelVal, twoIdx, ValueRange{currentTensor},
      [&](OpBuilder &b2, Location loc, Value dimIdx, ValueRange args2) {
        // dimIdx steps by 2, processing consecutive pairs
```

**Step 3: Compute rotation angle θ and trigonometric values**
```cpp
        // θ_j = 10000^(-2j/d_model) where j = dimIdx / 2
        Value j_i64 = b2.create<arith::DivSIOp>(loc, dimIdxI64, two_i64);  // Integer division!
        Value j_float = b2.create<arith::SIToFPOp>(loc, b2.getF32Type(), j_i64);

        Value twoJ = b2.create<arith::MulFOp>(loc, two_f32, j_float);
        Value ratio = b2.create<arith::DivFOp>(loc, twoJ, dModelFloat);

        // base^(-ratio) = exp(-ratio * log(base))
        Value theta = b2.create<math::ExpOp>(loc, 
          b2.create<arith::MulFOp>(loc, 
            b2.create<arith::NegFOp>(loc, ratio), 
            b2.create<math::LogOp>(loc, base)));

        Value angle = b2.create<arith::MulFOp>(loc, posFloat, theta);
        Value cosAngle = b2.create<math::CosOp>(loc, angle);
        Value sinAngle = b2.create<math::SinOp>(loc, angle);
```

**Step 4: Apply 2D rotation to dimension pair**
```cpp
        // Extract pair (x0, x1) from input
        Value x0 = b2.create<tensor::ExtractOp>(loc, input, ValueRange{pos, dimIdx});
        Value x1 = b2.create<tensor::ExtractOp>(loc, input, ValueRange{pos, dimIdxPlus1});

        // Rotation: [cos -sin] [x0]   [x0*cos - x1*sin]
        //           [sin  cos] [x1] = [x0*sin + x1*cos]
        Value out0 = b2.create<arith::SubFOp>(loc, 
          b2.create<arith::MulFOp>(loc, x0, cosAngle),
          b2.create<arith::MulFOp>(loc, x1, sinAngle));

        Value out1 = b2.create<arith::AddFOp>(loc,
          b2.create<arith::MulFOp>(loc, x0, sinAngle),
          b2.create<arith::MulFOp>(loc, x1, cosAngle));

        // Insert rotated pair back into output tensor
        Value tensor1 = b2.create<tensor::InsertOp>(loc, out0, tensor, ValueRange{pos, dimIdx});
        Value tensor2 = b2.create<tensor::InsertOp>(loc, out1, tensor1, ValueRange{pos, dimIdxPlus1});
```

**Critical Implementation Detail**: Computing pair index `j = dimIdx / 2` requires **integer division** (`arith::DivSIOp`), ensuring dims (0,1) get j=0, dims (2,3) get j=1, etc. Using floating-point division would cause numerical errors in theta computation. Trigonometric functions (`cos`, `sin`, `exp`, `log`) compile to LLVM intrinsics mapping to hardware instructions.

**Bufferization Transform**. After lowering, the bufferization pipeline converts `tensor.extract/insert` to `memref.load/store`, `tensor.empty` to `memref.alloc`, and tensor loop-carried values to memref in-place updates. The result is efficient memref code with nested loops and direct memory access.

**Integrating RoPE into Attention**. RoPE modifies the attention computation:

```python
def rope_attention(Q, K, V):
    """
    Attention with rotary position embeddings.
    Positions are implicit: 0, 1, 2, ..., seq_len-1

    Args:
        Q, K, V: [seq_len, d_model] query, key, value matrices

    Returns:
        output: [seq_len, d_model] attention output
    """
    # Apply RoPE to queries and keys (not values!)
    Q_rot = apply_rope_batch(Q)
    K_rot = apply_rope_batch(K)

    # Standard scaled dot-product attention
    d_k = Q.shape[-1]
    scores = (Q_rot @ K_rot.T) / np.sqrt(d_k)

    # Causal mask
    mask = np.triu(np.ones_like(scores), k=1) * -np.inf
    masked_scores = scores + mask

    attention_weights = softmax(masked_scores)
    output = attention_weights @ V  # V is NOT rotated

    return output
```

RoPE only modifies Q and K, leaving V unchanged. This asymmetry is intentional: position information affects attention scores (which positions to attend to) but not value aggregation (what information to extract).

## 13.6 Complete GPT Forward Pass

With all components implemented (embeddings, causal masking, RoPE, transformer blocks), we now assemble the complete GPT forward pass. This section demonstrates end-to-end inference: from input token IDs to output vocabulary logits, ready for next-token prediction or generation.

**Implementation as Composition Function**. The complete forward pass is implemented in the C++ bindings as a composition function that builds a computation graph from primitive operations:

```cpp
// Simplified internal flow
Tensor gpt_forward(...) {
  // Create computation graph nodes
  Tensor hidden = embedding(indices, embedding_table);  // Graph node 1

  for (int layer = 0; layer < num_layers; layer++) {
    hidden = gpt_block(hidden, layer_weights);  // Graph nodes 2, 3, ...
  }

  hidden = layer_norm(hidden, final_gamma, final_beta);  // Final graph node

  // Graph is compiled when forward() is called
  return hidden;  // Returns Tensor handle containing the graph
}
```

The `ch13.forward()` call triggers compilation: the computation graph is converted to MLIR IR, lowered to LLVM, and executed. This lazy compilation approach allows building complex graphs without immediate execution overhead.

**Testing End-to-End Forward Pass**. Validation ensures correctness and stability:

```python
# Model configuration
vocab_size, d_model, num_blocks, seq_len = 256, 64, 2, 8
token_ids = np.array([72, 101, 108, 108, 111, 32, 87, 111], dtype=np.int32)  # "Hello Wo"

# MLIR forward pass vs reference implementation
logits_mlir = gpt_forward(token_ids, embedding_table, block_weights, final_gamma, final_beta)
logits_ref = reference_gpt_forward(token_ids, embedding_table, block_weights, final_gamma, final_beta)

# Verify shape, correctness, and numerical stability
assert logits_mlir.shape == (seq_len, vocab_size)
np.testing.assert_allclose(logits_mlir, logits_ref, rtol=1e-3, atol=1e-5)
assert not (np.isnan(logits_mlir).any() or np.isinf(logits_mlir).any())
```

**Next Token Prediction**. Convert logits to probabilities and sample:

```python
probs = scipy.special.softmax(logits_mlir[-1, :])  # Last position
next_token_id = np.argmax(probs)  # Greedy decoding
```

For random weights, predictions are meaningless. For trained models (GPT-2, GPT-3), the model completes sequences sensibly: "Hello Wo" → "rld".

## 13.7 Autoregressive Text Generation

With the forward pass complete, we implement **generation**—producing text one token at a time by iteratively sampling from the model's output distribution.

**The Generation Loop**. Starting from a prompt, generate tokens iteratively:

```python
def generate(prompt_tokens, model_weights, max_new_tokens=50):
    tokens = list(prompt_tokens)
    for _ in range(max_new_tokens):
        logits = gpt_forward(np.array(tokens), model_weights)  # [current_len, vocab_size]
        next_token = sample(logits[-1, :])  # Sample from last position
        tokens.append(next_token)
    return np.array(tokens, dtype=np.int32)
```

Each iteration runs a forward pass on the full sequence (prompt + generated tokens), extracts logits for the last position, and samples the next token. This reprocessing is inefficient—Chapter 14 optimizes with **KV caching**.

**Sampling Strategies**. Control output diversity through different sampling methods:

**Greedy Decoding**: Always pick the most probable token (`np.argmax(logits)`). Deterministic but repetitive.

**Temperature Scaling**: Scale logits before sampling to control randomness:
```python
scaled_logits = logits / temperature  # temperature < 1.0: more confident, > 1.0: more random
probs = softmax(scaled_logits)
next_token = np.random.choice(len(probs), p=probs)
```
Low temperature (0.5-0.8) produces coherent text, high temperature (1.2-1.5) produces creative but sometimes incoherent text.

**Top-k Sampling**: Restrict sampling to k most probable tokens:
```python
top_k_indices = np.argsort(logits)[-k:]  # Get top k
top_k_probs = softmax(logits[top_k_indices])
sampled_idx = np.random.choice(k, p=top_k_probs)
return top_k_indices[sampled_idx]
```
Prevents sampling very low-probability nonsensical tokens. Typical: k=40-50.

**Nucleus (Top-p) Sampling**: Sample from smallest set with cumulative probability ≥ p:
```python
sorted_probs = np.sort(softmax(logits))[::-1]
cumsum = np.cumsum(sorted_probs)
cutoff = np.argmax(cumsum >= p) + 1  # Find nucleus
# Sample from top cutoff tokens
```
Adapts to model confidence: small nucleus when confident, large when uncertain. Typical: p=0.9-0.95.

**Combined Sampling**: Production systems combine temperature + top-k + nucleus for fine-grained control over output diversity and quality.

## 13.8 Testing Text Generation

The test suite in `test.py` verifies text generation functionality through greedy decoding and sampling. While using random weights (so outputs are random), these tests validate the generation mechanism that production systems use.

**Greedy Generation**. The core generation loop appends tokens autoregressively:

```python
def generate_greedy(prompt_tokens, embedding_table, all_weights, 
                    final_gamma, final_beta, max_new_tokens=20):
    tokens = list(prompt_tokens)
    for _ in range(max_new_tokens):
        current_tokens = np.array(tokens, dtype=np.int32)
        hidden = ch13.forward(ch13.gpt_forward(
            current_tokens, embedding_table, all_weights, final_gamma, final_beta
        ))
        logits = hidden[-1, :] @ embedding_table.T
        next_token = sample(logits / 1e-8, top_k=None)  # Very low temp = greedy
        tokens.append(next_token)
    return np.array(tokens, dtype=np.int32)
```

The test verifies: (1) correct number of tokens generated, (2) prompt preserved in output, (3) deterministic behavior with same seed, and (4) different prompts produce different outputs.

**Sampling Implementation**. The `sample()` function implements softmax with optional top-k filtering:

```python
def sample(logits, top_k=None):
    if top_k is not None:
        indices = np.argpartition(logits, -top_k)[-top_k:]
        top_logits = logits[indices]
        exp_logits = np.exp(top_logits - np.max(top_logits))
    else:
        exp_logits = np.exp(logits - np.max(logits))
    
    probs = exp_logits / np.sum(exp_logits)
    return int(np.random.choice(len(probs), p=probs))
```

Tests validate that sampling produces multiple distinct tokens (not just argmax) and that distributions are correctly normalized.

**Running Generation Tests**. The test suite executes with minimal models for speed:

```bash
python test.py
# Output shows:
# Test 1: Greedy generation
#   ✓ Generated correct number of tokens
#   ✓ Prompt tokens preserved in output
# Test 2: Greedy generation is deterministic
#   ✓ Greedy generation is deterministic
# Test 4: Sampling with softmax
#   ✓ Sampling produces diverse tokens
```

These tests use byte-level tokenization (vocab_size=256) and minimal architectures (d_model=32, num_layers=1) for fast validation. For a production demonstration with all serving optimizations (continuous batching, KV caching, chunked prefill), see Chapter 16's `demo.py`.

## 13.9 Testing and Validation

Building complex systems like GPT requires testing at three levels: **unit tests** (individual operations like embeddings, masking), **integration tests** (composed sub-systems like attention, transformer blocks), and **system tests** (complete forward pass). Each level catches different bug classes: arithmetic errors, composition issues, and emergent problems respectively.

**Representative Test Examples**:

```python
# Unit test: Embedding lookup (exact equality expected)
def test_embedding():
    output = gpt.embedding(token_ids, embedding_table)
    np.testing.assert_array_equal(output, embedding_table[token_ids])

# Integration test: Causal masking prevents future attention
def test_causal_masking():
    attention_weights = gpt.masked_softmax(scores)
    # Verify upper triangle is zero, rows sum to 1
    assert np.allclose(attention_weights.sum(axis=1), 1.0)
    assert np.all(np.triu(attention_weights, k=1) < 1e-6)

# System test: End-to-end correctness
def test_gpt_forward():
    logits = gpt.forward(token_ids, weights)
    assert logits.shape == (seq_len, vocab_size)
    np.testing.assert_allclose(logits, reference_implementation(token_ids, weights), rtol=1e-3)
```

**Debugging Strategy**. When tests fail: (1) isolate the failure by running unit tests first, (2) simplify inputs (seq_len=2, d_model=4) for human-readable outputs, (3) compare against reference implementations at each step, (4) check common issues (NaN/inf from divide-by-zero, shape mismatches from broadcasting, index out-of-bounds). Systematic debugging with data beats guessing.

## 13.10 Summary

Chapter 13 constructed complete GPT architecture: token embeddings, causal masking, stacked transformer blocks, RoPE, and autoregressive text generation. We implemented the full forward pass—from input token IDs to output vocabulary logits—and demonstrated iterative generation with multiple sampling strategies.

Key achievements:

- **Token Embeddings**: Lookup tables mapping discrete tokens to continuous vectors for neural network processing
- **Causal Masking**: Prevents future attention, enabling autoregressive text generation
- **Sequential Composition**: Stacked transformer blocks with residual connections enable deep representations
- **Rotary Position Embeddings**: Geometric rotations encoding relative positions for better length extrapolation
- **Autoregressive Generation**: Iterative token production with various sampling strategies (temperature, top-k, nucleus)
- **Complete Implementation**: Small-scale GPT architecturally identical to production systems (GPT-2, GPT-3, LLaMA)

The implementation demonstrates MLIR's scalability: token embeddings lower to memory lookups, RoPE to trigonometric operations, generation composes forward passes with sampling—all compiled to efficient native code.

**Current Limitations**: Each generation step reprocesses the entire sequence (O(n²) cost), no batching, full O(seq_len²) attention, CPU-only execution.

**Looking Ahead**: Chapter 14 optimizes this working implementation for production through KV caching (O(n²)→O(n)), Transform dialect optimizations, and declarative rewrite rules. Chapter 15 adds GPU acceleration. The architecture remains unchanged—optimizations happen transparently through MLIR's compilation pipeline.