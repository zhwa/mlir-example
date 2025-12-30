# Chapter 13: GPT Architecture (Part 1)

Chapter 12 built complete transformer blocks—self-attention, feedforward networks, layer normalization, and residual connections composed into reusable units. A single transformer block processes sequences while maintaining structure, but modern language models like GPT-2, GPT-3, and LLaMA require dozens of these blocks stacked sequentially, plus input/output layers for token processing. This chapter constructs complete GPT (Generative Pre-trained Transformer) architecture by stacking transformer blocks, adding token embeddings, implementing causal masking for autoregressive generation, and connecting all components into an end-to-end model.

GPT's architecture follows a straightforward pattern: convert input tokens to embeddings, process through stacked transformer blocks, and project hidden states back to vocabulary logits. The key innovation enabling autoregressive text generation is **causal masking**—preventing each token from attending to future tokens during self-attention. This constraint allows the model to generate text one token at a time, conditioning each new token only on previous context. Chapter 13 Part 1 focuses on this core architecture, establishing the foundation for text generation (Part 2 will cover Rotary Position Embeddings, sampling strategies, and generation pipelines).

The chapter demonstrates MLIR's scalability: the same compilation techniques from Chapters 9-12 (dialect operations, progressive lowering, optimization passes) handle GPT's complexity without architectural changes. We'll see how token embeddings lower to memory lookups, how causal masks integrate into attention, and how parameter management scales to models with millions of parameters. By the end of Part 1, you'll have a complete GPT forward pass—ready to compute logits for any input sequence.

## 13.1 GPT Architecture Overview

Before diving into implementation, we must understand GPT's structure: how components connect, what data flows where, and why this architecture enables language modeling. Modern GPTs (GPT-2, GPT-3, LLaMA) share fundamental architecture despite differing in scale—understanding the structure once applies to all variants.

**The Complete GPT Pipeline**. A GPT model processes token sequences through several stages:

```python
def gpt_forward(token_ids, weights):
    """
    Complete GPT forward pass.
    
    Args:
        token_ids: [seq_len] integer token IDs
        weights: model parameters (embeddings, block weights)
    
    Returns:
        logits: [seq_len, vocab_size] next-token predictions
    """
    # Stage 1: Token embedding lookup
    x = embedding_lookup(token_ids, weights['token_embedding'])  # [seq_len, d_model]
    
    # Stage 2: Add positional information (Chapter 13 Part 2 covers RoPE)
    # x = add_positional_encoding(x, positions)
    
    # Stage 3: Process through transformer blocks
    for block_weights in weights['transformer_blocks']:
        x = transformer_block(x, block_weights)  # [seq_len, d_model]
    
    # Stage 4: Final layer normalization
    x = layernorm(x, weights['final_gamma'], weights['final_beta'])  # [seq_len, d_model]
    
    # Stage 5: Project to vocabulary (language modeling head)
    logits = x @ weights['token_embedding'].T  # [seq_len, vocab_size]
    
    return logits
```

This structure is remarkably consistent across GPT variants. GPT-2 small (117M parameters) uses 12 transformer blocks with `d_model=768`; GPT-3 (175B parameters) uses 96 blocks with `d_model=12288`; the architecture remains identical, only scale changes.

**Causal Language Modeling Objective**. GPT models predict the next token given previous tokens:

```
P(token_n | token_1, token_2, ..., token_{n-1})
```

During training, the model sees the entire sequence but must predict each position using only earlier context. This is enforced through **causal masking** in self-attention: position `i` can attend to positions `[0, 1, ..., i]` but not `[i+1, i+2, ..., seq_len-1]`. Without causal masking, the model could "cheat" by looking at the answer (the next token), undermining the prediction task.

During inference (text generation), causal masking enables autoregressive generation: generate token 1, append it, generate token 2 conditioned on token 1, append it, generate token 3 conditioned on tokens 1-2, etc. Each generation step naturally satisfies the causal constraint because future tokens don't exist yet.

**Architecture Components in Detail**:

1. **Token Embeddings**: Convert discrete token IDs to continuous vectors. For vocabulary size `V` and model dimension `d_model`, the embedding table has shape `[V, d_model]`. Each token ID indexes into this table, retrieving its embedding vector.

2. **Transformer Blocks**: The stack of identical transformer blocks (Chapter 12) processes sequences. Each block applies self-attention (mixing information across positions) and feedforward networks (per-position transformation). Block count varies by model: GPT-2 small uses 12, GPT-3 uses 96.

3. **Layer Normalization**: Applied before each sub-layer (pre-norm architecture) and after the final transformer block. Stabilizes activations and improves training dynamics.

4. **Language Modeling Head**: Projects final hidden states back to vocabulary space. Typically uses **tied embeddings**: the output projection shares weights with the input embedding table (`logits = hidden @ embedding_table.T`). This reduces parameters and often improves performance.

**Why This Architecture Works**. GPT's design balances several considerations:

- **Causal Structure**: Autoregressive generation requires that each position depends only on previous positions. Causal masking enforces this during training, ensuring the learned model generalizes to generation.

- **Deep Context**: Stacking many transformer blocks (12, 24, 96 layers) allows the model to build hierarchical representations: lower layers capture syntax and local patterns, higher layers capture semantics and long-range dependencies.

- **Shared Computation**: All positions process in parallel during training (thanks to attention's parallelizability), making training efficient despite sequential generation at inference time.

- **Scalability**: The architecture scales smoothly—add more blocks, increase `d_model`, expand vocabulary. Training dynamics remain similar across scales (modulo learning rate tuning and optimization improvements).

**Parameter Count**. For a GPT model with:
- Vocabulary size `V = 50,257` (GPT-2's byte-pair encoding)
- Model dimension `d_model = 768`
- Block count `num_blocks = 12`
- Feedforward expansion `d_ff = 4 * d_model = 3072`

Parameter breakdown:
```
Token embeddings:    V × d_model = 50,257 × 768        ≈ 38.6M
Transformer blocks:  12 blocks × 3.1M params/block     ≈ 37.2M
Final layer norm:    2 × d_model = 2 × 768             ≈ 1.5K
─────────────────────────────────────────────────────────────
Total:                                                  ≈ 75.8M
```

This roughly matches GPT-2 small's actual parameter count (~117M, which includes positional embeddings not shown here). Most parameters reside in embeddings and transformer blocks—the architecture itself (layer norms, residuals) contributes negligibly.

**Chapter 13 Part 1 Scope**. This chapter implements:
- Token embedding lookup (Section 13.2)
- Causal masking for attention (Section 13.3)
- Stacking transformer blocks (Section 13.4)
- Complete forward pass (Section 13.5)
- Testing and validation (Section 13.6)

Part 2 will cover Rotary Position Embeddings (RoPE), autoregressive generation, sampling strategies, and text generation pipelines.

**Implementation Architecture**. Chapter 13 inherits Chapter 12's linalg-based lowering patterns for the 8 common transformer operations (LayerNorm, Linear, GELU, Add, Matmul, Transpose, Softmax, Scale). However, Chapter 13 adopts a **tensor-first architecture** where operations take tensor inputs and return tensor results (functional style), rather than using memref out-parameters. This modern MLIR best practice enables automatic bufferization while maintaining clean, composable IR.

These operations lower to structured `linalg` dialect operations, enabling optimization passes and portable compilation. Chapter 13 adds 4 GPT-specific operations (Embedding, CreateCausalMask, MaskedSoftmax, RoPE) that also use tensor-first style but lower directly to SCF loops for domain-specific logic not expressible in linalg's structured iteration model. This hybrid approach—linalg for regular computations, manual loops for specialized operations—balances optimization opportunities with implementation flexibility.

**Bufferization Pipeline**. After tensor-based IR is generated, the bufferization pipeline automatically converts functional tensor operations to efficient memref code:

1. **OneShotBufferize**: Converts tensor operations to bufferization dialect
2. **BufferResultsToOutParams**: Transforms function signatures from `(tensor) -> tensor` to `(memref, memref) -> ()`
3. **ConvertBufferizationToMemRef**: Finalizes conversion to memref dialect
4. **Lower to LLVM**: Standard memref/scf/arith to LLVM conversion

This approach provides the best of both worlds: clean tensor IR for operations and composition, efficient memref code for execution.

The common operations work identically in Chapters 11, 12, 13, and 14, providing architectural consistency. Changes to these operations (e.g., adding tiling optimizations or vectorization) propagate automatically across all chapters. The GPT-specific operations remain independent, allowing Chapters 13-14 to implement unique functionality (integer token indexing, conditional masking, pairwise dimension rotation) without constraining Chapter 12's design. This design ensures that LayerNorm and Linear use linalg.ReduceOp, linalg.GenericOp, linalg.MatmulOp, and linalg.FillOp rather than manual SCF loops—providing Transform dialect hooks for advanced optimizations in Chapter 14.

## 13.2 Token Embeddings: From IDs to Vectors

Language models operate on continuous vector spaces, but text consists of discrete tokens (words, subwords, or characters). Token embeddings bridge this gap by mapping each token ID to a learned vector representation. This section implements embedding lookup as an MLIR operation, demonstrates efficient lowering to memory access, and explores the role of embeddings in the model.

**Embedding as Lookup Table**. Conceptually, an embedding layer is a matrix `E` with shape `[vocab_size, d_model]`. Each row corresponds to one token's embedding vector. Given input token IDs `[t_1, t_2, ..., t_n]`, embedding lookup retrieves rows:

```
embedded[i] = E[token_ids[i]]  # Retrieve row token_ids[i] from E
```

For vocabulary size `V = 50,000` and `d_model = 768`, the embedding table contains 38.4 million parameters. During training (not covered in this book), these embeddings learn to capture semantic relationships: similar words get similar embeddings, enabling the model to generalize across related tokens.

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

**Tensor-First Architecture**. Chapter 13 adopts a tensor-first design with functional-style operations: operations take tensor inputs and return tensor results, rather than using out-parameter memrefs. This matches modern MLIR best practices and enables automatic bufferization. Input indices are `int32`, table and output are `float32`.

**Lowering to Standard Dialects**. Embedding lookup first lowers to tensor operations, then bufferization converts to memrefs with nested SCF loops:

```cpp
// src/TransformerPasses.cpp
struct EmbeddingOpLowering : public OpRewritePattern<EmbeddingOp> {
  using OpRewritePattern::OpRewritePattern;
  
  LogicalResult matchAndRewrite(EmbeddingOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value indices = op.getIndices();
    Value table = op.getTable();
    
    auto indicesType = cast<RankedTensorType>(indices.getType());
    auto tableType = cast<RankedTensorType>(table.getType());
    
    int64_t seqLen = indicesType.getShape()[0];
    int64_t dModel = tableType.getShape()[1];
    
    // Create output tensor type
    auto resultType = RankedTensorType::get(
      {seqLen, dModel}, tableType.getElementType()
    );
    
    // Allocate output tensor (will be bufferized to memref.alloc later)
    Value empty = rewriter.create<tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType()
    );
    
    Value zeroIdx = createConstantIndex(rewriter, loc, 0);
    Value oneIdx = createConstantIndex(rewriter, loc, 1);
    Value seqLenVal = createConstantIndex(rewriter, loc, seqLen);
    Value dModelVal = createConstantIndex(rewriter, loc, dModel);
    
    // For each position in sequence (generates SCF loop with tensor updates)
    Value result = rewriter.create<scf::ForOp>(
      loc, zeroIdx, seqLenVal, oneIdx, ValueRange{empty},
      [&](OpBuilder &b, Location loc, Value i, ValueRange iterArgs) {
        Value currentTensor = iterArgs[0];
        
        // Extract token ID (int32) and convert to index
        Value tokenId32 = b.create<tensor::ExtractOp>(loc, indices, ValueRange{i});
        Value tokenIdx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), tokenId32);
        
        // Copy embedding vector: extract slice from table, insert into output
        Value updatedTensor = b.create<scf::ForOp>(
          loc, zeroIdx, dModelVal, oneIdx, ValueRange{currentTensor},
          [&](OpBuilder &b2, Location loc, Value j, ValueRange args2) {
            Value tensor = args2[0];
            Value embVal = b2.create<tensor::ExtractOp>(loc, table, ValueRange{tokenIdx, j});
            Value updated = b2.create<tensor::InsertOp>(loc, embVal, tensor, ValueRange{i, j});
            b2.create<scf::YieldOp>(loc, updated);
          }).getResult(0);
        
        b.create<scf::YieldOp>(loc, updatedTensor);
      }).getResult(0);
    
    rewriter.replaceOp(op, result);
    return success();
  }
};
```

This lowering generates nested SCF loops operating on tensors: the outer loop iterates over sequence positions, the inner loop copies each embedding dimension using `tensor.extract` and `tensor.insert`. The implementation uses tensor operations rather than structured linalg operations, since embedding lookup is random-access (indexed by token ID) rather than strided iteration.

**Bufferization Pipeline**. After tensor-based lowering, the bufferization pipeline automatically converts tensor operations to efficient memref code:

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

**Testing Embedding Lookup**. Numerical validation:

```python
import numpy as np

# Create embedding table
vocab_size = 1000
d_model = 128
embedding_table = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02

# Input tokens
token_ids = np.array([5, 10, 7, 99], dtype=np.int32)

# MLIR implementation
output_mlir = compiler.embedding(token_ids, embedding_table)

# NumPy reference
output_numpy = embedding_table[token_ids]

# Verify exact equality (no floating-point arithmetic, just indexing)
assert np.array_equal(output_mlir, output_numpy)
print("✓ Embedding test passed")
```

Unlike operations involving arithmetic (matmul, attention), embedding lookup should produce **exact** results—it's pure memory access without floating-point computation. Any discrepancy indicates a bug in index calculation.

**Tied Embeddings**. Modern GPTs use the embedding table twice:
1. **Input**: Convert token IDs to embeddings
2. **Output**: Project final hidden states to vocabulary logits

The output projection is `logits = hidden @ embedding_table.T`, reusing the input embedding weights. This "weight tying" reduces parameters and improves performance—the model learns one representation space for both input and output tokens.

**Why Learn Embeddings?** Initial embeddings are random, meaningless vectors. During training, backpropagation adjusts embeddings so semantically similar tokens have similar vectors. For example, after training on large text corpora:

- `embedding("king") - embedding("man") + embedding("woman") ≈ embedding("queen")`
- `embedding("Paris") - embedding("France") + embedding("Germany") ≈ embedding("Berlin")`

These geometric relationships emerge from predicting next tokens: if "king" and "queen" appear in similar contexts, their embeddings must be similar for the model to make accurate predictions. Chapter 13 uses random embeddings (no training), but production models like GPT-2 have embeddings learned from billions of tokens.

## 13.3 Causal Masking: Preventing Future Attention

Autoregressive language models must predict each token using only previous context, never future tokens. During self-attention, this constraint is enforced through **causal masking**—zeroing attention weights for future positions. This section explains why causal masking matters, how it modifies attention computation, and how to implement it efficiently in MLIR.

**The Causality Requirement**. In language modeling, we predict token `t_i` conditioned on earlier tokens `[t_1, t_2, ..., t_{i-1}]`. If the model could see `t_i` when predicting `t_i`, the task becomes trivial (copy the input). Similarly, seeing `t_{i+1}` when predicting `t_i` is cheating—the model would learn to look ahead rather than understand context.

Causal masking enforces this constraint: when computing attention at position `i`, only positions `[0, 1, ..., i]` contribute to the weighted sum. Positions `[i+1, i+2, ..., seq_len-1]` receive zero attention weight.

**Attention Without Masking (Bidirectional)**. Recall Chapter 11's attention mechanism:

```
scores = Q @ K^T / sqrt(d_k)              # [seq_len, seq_len]
attention_weights = softmax(scores)        # [seq_len, seq_len]
output = attention_weights @ V             # [seq_len, d_model]
```

Each position attends to all positions. For a 4-token sequence, the attention weight matrix has no restrictions:

```
Attention weights (bidirectional):
       k=0   k=1   k=2   k=3
q=0   [ w00  w01  w02  w03 ]   # Token 0 attends to all tokens
q=1   [ w10  w11  w12  w13 ]   # Token 1 attends to all tokens
q=2   [ w20  w21  w22  w23 ]   # Token 2 attends to all tokens
q=3   [ w30  w31  w32  w33 ]   # Token 3 attends to all tokens
```

This is appropriate for tasks like sentence classification (BERT uses bidirectional attention) but breaks causality for language modeling.

**Attention With Causal Masking (Autoregressive)**. For language modeling, we mask future positions:

```
       k=0   k=1   k=2   k=3
q=0   [ w00   0     0     0  ]   # Token 0 attends only to itself
q=1   [ w10  w11    0     0  ]   # Token 1 attends to tokens 0-1
q=2   [ w20  w21   w22    0  ]   # Token 2 attends to tokens 0-2
q=3   [ w30  w31   w32   w33 ]   # Token 3 attends to all tokens
```

The attention matrix is **lower triangular**: `w[i, j] = 0` if `j > i`. This structure ensures each token sees only itself and previous tokens.

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

**Why -inf, Not Just 0?** Setting future scores to zero wouldn't work:

```python
# WRONG: Setting scores to 0
scores[i, j] = 0 if j > i else original_score[i, j]
attention_weights = softmax(scores)
# Problem: exp(0) = 1, so future positions still get non-zero attention!
```

Softmax exponentiates scores: `softmax(x)_i = exp(x_i) / sum(exp(x_j))`. Setting `x_i = 0` makes `exp(x_i) = 1`, still contributing to the weighted sum. We need `exp(x_i) = 0`, which requires `x_i = -inf`.

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

**Lowering Masked Softmax**. The lowering uses nested SCF loops to apply mask and compute softmax (similar to embedding lookup, the mask is generated on-the-fly):

```cpp
struct MaskedSoftmaxOpLowering : public OpRewritePattern<MaskedSoftmaxOp> {
  using OpRewritePattern::OpRewritePattern;
  
  LogicalResult matchAndRewrite(MaskedSoftmaxOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value mask = op.getMask();
    
    auto inputType = cast<RankedTensorType>(input.getType());
    auto shape = inputType.getShape();
    size_t ndim = shape.size();
    bool is2D = (ndim == 2);
    
    // Step 1: Add mask to logits (masked_logits = logits + mask)
    // Step 2: Compute max for numerical stability
    // Step 3: Compute exp(masked_logits - max) and sum
    // Step 4: Normalize by sum
    
    // Implementation uses nested scf.for loops with tensor updates,
    // supporting both 2D [seq_len, seq_len] and 3D [batch, seq_len, seq_len]
    
    // Create empty output tensor
    Value empty = rewriter.create<tensor::EmptyOp>(
      loc, inputType.getShape(), inputType.getElementType()
    );
    
    // Generate loops for max computation, exp sum, and normalization
    // (Full implementation in TransformerPasses.cpp, lines ~640-760)
    Value result = lowerMaskedSoftmaxTensorOps(rewriter, loc, input, mask, empty, is2D);
    
    rewriter.replaceOp(op, result);
    return success();
  }
};
```

The implementation generates nested SCF loops that:
1. Add mask to input logits (element-wise)
2. Find max value for numerical stability
3. Compute exp(logit + mask - max) and sum
4. Normalize: softmax[i] = exp(logit[i] + mask[i] - max) / sum

After bufferization, these tensor operations become efficient memref loads/stores with in-place updates. The mask doesn't require separate memory allocation—it's applied during the element-wise addition.

**Memory and Computation Trade-offs**. Generating the mask on-the-fly saves memory (no need to store `seq_len × seq_len` mask) but adds computation (comparison per element). For typical sequence lengths (512-2048 tokens), the memory savings dominate: a 2048×2048 mask requires 16MB (float32), while the comparison adds negligible overhead.

Advanced implementations use **block-sparse attention** (FlashAttention, covered in Chapter 14) that avoids materializing the full attention matrix, computing masked attention in smaller blocks with improved memory efficiency.

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

**Testing Causal Masking**. Validation strategy:

```python
# Input scores (before masking)
scores = np.random.randn(4, 4).astype(np.float32)

# MLIR masked softmax
output_mlir = compiler.masked_softmax(scores)

# NumPy reference
mask = np.triu(np.ones((4, 4)), k=1) * -np.inf
masked_scores = scores + mask
output_numpy = scipy.special.softmax(masked_scores, axis=-1)

# Verify
np.testing.assert_allclose(output_mlir, output_numpy, rtol=1e-5)

# Check causality: upper triangle should be zero
upper_triangle = np.triu(output_mlir, k=1)
assert np.allclose(upper_triangle, 0.0), "Future positions have non-zero attention!"

print("✓ Causal masking test passed")
```

The test verifies both numerical correctness (matching reference) and structural correctness (upper triangle is zero).

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

**Parameter Organization**. Each transformer block requires ~3.1M parameters (Chapter 12.4). For a 12-block GPT:

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

In production models, these parameters are learned during training. Chapter 13 uses random initialization for demonstration—the architecture works identically with trained weights.

**Information Flow Through Layers**. What does each layer do? Intuitively:

- **Lower layers** (blocks 1-4): Capture local patterns, syntax, basic word relationships
- **Middle layers** (blocks 5-8): Build intermediate representations, capture phrase-level semantics
- **Upper layers** (blocks 9-12): Model high-level semantics, long-range dependencies, task-specific features

This hierarchical representation emerges during training. The model learns to extract progressively abstract features—lower layers handle "how words combine," upper layers handle "what the text means."

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
        all_weights[base + 0],  all_weights[base + 1],   // Q
        all_weights[base + 2],  all_weights[base + 3],   // K
        all_weights[base + 4],  all_weights[base + 5],   // V
        all_weights[base + 6],  all_weights[base + 7],   // O
        all_weights[base + 8],  all_weights[base + 9],   // FFN W1
        all_weights[base + 10], all_weights[base + 11],  // FFN W2
        all_weights[base + 12], all_weights[base + 13],  // LN1
        all_weights[base + 14], all_weights[base + 15]   // LN2
    );
  }
  
  // Final layer norm
  return layer_norm(hidden, final_gamma, final_beta);
}
```

**Why Composition Functions Instead of Dialect Operations?** As discussed in Chapter 12, this design offers several advantages:

1. **Flexibility**: The Python layer can easily adjust control flow (number of layers, conditional blocks) without modifying the dialect
2. **Simplicity**: The dialect remains focused on primitive operations that benefit from custom lowering
3. **Optimization**: MLIR's optimizer sees the expanded graph of primitive operations, enabling cross-operation optimization

The `gpt_forward()` function builds a computation graph by calling primitive operations (`embedding`, `gpt_block`, `layer_norm`), then compiles and executes the entire graph as a single MLIR module.

**Gradient Flow in Deep Stacks**. Why don't gradients vanish in 96-layer GPT-3? Residual connections (Chapter 12.4) provide direct paths for gradients to flow backward:

```
∂Loss/∂x_block[i] = ∂Loss/∂x_block[i+1] × (1 + ∂transformer_block/∂x)
```

The `+1` term from the residual (`x + block(x)`) ensures gradients don't shrink exponentially—even if `∂transformer_block/∂x` is small, the identity path preserves gradient magnitude. This enables training very deep networks.

**Memory Consumption**. During forward pass, we must store activations for each block (needed for gradient computation during training). For a model with:
- Batch size: 8
- Sequence length: 512
- Model dimension: 768
- Number of blocks: 12

Activation memory per block:
```
Input/output: batch × seq_len × d_model × 4 bytes = 8 × 512 × 768 × 4 ≈ 12 MB
Attention scores: batch × num_heads × seq_len × seq_len × 4 = 8 × 12 × 512 × 512 × 4 ≈ 50 MB
FFN intermediate: batch × seq_len × d_ff × 4 = 8 × 512 × 3072 × 4 ≈ 50 MB
──────────────────────────────────────────────────────────────────────────────────
Total per block: ~112 MB
Total for 12 blocks: ~1.3 GB
```

This explains why large-batch training requires substantial GPU memory. Techniques like gradient checkpointing trade computation for memory by recomputing activations during backward pass rather than storing them.

**Testing Stacked Blocks**. Validation ensures data flows correctly:

```python
# Initialize random parameters
d_model = 128
d_ff = 512
num_blocks = 3

# Create block weights
all_weights = initialize_random_blocks(d_model, d_ff, num_blocks)

# Input
x = np.random.randn(16, d_model).astype(np.float32)  # [seq_len=16, d_model=128]

# MLIR forward pass
output_mlir = compiler.gpt_forward(x, all_weights, num_blocks)

# Reference: manually apply blocks sequentially
output_ref = x
for block_weights in all_weights:
    output_ref = transformer_block_reference(output_ref, block_weights)

# Verify
np.testing.assert_allclose(output_mlir, output_ref, rtol=1e-3)
assert output_mlir.shape == (16, d_model), f"Wrong output shape: {output_mlir.shape}"
print("✓ Stacked blocks test passed")
```

The test compares MLIR's stacked implementation against manually chaining block operations, ensuring correct composition.

## 13.5 Complete GPT Forward Pass

With all components implemented (embeddings, causal masking, transformer blocks), we now assemble the complete GPT forward pass. This section demonstrates end-to-end inference: from input token IDs to output vocabulary logits, ready for next-token prediction or generation.

**The Full Forward Pass**. Combining all stages:

```python
def gpt_forward(token_ids, weights):
    """
    Complete GPT forward pass.
    
    Args:
        token_ids: [seq_len] int32 token IDs
        weights: dict containing all model parameters:
            - 'embedding_table': [vocab_size, d_model]
            - 'block_weights': list of num_blocks transformer block parameters
            - 'final_gamma': [d_model] final layer norm scale
            - 'final_beta': [d_model] final layer norm shift
    
    Returns:
        logits: [seq_len, vocab_size] next-token predictions
    """
    # Stage 1: Embedding lookup
    embeddings = embedding_lookup(token_ids, weights['embedding_table'])
    # Shape: [seq_len, d_model]
    
    # Stage 2: Process through transformer blocks
    hidden = embeddings
    for block_weights in weights['block_weights']:
        hidden = transformer_block(hidden, block_weights)
    # Shape: [seq_len, d_model]
    
    # Stage 3: Final layer normalization
    hidden = layernorm(hidden, weights['final_gamma'], weights['final_beta'])
    # Shape: [seq_len, d_model]
    
    # Stage 4: Project to vocabulary (language modeling head)
    logits = hidden @ weights['embedding_table'].T
    # Shape: [seq_len, vocab_size]
    
    return logits
```

Each position in the sequence receives a vocabulary-sized logit vector. For training, these logits compare against true next tokens via cross-entropy loss. For generation, we sample from the logit distribution at the last position.

**Tied Embeddings in Action**. Notice Stage 4 reuses the embedding table for output projection. Why tie weights?

1. **Parameter Efficiency**: Reduces parameters by `vocab_size × d_model`. For GPT-2 (vocab 50K, d_model 768), this saves ~38M parameters.

2. **Improved Generalization**: The model learns one shared representation space—input and output tokens use the same semantic embeddings. This constraint often improves performance.

3. **Theoretical Motivation**: In Transformer architectures, the embedding layer and output layer are inverses of each other (one maps tokens→vectors, the other maps vectors→tokens). Sharing weights enforces this symmetry.

Not all models tie weights (GPT-3 uses separate output projection), but it's common and effective for smaller models.

**Python API Implementation**. The complete forward pass is exposed through the `gpt_forward()` function:

```python
# Python usage
import ch13

# Model parameters
token_ids = np.array([72, 101, 108, 108, 111], dtype=np.int32)  # "Hello"
embedding_table = np.random.randn(vocab_size, d_model).astype(np.float32)
all_weights = [layer_weights for layer in range(num_layers)]  # 16 weights per layer
final_gamma = np.ones(d_model, dtype=np.float32)
final_beta = np.zeros(d_model, dtype=np.float32)

# Forward pass
hidden_states = ch13.forward(
    ch13.gpt_forward(token_ids, embedding_table, all_weights, final_gamma, final_beta)
)

# Add output projection for next-token prediction
logits = ch13.forward(ch13.matmul(hidden_states, ch13.transpose(embedding_table)))
```

The implementation follows Chapter 12's pattern: `gpt_forward()` is a composition function (not a dialect operation) that builds a computation graph from primitive operations. The graph builder API allows operations to be composed before compilation:

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

**Testing End-to-End Forward Pass**. Comprehensive validation:

```python
# Model configuration
vocab_size = 256  # Byte-level tokens
d_model = 64
num_heads = 4
d_ff = d_model * 4
num_blocks = 2
seq_len = 8

# Initialize random weights
embedding_table = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02
block_weights = [initialize_random_block(d_model, d_ff) for _ in range(num_blocks)]
final_gamma = np.ones(d_model, dtype=np.float32)
final_beta = np.zeros(d_model, dtype=np.float32)

# Input tokens
token_ids = np.array([72, 101, 108, 108, 111, 32, 87, 111], dtype=np.int32)  # "Hello Wo"

# MLIR forward pass
logits_mlir = gpt_forward(token_ids, embedding_table, block_weights, final_gamma, final_beta)

# Reference implementation (NumPy/PyTorch)
logits_ref = reference_gpt_forward(token_ids, embedding_table, block_weights, final_gamma, final_beta)

# Verify
assert logits_mlir.shape == (seq_len, vocab_size), f"Wrong output shape: {logits_mlir.shape}"
np.testing.assert_allclose(logits_mlir, logits_ref, rtol=1e-3, atol=1e-5)
print("✓ GPT forward pass test passed")

# Verify logits are reasonable (not NaN/inf)
assert not np.isnan(logits_mlir).any(), "Logits contain NaN!"
assert not np.isinf(logits_mlir).any(), "Logits contain inf!"
print("✓ Numerical stability verified")
```

The test validates correctness (matching reference) and stability (no NaN/inf). Production tests also verify:
- Output probabilities sum to 1 after softmax
- Most probable tokens make linguistic sense (for trained models)
- Performance meets latency requirements

**Next Token Prediction**. Given logits, predict the next token:

```python
# Get logits for the last position (what comes after "Hello Wo"?)
last_logits = logits_mlir[-1, :]  # [vocab_size]

# Convert to probabilities
probs = scipy.special.softmax(last_logits)

# Greedy decoding: pick most probable token
next_token_id = np.argmax(probs)
next_token = chr(next_token_id)  # Convert to character (byte-level)

print(f"Next token: {next_token} (ID: {next_token_id})")
```

For random weights, the predicted token is meaningless. For trained models (GPT-2, GPT-3), the prediction completes the sequence sensibly: "Hello Wo" → "rld" (completing "World").

**Performance Characteristics**. For the minimal configuration (seq_len=8, d_model=64, num_blocks=2):

- **Embedding lookup**: ~0.001 ms (trivial memory access)
- **Transformer blocks**: ~0.5 ms × 2 = 1.0 ms (dominates)
- **Final layer norm**: ~0.001 ms
- **Output projection**: ~0.01 ms
- **Total latency**: ~1.02 ms

For production scale (seq_len=2048, d_model=12288, num_blocks=96), latency exceeds seconds without optimization. Chapter 14 addresses this through KV caching, operator fusion, and quantization.

## 13.6 Testing and Validation

Building a complex system like GPT requires rigorous testing at multiple levels: individual components (embeddings, masking), composed sub-systems (attention, blocks), and end-to-end behavior. This section presents a comprehensive testing strategy, debugging techniques, and validation against reference implementations.

**Testing Pyramid**. We test at three levels:

1. **Unit Tests**: Individual operations in isolation
2. **Integration Tests**: Composed operations (attention, transformer blocks)
3. **System Tests**: Complete forward pass with realistic inputs

Each level catches different bugs: unit tests find arithmetic errors, integration tests find composition issues, system tests find emergent problems.

**Unit Test Examples**:

```python
def test_embedding_lookup():
    """Test embedding operation correctness."""
    vocab_size, d_model = 100, 32
    embedding_table = np.random.randn(vocab_size, d_model).astype(np.float32)
    token_ids = np.array([5, 10, 15], dtype=np.int32)
    
    output = gpt.embedding(token_ids, embedding_table)
    expected = embedding_table[token_ids]
    
    assert output.shape == (3, d_model)
    np.testing.assert_array_equal(output, expected)  # Exact equality for lookup
    print("✓ Embedding lookup test passed")

def test_causal_masking():
    """Test causal mask prevents future attention."""
    seq_len = 4
    scores = np.random.randn(seq_len, seq_len).astype(np.float32)
    
    attention_weights = gpt.masked_softmax(scores)
    
    # Verify upper triangle is zero
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            assert attention_weights[i, j] < 1e-6, f"Future attention at [{i},{j}]: {attention_weights[i, j]}"
    
    # Verify rows sum to 1 (valid probability distribution)
    row_sums = attention_weights.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)
    print("✓ Causal masking test passed")

def test_layer_normalization():
    """Test layer norm statistics."""
    x = np.random.randn(8, 64).astype(np.float32) * 10 + 5  # Non-zero mean
    gamma = np.ones(64, dtype=np.float32)
    beta = np.zeros(64, dtype=np.float32)
    
    output = gpt.layernorm(x, gamma, beta)
    
    # Verify mean ≈ 0, variance ≈ 1 (per token)
    for i in range(8):
        assert abs(output[i].mean()) < 0.1, f"Token {i} mean not near zero"
        assert abs(output[i].var() - 1.0) < 0.1, f"Token {i} variance not near 1"
    print("✓ Layer normalization test passed")
```

Unit tests isolate failures: if `test_embedding_lookup` fails, we know the bug is in embedding implementation, not elsewhere.

**Integration Test Examples**:

```python
def test_transformer_block():
    """Test complete transformer block."""
    seq_len, d_model, d_ff = 16, 64, 256
    x = np.random.randn(seq_len, d_model).astype(np.float32)
    block_weights = initialize_random_block(d_model, d_ff)
    
    # MLIR implementation
    output_mlir = gpt.transformer_block(x, block_weights)
    
    # Reference implementation (PyTorch or NumPy)
    output_ref = reference_transformer_block(x, block_weights)
    
    assert output_mlir.shape == (seq_len, d_model)
    np.testing.assert_allclose(output_mlir, output_ref, rtol=1e-3)
    print("✓ Transformer block test passed")

def test_causal_attention():
    """Test attention with causal masking."""
    seq_len, d_model = 8, 32
    x = np.random.randn(seq_len, d_model).astype(np.float32)
    
    # Simple attention (Q=K=V=x for testing)
    Q = K = V = x
    output = gpt.causal_attention(Q, K, V)
    
    # Verify causality: changing future positions shouldn't affect past outputs
    x_modified = x.copy()
    x_modified[-1, :] = 999  # Change last position
    output_modified = gpt.causal_attention(x_modified, x_modified, x_modified)
    
    # First seq_len-1 positions should be unchanged
    np.testing.assert_allclose(output[:7], output_modified[:7], rtol=1e-5)
    print("✓ Causal attention test passed")
```

Integration tests catch composition bugs: incorrect tensor shapes between operations, wrong parameter passing, etc.

**System Test Examples**:

```python
def test_gpt_forward_shape():
    """Test GPT forward pass output shape."""
    vocab_size, d_model, num_blocks = 256, 64, 2
    seq_len = 10
    
    token_ids = np.random.randint(0, vocab_size, size=seq_len, dtype=np.int32)
    weights = initialize_random_gpt(vocab_size, d_model, num_blocks)
    
    logits = gpt.forward(token_ids, weights)
    
    assert logits.shape == (seq_len, vocab_size), f"Wrong shape: {logits.shape}"
    assert not np.isnan(logits).any(), "Logits contain NaN"
    assert not np.isinf(logits).any(), "Logits contain inf"
    print("✓ GPT forward shape test passed")

def test_gpt_numerical_correctness():
    """Test GPT against reference implementation."""
    vocab_size, d_model, num_blocks = 128, 32, 2
    seq_len = 5
    
    token_ids = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    weights = initialize_random_gpt(vocab_size, d_model, num_blocks)
    
    # MLIR implementation
    logits_mlir = gpt.forward(token_ids, weights)
    
    # PyTorch reference
    logits_torch = pytorch_gpt_forward(token_ids, weights)
    
    np.testing.assert_allclose(logits_mlir, logits_torch, rtol=1e-3, atol=1e-5)
    print("✓ GPT numerical correctness test passed")

def test_gpt_deterministic():
    """Test that forward pass is deterministic."""
    token_ids = np.array([1, 2, 3, 4], dtype=np.int32)
    weights = initialize_random_gpt(256, 64, 2)
    
    # Run twice with same inputs
    logits1 = gpt.forward(token_ids, weights)
    logits2 = gpt.forward(token_ids, weights)
    
    # Should be identical (bit-exact)
    np.testing.assert_array_equal(logits1, logits2)
    print("✓ GPT deterministic test passed")
```

System tests validate end-to-end behavior: correct shapes, numerical accuracy, determinism.

**Debugging Strategies**. When tests fail:

1. **Isolate the Failure**: Binary search which component is wrong (run unit tests, then integration tests)

2. **Print Intermediate Values**: Add debug prints in MLIR lowering or Python code to inspect tensors

3. **Compare Against Reference**: Compute the same operation in NumPy/PyTorch, compare intermediate results

4. **Simplify the Input**: Use minimal examples (seq_len=2, d_model=4) to make outputs human-readable

5. **Check for Common Bugs**:
   - NaN/inf from divide by zero
   - Shape mismatches from incorrect broadcasting
   - Index out of bounds from wrong loop limits
   - Uninitialized memory from missing allocations

**Example Debugging Session**:

```python
# Bug: GPT forward returns NaN

# Step 1: Run unit tests to isolate
test_embedding_lookup()  # ✓ Pass
test_layernorm()         # ✓ Pass
test_causal_attention()  # ✗ Fail - NaN detected!

# Step 2: Simplify the failing test
def test_causal_attention_simple():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # 2×2
    output = gpt.causal_attention(x, x, x)
    print(output)  # [[NaN, NaN], [NaN, NaN]]

# Step 3: Compare with reference
ref_output = reference_causal_attention(x, x, x)
print(ref_output)  # [[1.0, 2.0], [2.33, 3.33]] - reference works!

# Step 4: Inspect intermediate values
def causal_attention_debug(Q, K, V):
    scores = Q @ K.T
    print("Scores:", scores)  # [[5, 11], [11, 25]]
    
    masked = apply_causal_mask(scores)
    print("Masked:", masked)  # [[5, -inf], [11, 25]]
    
    attention = softmax(masked)
    print("Attention:", attention)  # [[NaN, NaN], [NaN, NaN]] - Bug is in softmax!

# Step 5: Fix the bug
# Bug was: softmax([-inf, -inf]) = [NaN, NaN]
# Fix: Handle all-negative-infinity rows (first position has no valid attention targets)
```

Systematic debugging finds bugs quickly—avoid guessing, use data.

**Performance Regression Tests**. Beyond correctness, test performance:

```python
def test_gpt_performance():
    """Test that forward pass meets latency requirements."""
    token_ids = np.random.randint(0, 256, size=32, dtype=np.int32)
    weights = initialize_random_gpt(256, 64, 2)
    
    # Warmup
    for _ in range(10):
        _ = gpt.forward(token_ids, weights)
    
    # Timed runs
    import time
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = gpt.forward(token_ids, weights)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    mean_latency = np.mean(times)
    print(f"Mean latency: {mean_latency*1000:.2f} ms")
    
    # Assert latency requirement
    assert mean_latency < 0.01, f"Too slow: {mean_latency*1000:.2f} ms > 10 ms"
    print("✓ Performance test passed")
```

Performance tests catch regressions: if an optimization accidentally breaks, latency increases, and the test fails.

## 13.7 Summary

Chapter 13 Part 1 constructed complete GPT architecture by composing token embeddings, causal masking, and stacked transformer blocks. We implemented the full forward pass—from input token IDs to output vocabulary logits—demonstrating how MLIR handles models with millions of parameters and complex data flow.

Key insights:

- **Token Embeddings**: Simple lookup tables that map discrete tokens to continuous vectors, enabling neural network processing of text
- **Causal Masking**: Enforcing autoregressive structure by preventing future attention, essential for language modeling and generation
- **Sequential Composition**: Stacking transformer blocks creates deep representations, with residual connections enabling gradient flow through dozens of layers
- **Tied Embeddings**: Reusing input embeddings for output projection reduces parameters and improves generalization

Chapter 13 Part 1 established the complete GPT forward pass. The model can now compute next-token predictions for any input sequence, though we haven't yet covered how to generate text autoregressively (sampling from logits, managing context windows) or modern positional encodings (RoPE).

**Looking Ahead**. Chapter 13 Part 2 will implement:
- **Rotary Position Embeddings (RoPE)**: Modern positional encoding enabling better length extrapolation
- **Autoregressive Generation**: Sampling strategies (greedy, temperature, top-k) for text generation
- **Generation Pipeline**: Iteratively generating tokens, managing KV cache (preview of Chapter 14's optimization)
- **Text Generation Demo**: Complete end-to-end example generating text from prompts

Together, Parts 1 and 2 deliver a fully functional GPT implementation—small-scale but architecturally identical to production models like GPT-2, GPT-3, and LLaMA. Chapter 14 will then optimize this architecture for production serving with FlashAttention, quantization, and batching.