# Chapter 12: Transformer Blocks

Chapter 11 built scaled dot-product attention—the core mechanism enabling transformers to weigh input relevance dynamically. We implemented query-key similarity scoring, softmax normalization, and value aggregation, achieving numerically correct results through MLIR's JIT compilation pipeline. Attention alone, however, doesn't constitute a transformer. Production transformer architectures combine attention with feedforward networks, layer normalization, and residual connections, forming **transformer blocks**—the fundamental building blocks of models like GPT, BERT, and LLaMA.

This chapter composes Chapter 11's attention with additional components to build complete transformer blocks. We'll implement **layer normalization** (normalizing activations across the embedding dimension for training stability), **feedforward networks** (two-layer MLPs with nonlinear activations providing per-position computation), and **residual connections** (skip connections enabling gradient flow in deep networks). The Python API remains simple—users call `transformer_block(x)` and get output—but underneath, MLIR orchestrates complex multi-operation pipelines with automatic optimization.

Chapter 12's architecture follows established patterns: we extend the Transformer dialect (from Chapter 11's attention operations) with `transformer.layer_norm`, `transformer.linear`, `transformer.gelu`, and `transformer.add` operations, implement lowering patterns to standard dialects, and maintain the computation graph API. The result is a reusable transformer block abstraction suitable for building larger models (Chapter 13's GPT) while demonstrating how MLIR's compositional design scales from individual operations to complex architectural components.

The chapter progresses from understanding transformer block architecture (why each component matters), through implementing layer normalization (variance calculation and normalization), feedforward networks (linear layers with GELU activation), to composing everything with residual connections. We'll see how MLIR's optimization passes (Chapter 10's fusion and vectorization) automatically apply to these new operations without additional code. By the end, you'll have a complete transformer block implementation and understand how production transformers decompose into manageable, optimizable components.

## 12.1 Transformer Block Architecture

Before implementing code, we must understand what a transformer block computes and why its components matter. Modern transformer architectures (GPT, LLaMA, DeepSeek) share a common structure: stacked transformer blocks, each containing attention and feedforward sub-layers with layer normalization and residual connections. This section dissects the architecture, explaining each component's purpose and how they interact.

**The Standard Transformer Block**. A transformer block processes input `x` (shape: `[seq_len, d_model]`) through two main sub-layers:

```
TransformerBlock(x):
  # Sub-layer 1: Multi-head attention with residual
  attn_out = MultiHeadAttention(LayerNorm(x))
  x = x + attn_out

  # Sub-layer 2: Feedforward network with residual
  ffn_out = FeedForward(LayerNorm(x))
  x = x + ffn_out

  return x
```

This is the **pre-normalization** architecture (normalize before sub-layers), popularized by GPT-2 and now standard in modern transformers. The alternative **post-normalization** architecture (normalize after residual addition) was used in the original "Attention is All You Need" paper but proved harder to train for deep networks. We'll implement pre-norm because it's what production models use.

**Multi-Head Attention Review**. Chapter 11 implemented scaled dot-product attention:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Multi-head attention extends this by running multiple attention heads in parallel with different learned projections:

```python
# Multi-head attention (conceptual)
def multi_head_attention(x, W_q, W_k, W_v, W_o, num_heads):
    # Project input to Q, K, V
    Q = x @ W_q  # [seq_len, d_model] @ [d_model, d_model] → [seq_len, d_model]
    K = x @ W_k
    V = x @ W_v

    # Split into multiple heads
    Q_heads = split(Q, num_heads)  # num_heads × [seq_len, d_k]
    K_heads = split(K, num_heads)
    V_heads = split(V, num_heads)

    # Run attention on each head
    head_outputs = [attention(q, k, v) for q, k, v in zip(Q_heads, K_heads, V_heads)]

    # Concatenate heads and project
    concat = concatenate(head_outputs)  # [seq_len, d_model]
    output = concat @ W_o  # [seq_len, d_model] @ [d_model, d_model] → [seq_len, d_model]

    return output
```

Multiple heads let the model attend to different aspects of the input simultaneously—one head might focus on syntactic dependencies, another on semantic relationships. Chapter 12 will implement this multi-head structure on top of Chapter 11's single-head attention primitive.

**Layer Normalization**. Training deep neural networks suffers from **internal covariate shift**—layer input distributions change as earlier layers update—though modern research suggests normalization primarily works by **smoothing the optimization landscape**, making gradients more predictable. Layer normalization addresses this by normalizing activations to zero mean and unit variance:

$$\text{LayerNorm}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \boldsymbol{\beta}$$

Where $\mu$ and $\sigma^2$ are the mean and variance computed across the embedding dimension $d_{\text{model}}$, and $\odot$ represents element-wise multiplication with the learnable parameter vectors $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$. The epsilon value (typically $10^{-5}$ or $10^{-6}$) prevents division by zero.

Layer normalization computes these statistics **per token independently** (unlike batch normalization which normalizes across batches). For input shape `[seq_len, d_model]`, we compute `seq_len` separate normalizations—one for each token's embedding vector. This independence across tokens makes layer norm well-suited for variable-length sequences.

**Pre-Norm vs Post-Norm Architecture**. The original Transformer (Vaswani et al., 2017) applied LayerNorm **after** the residual connection (Post-Norm): `LayerNorm(x + SubLayer(x))`. Modern transformers (GPT-3, LLaMA) apply it **before** the sub-layer (Pre-Norm): `x + SubLayer(LayerNorm(x))`, which provides significantly more stable training for deep networks. Chapter 12 implements Pre-Norm architecture, the current production standard.

**Feedforward Networks**. After attention aggregates information across tokens, the feedforward network processes each token **independently** through a position-wise sub-network (typically a two-layer MLP or gated linear unit):

$$\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

Where:
- $\mathbf{W}_1$: `[d_model, d_ff]` (typically `d_ff = 4 * d_model`)
- $\mathbf{W}_2$: `[d_ff, d_model]`
- GELU: Gaussian Error Linear Unit activation (smooth approximation of ReLU)

The feedforward network expands dimensionality to $d_{\text{ff}}$ (creating a bottleneck), applies nonlinearity, then projects back to $d_{\text{model}}$. This per-token processing provides additional model capacity and nonlinearity beyond attention's linear transformations.

**Architecture Variants**. While the classical 4× expansion is standard (GPT-2/3), modern architectures use variations:

- **GPT-2/3**: 4× expansion with GELU, includes bias terms
- **LLaMA-2 70B**: 3.5× expansion (d_model=8192, d_ff=28672) with **SwiGLU** activation and **no bias terms** for improved training stability and hardware efficiency
- **SwiGLU**: A gated linear unit using three weight matrices instead of two, achieving better performance

Chapter 12 implements the classical two-layer MLP with GELU and 4× expansion, following GPT-2/3 architecture. The principles extend naturally to gated variants like SwiGLU.

**Residual Connections**. Each sub-layer (attention, feedforward) wraps in a residual connection:

$$\mathbf{x} = \mathbf{x} + \text{SubLayer}(\mathbf{x})$$

Residuals enable gradient flow in deep networks: gradients can bypass sub-layers through the identity path, preventing vanishing gradients. This technique, borrowed from ResNet (2015), is essential for training transformers with dozens or hundreds of layers. Without residuals, deep transformer training fails—gradients vanish, and the model doesn't learn. The residual allows the model to learn **incremental updates** to embeddings rather than completely reinventing representations at each layer.

Production transformers stack many blocks: GPT-2 small uses 12 blocks, GPT-3 (175B) uses 96 blocks, LLaMA-2 70B uses 80 blocks. Each block performs the same computation (modulo learned parameters), demonstrating the architecture's modularity.

Chapter 12 focuses on the core transformer block structure, establishing patterns that Chapter 13 will extend for GPT-style autoregressive language modeling.

## 12.2 Layer Normalization: Stabilizing Activations

Layer normalization is the first new operation we'll add to the Transformer dialect. It normalizes each token's embedding vector to zero mean and unit variance, with learned scale (`gamma`) and shift (`beta`) parameters. This section implements layer norm's mathematical definition, explores numerical stability considerations, and shows how it lowers to MLIR's standard dialects.

**The Mathematics**. For an input vector `x` of dimension `d_model`, layer normalization computes:

```
mean = (1/d_model) * sum(x_i) for i in [0, d_model)
variance = (1/d_model) * sum((x_i - mean)^2) for i in [0, d_model)
x_normalized = (x - mean) / sqrt(variance + epsilon)
output = gamma * x_normalized + beta
```

For batched input with shape `[seq_len, d_model]`, we compute `seq_len` independent normalizations—one per token. Each token's `d_model` features are normalized independently of other tokens' features.

**Why Normalization Helps**. Normalization stabilizes training by maintaining consistent input statistics (zero mean, unit variance) and equalizing gradient scales across parameters. The learned parameters $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$ (shape `[d_model]`) provide flexibility—the network can adjust normalization strength per feature dimension if strict normalization is too restrictive. The epsilon term (typically $10^{-5}$) prevents division by zero when all inputs are identical.

**Operation Definition in Transformer Dialect**. We extend the Transformer dialect with a layer norm operation:

```tablegen
// inc/TransformerOps.td
def Transformer_LayerNormOp : Transformer_Op<"layer_norm"> {
  let summary = "Layer normalization operation";
  let description = [{
    Computes layer normalization over the last dimension:

      mean = sum(input) / d_model
      variance = sum((input - mean)^2) / d_model
      normalized = (input - mean) / sqrt(variance + epsilon)
      output = gamma * normalized + beta

    For input shape [seq_len, d_model], computes seq_len independent normalizations.
    Epsilon (1e-5) is hardcoded in the lowering pattern for numerical stability.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$gamma,  // [d_model]
    AnyTensor:$beta    // [d_model]
  );

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $gamma `,` $beta
    attr-dict `:` `(` type($input) `,` type($gamma) `,` type($beta) `)` `->` type($result)
  }];
}
```

The operation takes input tensor, gamma/beta parameters, and returns normalized output with the same shape as input.

**Lowering to Linalg**. Layer normalization lowers to a sequence of Linalg operations combining reductions and element-wise computations. The lowering follows four stages:

```cpp
// src/TransformerPasses.cpp
struct LayerNormOpLowering : public OpRewritePattern<LayerNormOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LayerNormOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value gamma = op.getGamma();
    Value beta = op.getBeta();
    float epsilon = 1e-5f;

    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    ArrayRef<int64_t> shape = resultType.getShape();  // [seq_len, d_model]
    int rank = shape.size();

    // Create reduced shape for mean and variance: [seq_len]
    SmallVector<int64_t> reducedShape(shape.begin(), shape.end() - 1);
    auto reducedType = RankedTensorType::get(reducedShape, rewriter.getF32Type());
```

**Stage 1: Compute Mean via Two-Stage Reduction**. First `linalg.reduce` computes sums along the $d_{\text{model}}$ dimension, then `linalg.generic` normalizes—this two-stage approach enables fusion opportunities:

```cpp
    // Sum reduction along last dimension
    Value zero = createConstantFloat(rewriter, loc, 0.0f);
    Value meanBuffer = rewriter.create<linalg::FillOp>(loc, zero, 
                         rewriter.create<tensor::EmptyOp>(loc, reducedType, ...)).getResult(0);

    Value meanSum = rewriter.create<linalg::ReduceOp>(
        loc, ValueRange{input}, ValueRange{meanBuffer},
        SmallVector<int64_t>{rank - 1},  // reduce along d_model
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        }).getResult(0);

    // Normalize: mean = sum / d_model
    Value dModel = createConstantFloat(rewriter, loc, static_cast<float>(shape[rank - 1]));
    Value meanResult = rewriter.create<linalg::GenericOp>(
        loc, reducedType, ValueRange{meanSum}, ValueRange{...},
        meanNormalizeMaps, reducedIteratorTypes,
        [dModel](OpBuilder &b, Location loc, ValueRange args) {
          Value normalized = b.create<arith::DivFOp>(loc, args[0], dModel);
          b.create<linalg::YieldOp>(loc, normalized);
        }).getResult(0);
```

**Stage 2: Broadcast Semantics with Affine Maps**. Affine maps explicitly encode how 1D statistics (mean, variance) broadcast across 2D tensors—no manual index arithmetic required:

```cpp
    // Compute centered = input - mean (with broadcasting)
    Value centeredBuffer = rewriter.create<linalg::GenericOp>(
        loc, resultType,
        ValueRange{input, meanResult},
        ValueRange{rewriter.create<tensor::EmptyOp>(loc, resultType, ...)},
        centeringMaps,  // Affine map broadcasts mean from [seq_len] to [seq_len, d_model]
        iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value centered = b.create<arith::SubFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, centered);
        }).getResult(0);

    // Compute variance via reduction and normalization
    Value varianceSum = rewriter.create<linalg::ReduceOp>(
        loc, ValueRange{centeredBuffer}, ValueRange{...},
        SmallVector<int64_t>{rank - 1},
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value squared = b.create<arith::MulFOp>(loc, args[0], args[0]);
          Value sum = b.create<arith::AddFOp>(loc, squared, args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        }).getResult(0);

    Value epsilonVal = createConstantFloat(rewriter, loc, epsilon);
    Value invStdResult = rewriter.create<linalg::GenericOp>(
        loc, reducedType, ValueRange{varianceSum}, ValueRange{...},
        meanNormalizeMaps, reducedIteratorTypes,
        [dModel, epsilonVal](OpBuilder &b, Location loc, ValueRange args) {
          Value variance = b.create<arith::DivFOp>(loc, args[0], dModel);
          Value variancePlusEps = b.create<arith::AddFOp>(loc, variance, epsilonVal);
          Value invStd = b.create<math::RsqrtOp>(loc, variancePlusEps);  // 1/sqrt(variance + epsilon)
          b.create<linalg::YieldOp>(loc, invStd);
        }).getResult(0);
```

**Stage 3: Apply Scale and Shift**. Temporary buffers for intermediate results enable in-place updates and memory reuse after deallocation:

```cpp
    // Final normalization: gamma * (centered * invStd) + beta
    Value result = rewriter.create<linalg::GenericOp>(
        loc, resultType,
        ValueRange{centeredBuffer, invStdResult, gamma, beta},
        ValueRange{rewriter.create<tensor::EmptyOp>(loc, resultType, ...)},
        normalizeMaps,  // Broadcasting maps for invStd, gamma, beta
        iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          // args: centered, invStd, gamma, beta
          Value normalized = b.create<arith::MulFOp>(loc, args[0], args[1]);
          Value scaled = b.create<arith::MulFOp>(loc, normalized, args[2]);
          Value finalResult = b.create<arith::AddFOp>(loc, scaled, args[3]);
          b.create<linalg::YieldOp>(loc, finalResult);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};
```

The pattern mirrors Chapter 11's softmax lowering: reduce to compute statistics, then element-wise operations with broadcasting. MLIR's Linalg dialect provides the abstraction layer—operations describe *what* to compute (reductions, element-wise ops), enabling subsequent passes to determine *how* (loop tiling, vectorization, parallelization).

**Testing Layer Normalization**. Numerical correctness verification:

```python
import numpy as np

def layernorm_reference(x, gamma, beta, epsilon=1e-5):
    """NumPy reference implementation."""
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(variance + epsilon)
    return gamma * normalized + beta

# Test case
x = np.random.randn(4, 512).astype(np.float32)  # [seq_len=4, d_model=512]
gamma = np.ones(512, dtype=np.float32)
beta = np.zeros(512, dtype=np.float32)

# MLIR implementation
result_mlir = compiler.layernorm(x, gamma, beta)

# NumPy reference
result_numpy = layernorm_reference(x, gamma, beta)

# Verify
np.testing.assert_allclose(result_mlir, result_numpy, rtol=1e-4, atol=1e-6)
```

The tolerance thresholds (relative tolerance 1e-4, absolute tolerance 1e-6) account for floating-point precision differences between MLIR's compiled code and NumPy's optimized BLAS implementations.

**Performance Characteristics**. Layer normalization's complexity is O(seq_len × d_model)—linear in input size. The operation is **memory-bound**: reading input, writing output, and computing mean/variance require few arithmetic operations per memory access. This makes layer norm a good target for fusion (Chapter 10)—combining it with surrounding operations reduces memory traffic.

Modern ML frameworks often fuse layer norm with attention or feedforward operations. For example, **pre-norm fusion** combines `LayerNorm(x)` directly into the attention computation, eliminating the intermediate normalized tensor. Chapter 12's modular design enables such optimizations through MLIR's pattern rewriting infrastructure, though we won't implement aggressive fusion until Chapter 14.

Layer normalization establishes the pattern for subsequent operations: define high-level dialect operation, implement lowering to standard dialects, verify numerical correctness against reference implementations, and enable composition with other operations. Feedforward networks (next section) follow this same pattern, building toward the complete transformer block.

## 12.3 Feedforward Networks: Per-Token MLPs

After attention mixes information across sequence positions, the transformer block applies a feedforward network to each token independently. This two-layer MLP (Multi-Layer Perceptron) provides additional representational capacity and nonlinearity, transforming each token's embedding through an expanded intermediate dimension before projecting back to the model dimension. This section implements feedforward networks with GELU activation, and demonstrates how MLIR optimizes these dense computations.

**The Feedforward Architecture**. A standard transformer feedforward network consists of two linear layers with nonlinear activation between them:

```
FFN(x) = Linear_2(GELU(Linear_1(x)))

Where:
  Linear_1: x @ W_1 + b_1    # [d_model] → [d_ff]
  GELU: Gaussian Error Linear Unit activation
  Linear_2: x @ W_2 + b_2    # [d_ff] → [d_model]
```

The first linear layer expands dimensionality from `d_model` to `d_ff`, creating an intermediate representation with higher capacity. The second layer projects back to `d_model`, matching the residual connection's expected shape. This bottleneck architecture (expand → activate → project) is a common pattern in neural networks.

**Architecture Choices**. The expansion ratio varies by model family:

- **GPT-2/3**: Use `d_ff = 4 * d_model` with GELU activation
- **LLaMA-2**: Uses `d_ff = 3.5 * d_model` with SwiGLU (a gated linear unit requiring three weight matrices)
- **Mixture-of-Experts (MoE)**: Expands `d_ff` dramatically but routes each token to only a subset of parameters

Chapter 12 implements a standard two-layer MLP with GELU and 4× expansion (GPT-2/3 style), providing a solid foundation that extends naturally to modern variants.

**Implementation Note**: The feedforward network is implemented as a **Python API function** (`ffn`) that composes primitive operations (`linear`, `gelu`) rather than a single dialect operation. This compositional approach enables operation reuse and independent optimization of each component.

**GELU Activation**. The **Gaussian Error Linear Unit** is a smooth approximation of ReLU:

```
GELU(x) = x * Φ(x)

Where Φ(x) is the cumulative distribution function of the standard normal distribution:
  Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
```

Intuitively, GELU weights inputs by their magnitude—large positive values pass through nearly unchanged (Φ(x) ≈ 1), large negative values are zeroed (Φ(x) ≈ 0), and values near zero are smoothly interpolated. This smooth transition avoids ReLU's abrupt gradient discontinuity at zero, potentially providing better gradient flow during training.

Computing GELU exactly requires the error function (`erf`), which is expensive. Production implementations use **approximations**:

```python
# Tanh approximation (fast, accurate to ~1e-3)
def gelu_approx(x):
    return 0.5 * x * (1.0 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
```

This approximation, provides excellent accuracy while using only multiplication, addition, and tanh (which hardware and libraries optimize heavily). MLIR's Math dialect includes `math.erf` and `math.tanh`, allowing us to choose between exact and approximate implementations.

**Primitive Operations**. Rather than defining a monolithic FFN dialect operation, Chapter 12 provides primitive operations that compose to implement feedforward networks:

```tablegen
// inc/TransformerOps.td - Individual building blocks
def Transformer_LinearOp : Transformer_Op<"linear"> {
  let summary = "Linear transformation with bias";
  let description = [{ Computes: output = input @ weight^T + bias }];
  let arguments = (ins AnyTensor:$input, AnyTensor:$weight, AnyTensor:$bias);
  let results = (outs AnyTensor:$result);
}

def Transformer_GeluOp : Transformer_Op<"gelu"> {
  let summary = "Gaussian Error Linear Unit activation";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}
```

The Python API provides a composite `ffn()` function:

```python
# Python API (bindings.cpp)
def ffn(input, w1, b1, w2, b2):
    """Feedforward network: Linear -> GELU -> Linear"""
    hidden = linear(input, w1, b1)      # [seq_len, d_ff]
    activated = gelu(hidden)            # [seq_len, d_ff]
    output = linear(activated, w2, b2)  # [seq_len, d_model]
    return output
```

This compositional design enables operation reuse—`LinearOp` is used in both FFN layers and in attention projections.

**Lowering to Linalg**. Each primitive operation lowers independently to structured Linalg operations. The lowering follows three stages for `LinearOp` and one stage for `GeluOp`.

**Stage 1: Weight Transpose**. The `LinearOp` performs `input @ weight^T + bias`, requiring weight transposition from `[out_features, in_features]` to `[in_features, out_features]` for matmul compatibility:

```cpp
// src/TransformerPasses.cpp - LinearOpLowering
struct LinearOpLowering : public OpRewritePattern<LinearOp> {
  LogicalResult matchAndRewrite(LinearOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();   // [seq_len, in_features]
    Value weight = op.getWeight(); // [out_features, in_features]
    Value bias = op.getBias();     // [out_features]

    // Transpose weight: [out_features, in_features] → [in_features, out_features]
    auto transposedType = RankedTensorType::get({inFeatures, outFeatures}, rewriter.getF32Type());
    Value transposedWeight = rewriter.create<linalg::TransposeOp>(
        loc, weight,
        rewriter.create<tensor::EmptyOp>(loc, transposedType, ...),
        SmallVector<int64_t>{1, 0}  // swap dimensions
    ).getResult()[0];
```

**Stage 2: Matrix Multiplication**. Perform `input @ transposed_weight` using `linalg.matmul` with zero-initialized output:

```cpp
    // Matmul with zero initialization
    Value zero = createConstantFloat(rewriter, loc, 0.0f);
    Value filledTensor = rewriter.create<linalg::FillOp>(loc, zero,
                           rewriter.create<tensor::EmptyOp>(loc, resultType, ...)).getResult(0);

    Value matmulResult = rewriter.create<linalg::MatmulOp>(
        loc,
        ValueRange{input, transposedWeight},
        ValueRange{filledTensor}
    ).getResult(0);  // [seq_len, out_features]
```

**Stage 3: Bias Addition with Broadcasting**. Add bias using `linalg.generic` with affine maps that broadcast `[out_features]` across `seq_len`:

```cpp
    // Bias broadcasts from [out_features] to [seq_len, out_features]
    auto identityMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
    auto biasBroadcastMap = AffineMap::get(2, 0,
        {rewriter.getAffineDimExpr(1)},  // only second dimension
        rewriter.getContext());

    Value result = rewriter.create<linalg::GenericOp>(
        loc, resultType,
        ValueRange{matmulResult, bias},
        ValueRange{rewriter.create<tensor::EmptyOp>(loc, resultType, ...)},
        SmallVector<AffineMap>{identityMap, biasBroadcastMap, identityMap},
        SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};
```

**GELU Lowering**. The `GeluOp` lowers to element-wise computation using the standard approximation $\text{GELU}(x) \approx 0.5 \cdot x \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)))$:

```cpp
struct GeluOpLowering : public OpRewritePattern<GeluOp> {
  LogicalResult matchAndRewrite(GeluOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());

    // GELU approximation constants
    Value c0_5 = createConstantFloat(rewriter, loc, 0.5f);
    Value c1 = createConstantFloat(rewriter, loc, 1.0f);
    Value cSqrt2OverPi = createConstantFloat(rewriter, loc, 0.7978845608f);
    Value c0_044715 = createConstantFloat(rewriter, loc, 0.044715f);

    // Element-wise GELU using linalg.generic
    Value result = rewriter.create<linalg::GenericOp>(
        loc, resultType,
        ValueRange{input},
        ValueRange{rewriter.create<tensor::EmptyOp>(loc, resultType, ...)},
        ...,  // Identity maps, parallel iterators
        [c0_5, c1, cSqrt2OverPi, c0_044715](OpBuilder &b, Location loc, ValueRange args) {
          Value x = args[0];
          // GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
          Value x2 = b.create<arith::MulFOp>(loc, x, x);
          Value x3 = b.create<arith::MulFOp>(loc, x2, x);
          Value term = b.create<arith::MulFOp>(loc, c0_044715, x3);
          Value inner = b.create<arith::AddFOp>(loc, x, term);
          Value scaled = b.create<arith::MulFOp>(loc, cSqrt2OverPi, inner);
          Value tanhVal = b.create<math::TanhOp>(loc, scaled);
          Value onePlusTanh = b.create<arith::AddFOp>(loc, c1, tanhVal);
          Value halfX = b.create<arith::MulFOp>(loc, c0_5, x);
          Value geluResult = b.create<arith::MulFOp>(loc, halfX, onePlusTanh);
          b.create<linalg::YieldOp>(loc, geluResult);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};
```

The complete FFN operation composes these building blocks:

```python
# High-level FFN API
ffn_out = ch12.ffn(input, W1, b1, W2, b2)

# Lowers to:
hidden = ch12.linear(input, W1, b1)     # → linalg ops
activated = ch12.gelu(hidden)            # → linalg.generic
output = ch12.linear(activated, W2, b2)  # → linalg ops

# After bufferization pipeline, these become efficient memref operations
```

**Why This Design Works**. Breaking FFN into composable operations (`linear`, `gelu`) rather than monolithic lowering provides:

1. **Reusability**: `LinearOp` is used twice in FFN but also in attention projections (Q/K/V/O matrices)—single implementation serves multiple use cases
2. **Optimization opportunities**: Separate operations enable per-operation optimizations (matrix multiplication tiling, GELU fusion) before composition-level optimizations (fusing linear+GELU)
3. **Maintenance**: Updating linear layer implementation (e.g., adding quantization) automatically benefits all users

The Linalg-based implementation enables Chapter 14's production optimizations: `linalg.matmul` tiles efficiently for cache locality, `linalg.generic` vectorizes GELU computation, and fusion passes combine operations to reduce memory traffic.

**Performance Analysis**. Feedforward networks dominate transformer computation for typical sequence lengths—they account for approximately **2/3 of total FLOPs** in standard architectures at common context lengths. For input `[seq_len, d_model]` with `d_ff = 4 × d_model`, counting Multiply-Accumulate (MAC) operations as 2 FLOPs each:

- **Linear_1 FLOPs**: `2 × seq_len × d_model × d_ff = 2 × seq_len × d_model × (4 × d_model) = 8 × seq_len × d_model²`
- **GELU FLOPs**: `seq_len × d_ff = 4 × seq_len × d_model` (negligible: ~0.05% of FFN compute)
- **Linear_2 FLOPs**: `2 × seq_len × d_ff × d_model = 2 × seq_len × (4 × d_model) × d_model = 8 × seq_len × d_model²`
- **Total FFN FLOPs**: `16 × seq_len × d_model²`

Compare to attention: Q/K/V/O projections contribute `8 × seq_len × d_model²`, while the attention computation itself (QK^T scores and attention@V) adds `4 × seq_len² × d_model`, totaling roughly `8 × seq_len × d_model² + 4 × seq_len² × d_model`. The crossover point occurs around `seq_len ≈ 2 × d_model`. For GPT-2 (`d_model = 768`, context length 1024), FFN accounts for ~65% of compute. For very long sequences (`seq_len >> d_model`), the quadratic attention term dominates.

**Why Fusion Matters**. In production, FFNs are often **memory-bound** rather than compute-bound—memory bandwidth (reading weights, writing activations) limits throughput more than arithmetic units. Fusing GELU with the preceding matmul eliminates an intermediate buffer, reducing memory traffic by ~25% for that operation. This is why Chapter 14's production optimizations focus on operator fusion (computing GELU inline as matmul writes outputs) and quantization (reducing memory bandwidth via INT8/FP16 weights) rather than just arithmetic optimization.

**Testing Feedforward Networks**. Numerical validation against NumPy reference implementation:

```python
# test_jit.py - Test 5: FFN
d_model, d_ff = 4, 8
x = np.random.randn(2, d_model).astype(np.float32)
w1 = np.random.randn(d_ff, d_model).astype(np.float32)
b1 = np.random.randn(d_ff).astype(np.float32)
w2 = np.random.randn(d_model, d_ff).astype(np.float32)
b2 = np.random.randn(d_model).astype(np.float32)

# MLIR implementation
x_tensor = ch12.Tensor(x)
output = ch12.ffn(x_tensor, w1, b1, w2, b2)
result = ch12.forward(output)

# NumPy reference: FFN(x) = Linear(GELU(Linear(x)))
def gelu_ref(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

hidden = x @ w1.T + b1
activated = gelu_ref(hidden)
expected = activated @ w2.T + b2

np.testing.assert_allclose(result, expected, rtol=1e-4)
print("✓ FFN test PASSED!")
```

The test validates the complete FFN pipeline: both linear layers with bias addition and GELU activation. The tolerance (1e-4 relative) accounts for floating-point accumulated errors across two matrix multiplications and the GELU approximation. Chapter 12 tests all components individually (linear, GELU, layer_norm) before testing composite operations (FFN, attention, transformer_block).

**Fusion Opportunities**. Feedforward networks present excellent fusion opportunities, and Chapter 12 applies the optimization techniques learned in Chapter 10:

1. **Bias addition fusion**: Merge bias addition into matmul (common in BLAS libraries)
2. **GELU fusion**: Compute GELU inline during matmul output (eliminates intermediate buffer)
3. **Layer norm fusion**: Combine adjacent element-wise operations

Chapter 12's compilation pipeline includes Linalg fusion passes (from Chapter 10) that automatically merge adjacent operations:

```cpp
// Optimization pipeline (bindings.cpp)
pm.addPass(createLinalgGeneralizeNamedOpsPass());    // Convert to generic form
pm.addPass(createCanonicalizerPass());               // Cleanup
pm.addPass(createLinalgElementwiseOpFusionPass());   // Fuse adjacent operations
pm.addPass(createCanonicalizerPass());               // Cleanup again
// ... bufferization ...
pm.addPass(createLoopInvariantCodeMotionPass());     // Hoist constants
```

For example, fusion combines bias addition with matmul:

```mlir
// Before fusion
%hidden = linalg.matmul ins(%input, %w1) outs(%init)
%biased = linalg.generic ... ins(%hidden, %b1) ...  // Separate bias addition

// After fusion
%biased = linalg.generic {
  // Combined matmul + bias in single loop nest
} ins(%input, %w1, %b1) outs(%init)
```

This fusion eliminates intermediate buffer traffic (`%hidden`), improving memory locality. The optimization passes also apply loop invariant code motion (hoisting constants like epsilon, sqrt(d_k) outside loops) and common subexpression elimination. These optimizations happen automatically without changing the high-level Python API—users call `transformer_block()`, and MLIR applies Chapter 10's optimizations transparently. Chapter 14 will add production-grade optimizations (Transform dialect tiling/vectorization, KV caching) building on these foundations.

Feedforward networks complete the second major component of transformer blocks. Combined with layer normalization (Section 12.2) and attention (Chapter 11), we're ready to compose these pieces into the full transformer block architecture (next section).

## 12.4 Composing the Transformer Block

With layer normalization, attention, and feedforward networks implemented individually, we now compose them into a complete transformer block. This section demonstrates how residual connections integrate the components, explores parameter management and testing strategies, and shows how MLIR's optimization passes work transparently across the composed operations.

**The Full Composition**. A transformer block combines all components with specific ordering:

```python
def transformer_block(x, params):
    # Sub-layer 1: Multi-head attention with residual
    normed1 = layernorm(x, params['ln1_gamma'], params['ln1_beta'])
    attn_out = multi_head_attention(normed1, params['attn_w*'])
    x = x + attn_out  # Residual connection

    # Sub-layer 2: Feedforward with residual
    normed2 = layernorm(x, params['ln2_gamma'], params['ln2_beta'])
    ffn_out = feedforward(normed2, params['ffn_w*'])
    x = x + ffn_out  # Residual connection

    return x
```

This follows the pre-normalization pattern: normalize before each sub-layer, then add the sub-layer output via residual connection. The ordering is critical—post-norm (normalizing after residual addition) has different training dynamics and is less stable for deep networks.

**Parameter Dimensions**. For a transformer block with `d_model = 512`, `d_ff = 2048` (4× expansion), `num_heads = 8`:

```
Layer Norm 1:  gamma [512] + beta [512]                      = 1,024
Attention:     W_q/k/v/o [512, 512] × 4                      = 1,048,576
               (Note: biases omitted per Chapter 11 design)
Layer Norm 2:  gamma [512] + beta [512]                      = 1,024
Feedforward:   W_1 [512, 2048] = 1,048,576
               b_1 [2048] = 2,048
               W_2 [2048, 512] = 1,048,576
               b_2 [512] = 512
               (FFN subtotal: 2,099,712)
──────────────────────────────────────────────────────────────────────
Total:                                                         3,150,336
```

Approximately **3.15 million parameters per transformer block**. GPT-2 small (12 blocks) has ~38 million parameters in transformer blocks alone (plus embeddings).

**Residual Connections and Gradient Flow**. Residuals provide direct gradient paths during training:

$$\frac{\partial \text{Loss}}{\partial x} = \frac{\partial \text{Loss}}{\partial \text{output}} \times \left(1 + \frac{\partial \text{sub\_layer}}{\partial x}\right)$$

The `+1` term is the identity path—even if gradients vanish through the sub-layer, they remain non-zero via the residual. This prevents the vanishing gradient problem in deep networks. Residuals require matching shapes: attention and feedforward both map `[seq_len, d_model] → [seq_len, d_model]`.

**Composable Design Philosophy**. Rather than monolithic `Transformer_BlockOp` and `Transformer_FFNOp` dialect operations, Chapter 12 composes primitive operations (`layer_norm`, `linear`, `gelu`, `add`, `attention`). Benefits:

1. **Reusability**: `linear` used in FFN, attention projections, and output layers
2. **Independent optimization**: Each primitive optimized separately before composition-level optimizations
3. **Flexibility**: Easy to swap components (SwiGLU vs GELU, RMSNorm vs LayerNorm)
4. **Maintainability**: Single source of truth for each operation type

The dialect provides **building blocks** (8 primitive operations), and the Python API provides **compositions** (ffn, attention, transformer_block).

**Applied Optimizations**. MLIR's passes from Chapter 10 work transparently across transformer blocks:

- **Linalg Fusion**: Fuses adjacent operations (layer norm's variance computation with normalization, scaling with QK^T)
- **Loop Invariant Code Motion**: Hoists constants (`epsilon`, `sqrt(d_k)`) outside loops
- **Common Subexpression Elimination**: Removes redundant computations

These apply automatically—no transformer-specific optimization passes needed. The composable pass infrastructure optimizes lowered IR regardless of high-level origin.

**Testing and Validation**. Chapter 12's test suite ([test_jit.py](ch.12.Transformer/test_jit.py)) validates each component individually before testing composition:

```python
# Test 2: LayerNorm
x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
gamma = np.ones(4, dtype=np.float32)
beta = np.zeros(4, dtype=np.float32)

output = ch12.layer_norm(ch12.Tensor(x), gamma, beta)
result = ch12.forward(output)

# NumPy reference
mean = np.mean(x, axis=1, keepdims=True)
var = np.var(x, axis=1, keepdims=True)  # Biased variance (divide by N)
expected = (x - mean) / np.sqrt(var + 1e-5)

np.testing.assert_allclose(result, expected, atol=1e-5)
print("✓ LayerNorm test PASSED!")
```

**Note**: `np.var` uses biased variance by default (divides by N, not N-1), matching standard LayerNorm implementations.

Tests validate:
- Individual operations (layernorm, linear, gelu, matmul, transpose, softmax, scale, add, attention)
- Composite operations (ffn, transformer_block)
- Multi-block stacking (verifying no NaN/inf propagation)

**Common Debugging Issues**:

1. **NaN Propagation**: Layer norm with zero variance → divide-by-zero → NaN. Fixed by epsilon term (1e-5).
2. **Shape Mismatches**: Residual addition requires matching shapes. Ensure all operations preserve `[seq_len, d_model]`.
3. **Parameter Initialization**: Large random values → exploding activations. Use scaled initialization (0.02 std dev).

**Performance Characteristics**. For typical configurations (seq_len=512, d_model=768):
- **Feedforward**: ~65% of FLOPs, compute-bound (16 × seq_len × d_model²)
- **Attention**: ~30% of FLOPs, memory-bound due to softmax and attention score materialization (8 × seq_len × d_model² + 4 × seq_len² × d_model)
- **Layer Norms**: ~5% of FLOPs, negligible

The transformer block is now complete. Chapter 13 will stack multiple blocks, add positional embeddings and output layers, and implement GPT-style autoregressive generation. Chapter 14 will optimize this architecture with Transform dialect techniques (tiling, vectorization), KV caching, and advanced lowering patterns for production serving.

## 12.5 Summary

Chapter 12 assembled complete transformer blocks by composing layer normalization, multi-head attention, and feedforward networks with residual connections. We implemented primitive operations as Transformer dialect operations with memref-based semantics, defined lowering patterns to Linalg operations for optimization, and composed these primitives into higher-level functions at the Python API level.

Key insights:

- **Compositional Design**: Transformer blocks compose from 8 primitive dialect operations (layer_norm, linear, gelu, matmul, transpose, softmax, scale, add), with higher-level abstractions (ffn, attention, transformer_block) implemented as Python API functions. Each primitive is optimizable separately yet works together efficiently.
- **Pre-Normalization Architecture**: Modern transformers normalize before sub-layers (pre-norm) rather than after (post-norm), improving training stability for deep networks
- **Residual Connections**: Skip connections enable gradient flow through deep architectures, preventing vanishing gradients that plagued earlier deep learning
- **Linalg-Based Lowering**: All operations lower to structured Linalg operations (linalg.reduce, linalg.generic, linalg.matmul), enabling MLIR's optimization passes to apply fusion, vectorization, and loop tiling automatically

Chapter 12 established transformer blocks as reusable, testable components with ~3 million parameters each. These blocks are the fundamental units of models like GPT-2 (12 blocks), GPT-3 (96 blocks), and LLaMA (32-80 blocks depending on variant).

**Looking Ahead**. Chapter 13 builds complete GPT architecture by stacking transformer blocks, adding token/positional embeddings, implementing causal masking for autoregressive generation, and developing text generation capabilities. We'll see how transformer blocks replicate throughout the model, differentiated only by learned parameters. Chapter 14 then optimizes this architecture for production serving with Transform dialect optimizations, KV caching for efficient autoregressive inference, and advanced compilation techniques—achieving significant performance improvements for production deployment.