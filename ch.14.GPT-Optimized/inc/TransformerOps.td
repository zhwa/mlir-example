//===- TransformerOps.td - Transformer operations ---------*- tablegen -*-===//
//
// Chapter 12: Transformer Block Operations
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMER_OPS
#define TRANSFORMER_OPS

include "TransformerDialect.td"

//===----------------------------------------------------------------------===//
// Phase 1: Layer Normalization
//===----------------------------------------------------------------------===//

def Transformer_LayerNormOp : Transformer_Op<"layer_norm"> {
  let summary = "Layer normalization operation";
  let description = [{
    Normalizes input across the last dimension (embedding dimension):
    
    mean = mean(input, axis=-1)
    variance = var(input, axis=-1)
    normalized = (input - mean) / sqrt(variance + epsilon)
    output = normalized * gamma + beta
    
    Where gamma and beta are learnable parameters (scale and shift).
    Epsilon is a small constant for numerical stability (typically 1e-5).
    
    Example:
    ```mlir
    transformer.layer_norm %input, %gamma, %beta, %output, %epsilon
      : memref<?x?xf32>, memref<?xf32>, memref<?xf32>, memref<?x?xf32>, f32
    ```
    
    Input shape: [seq_len, d_model]
    Gamma/Beta shape: [d_model]
    Output shape: [seq_len, d_model]
  }];
  
  let arguments = (ins 
    AnyTensor:$input,
    AnyTensor:$gamma,    // Scale parameter
    AnyTensor:$beta      // Shift parameter
    // NOTE: epsilon removed - pass as runtime value to avoid BytecodeOpInterface issues
  );
  
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $gamma `,` $beta
    attr-dict `:` type($input) `,` type($gamma) `,` type($beta) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Phase 2: Feed-Forward Network Operations
//===----------------------------------------------------------------------===//

def Transformer_LinearOp : Transformer_Op<"linear"> {
  let summary = "Linear transformation with bias";
  let description = [{
    Computes: output = input @ weight^T + bias
    
    Example:
    ```mlir
    transformer.linear %input, %weight, %bias, %output
      : memref<?x?xf32>, memref<?x?xf32>, memref<?xf32>, memref<?x?xf32>
    ```
    
    Input shape: [seq_len, in_features]
    Weight shape: [out_features, in_features]
    Bias shape: [out_features]
    Output shape: [seq_len, out_features]
  }];
  
  let arguments = (ins 
    AnyTensor:$input,
    AnyTensor:$weight,
    AnyTensor:$bias
  );
  
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $weight `,` $bias
    attr-dict `:` type($input) `,` type($weight) `,` type($bias) `->` type($result)
  }];
}

def Transformer_GeluOp : Transformer_Op<"gelu"> {
  let summary = "Gaussian Error Linear Unit activation";
  let description = [{
    Applies GELU activation: GELU(x) = x * Φ(x)
    where Φ(x) is the cumulative distribution function of the standard normal distribution.
    
    Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    
    Example:
    ```mlir
    transformer.gelu %input : tensor<?x?xf32> -> tensor<?x?xf32>
    ```
  }];
  
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Phase 3: Attention Operations
//===----------------------------------------------------------------------===//


def Transformer_MatmulOp : Transformer_Op<"matmul"> {
  let summary = "Matrix multiplication";
  let description = [{
    Performs matrix multiplication: output = lhs @ rhs
    
    Supports 2D and 3D tensors (batched matmul).
    For 2D: (M, K) @ (K, N) -> (M, N)
    For 3D: (B, M, K) @ (B, K, N) -> (B, M, N)
  }];
  
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Transformer_TransposeOp : Transformer_Op<"transpose"> {
  let summary = "Matrix transpose";
  let description = [{
    Transposes the last two dimensions of a tensor.
    
    For 2D: (M, N) -> (N, M)
    For 3D: (B, M, N) -> (B, N, M)
    
    Example:
    ```mlir
    transformer.transpose %input : tensor<?x?xf32> -> tensor<?x?xf32>
    ```
  }];
  
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Transformer_SoftmaxOp : Transformer_Op<"softmax"> {
  let summary = "Softmax activation";
  let description = [{
    Applies softmax along the last dimension.
    
    Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    
    For numerical stability, uses the max trick:
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    
    Example:
    ```mlir
    transformer.softmax %input, %output : memref<?x?xf32>, memref<?x?xf32>
    ```
  }];
  
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Transformer_ScaleOp : Transformer_Op<"scale"> {
  let summary = "Scale tensor by constant";
  let description = [{
    Multiplies all elements by a scalar: output = input * scale
    
    Used for scaling attention scores by 1/sqrt(d_k).
    
    Note: scale passed as memref to avoid attribute/BytecodeOpInterface issues.
  }];
  
  let arguments = (ins AnyTensor:$input, AnyTensor:$scale);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input `,` $scale attr-dict `:` type($input) `,` type($scale) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Chapter 13: GPT-specific Operations
//===----------------------------------------------------------------------===//

def Transformer_EmbeddingOp : Transformer_Op<"embedding"> {
  let summary = "Token embedding lookup";
  let description = [{
    Performs embedding lookup for token IDs.
    
    For each token ID in the input sequence, retrieves the corresponding
    embedding vector from the embedding table.
    
    Example:
    ```mlir
    transformer.embedding %indices, %table, %output
      : memref<?xi32>, memref<?x?xf32>, memref<?x?xf32>
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

//===----------------------------------------------------------------------===//
// Causal Masking Operations (Chapter 13)
//===----------------------------------------------------------------------===//

def Transformer_CreateCausalMaskOp : Transformer_Op<"create_causal_mask"> {
  let summary = "Create causal (lower triangular) attention mask";
  let description = [{
    Creates a causal attention mask for autoregressive generation.
    The mask is lower triangular: positions can only attend to previous
    positions (including themselves), not future positions.
    
    Mask values:
    - mask[i][j] = 0.0 if j <= i (can attend)
    - mask[i][j] = -inf if j > i (cannot attend to future)
    
    When added to attention logits before softmax, -inf ensures
    softmax outputs 0 for future positions.
    
    Example:
    ```mlir
    transformer.create_causal_mask : tensor<4x4xf32>
    // Output:
    // [  0.0  -inf  -inf  -inf ]
    // [  0.0   0.0  -inf  -inf ]
    // [  0.0   0.0   0.0  -inf ]
    // [  0.0   0.0   0.0   0.0 ]
    ```
  }];
  
  let arguments = (ins);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

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
    transformer.masked_softmax %logits, %mask, %output 
      : memref<2x4x4xf32>, memref<4x4xf32>, memref<2x4x4xf32>
    ```
  }];
  
  let arguments = (ins AnyTensor:$input, AnyTensor:$mask);
  
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $mask
    attr-dict `:` type($input) `,` type($mask) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Rotary Position Embeddings (Chapter 13)
//===----------------------------------------------------------------------===//

def Transformer_RoPEOp : Transformer_Op<"rope"> {
  let summary = "Apply Rotary Position Embeddings (RoPE)";
  let description = [{
    Applies rotary position embeddings to input tensor (typically Q or K in attention).
    RoPE encodes position information by rotating pairs of dimensions.
    
    For each position i and dimension pair (2j, 2j+1):
    - θ_j = base^(-2j/d) where base=10000, d=head_dim
    - output[i, 2j]   = input[i, 2j]   * cos(i*θ_j) - input[i, 2j+1] * sin(i*θ_j)
    - output[i, 2j+1] = input[i, 2j]   * sin(i*θ_j) + input[i, 2j+1] * cos(i*θ_j)
    
    This rotation preserves relative position information and enables length extrapolation.
    
    Input shape: [seq_len, d_model] or [batch, seq_len, d_model]
    Output shape: same as input
    
    Example:
    ```mlir
    transformer.rope %input : tensor<32x64xf32> -> tensor<32x64xf32>
    ```
    
    References:
    - RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
    - Used in LLaMA, GPT-NeoX, and other modern LLMs
  }];
  
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Utility Operations
//===----------------------------------------------------------------------===//

def Transformer_AddOp : Transformer_Op<"add"> {
  let summary = "Element-wise addition";
  let description = [{
    Computes element-wise addition: output = lhs + rhs
    Used for residual connections in transformer blocks.
    
    Example:
    ```mlir
    transformer.add %lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
    ```
  }];
  
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";

}

#endif // TRANSFORMER_OPS
