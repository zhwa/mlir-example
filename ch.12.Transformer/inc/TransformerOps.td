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
    %result = transformer.layer_norm %input, %gamma, %beta
      : (tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
    ```
    
    Input shape: [seq_len, d_model]
    Gamma/Beta shape: [d_model]
    Result shape: [seq_len, d_model]
  }];
  
  let arguments = (ins 
    AnyTensor:$input,
    AnyTensor:$gamma,    // Scale parameter
    AnyTensor:$beta      // Shift parameter
    // NOTE: epsilon handled as constant in lowering for numerical stability
  );
  
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $gamma `,` $beta
    attr-dict `:` `(` type($input) `,` type($gamma) `,` type($beta) `)` `->` type($result)
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
    %result = transformer.linear %input, %weight, %bias
      : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
    ```
    
    Input shape: [seq_len, in_features]
    Weight shape: [out_features, in_features]
    Bias shape: [out_features]
    Result shape: [seq_len, out_features]
  }];
  
  let arguments = (ins 
    AnyTensor:$input,
    AnyTensor:$weight,
    AnyTensor:$bias
  );
  
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $weight `,` $bias
    attr-dict `:` `(` type($input) `,` type($weight) `,` type($bias) `)` `->` type($result)
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
    %result = transformer.gelu %input : tensor<?x?xf32> -> tensor<?x?xf32>
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
    Performs matrix multiplication: result = lhs @ rhs
    
    Supports 2D and 3D tensors (batched matmul).
    For 2D: (M, K) @ (K, N) -> (M, N)
    For 3D: (B, M, K) @ (B, K, N) -> (B, M, N)
  }];
  
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

def Transformer_TransposeOp : Transformer_Op<"transpose"> {
  let summary = "Matrix transpose";
  let description = [{
    Transposes the last two dimensions of a tensor.
    
    For 2D: (M, N) -> (N, M)
    For 3D: (B, M, N) -> (B, N, M)
    
    Example:
    ```mlir
    %result = transformer.transpose %input : tensor<?x?xf32> -> tensor<?x?xf32>
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
    %result = transformer.softmax %input : tensor<?x?xf32> -> tensor<?x?xf32>
    ```
  }];
  
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Transformer_ScaleOp : Transformer_Op<"scale"> {
  let summary = "Scale tensor by constant";
  let description = [{
    Multiplies all elements by a scalar: result = input * scale
    
    Used for scaling attention scores by 1/sqrt(d_k).
    Scale can be a scalar tensor or broadcasted tensor.
  }];
  
  let arguments = (ins AnyTensor:$input, AnyTensor:$scale);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input `,` $scale attr-dict `:` `(` type($input) `,` type($scale) `)` `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Utility Operations
//===----------------------------------------------------------------------===//

def Transformer_AddOp : Transformer_Op<"add"> {
  let summary = "Element-wise addition";
  let description = [{
    Computes element-wise addition: result = lhs + rhs
    Used for residual connections in transformer blocks.
    
    Example:
    ```mlir
    %result = transformer.add %lhs, %rhs : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    ```
  }];
  
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

#endif // TRANSFORMER_OPS
