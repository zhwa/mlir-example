//===- TransformerOps.td - Transformer operations ---------*- tablegen -*-===//
//
// Defines operations for transformer attention mechanisms.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMER_OPS
#define TRANSFORMER_OPS

include "TransformerDialect.td"

//===----------------------------------------------------------------------===//
// Matrix Multiplication (same as nn.matmul from Chapter 9)
//===----------------------------------------------------------------------===//

def Transformer_MatmulOp : Transformer_Op<"matmul"> {
  let summary = "Matrix multiplication";
  let description = [{
    Performs matrix multiplication: C = A @ B

    Supports 2D and 3D tensors (batched matmul).
    For 2D: (M, K) @ (K, N) -> (M, N)
    For 3D: (B, M, K) @ (B, K, N) -> (B, M, N)
  }];

  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);
  let assemblyFormat = "$lhs `,` $rhs `,` $output attr-dict `:` type($lhs) `,` type($rhs) `,` type($output)";
}

//===----------------------------------------------------------------------===//
// Element-wise Operations
//===----------------------------------------------------------------------===//

def Transformer_AddOp : Transformer_Op<"add"> {
  let summary = "Element-wise addition";
  let description = [{ output = lhs + rhs }];

  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);
  let assemblyFormat = "$lhs `,` $rhs `,` $output attr-dict `:` type($lhs) `,` type($rhs) `,` type($output)";
}

def Transformer_MulOp : Transformer_Op<"mul"> {
  let summary = "Element-wise multiplication";
  let description = [{ output = lhs * rhs }];

  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);
  let assemblyFormat = "$lhs `,` $rhs `,` $output attr-dict `:` type($lhs) `,` type($rhs) `,` type($output)";
}

//===----------------------------------------------------------------------===//
// Softmax
//===----------------------------------------------------------------------===//

def Transformer_SoftmaxOp : Transformer_Op<"softmax"> {
  let summary = "Softmax activation";
  let description = [{
    Applies softmax along the last dimension with numerical stability.

    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
  }];

  let arguments = (ins AnyMemRef:$input, AnyMemRef:$output);
  let assemblyFormat = "$input `,` $output attr-dict `:` type($input) `,` type($output)";
}

//===----------------------------------------------------------------------===//
// Transpose
//===----------------------------------------------------------------------===//

def Transformer_TransposeOp : Transformer_Op<"transpose"> {
  let summary = "Transpose last two dimensions";
  let description = [{
    Transposes the last two dimensions of a tensor.

    For 2D: (M, N) -> (N, M)
    For 3D: (B, M, N) -> (B, N, M)
  }];

  let arguments = (ins AnyMemRef:$input, AnyMemRef:$output);
  let assemblyFormat = "$input `,` $output attr-dict `:` type($input) `,` type($output)";
}

//===----------------------------------------------------------------------===//
// Scaled Dot-Product Attention (Core Operation)
//===----------------------------------------------------------------------===//

def Transformer_AttentionOp : Transformer_Op<"attention"> {
  let summary = "Scaled dot-product attention";
  let description = [{
    Computes scaled dot-product attention:

    scores = (Q @ K^T) / sqrt(head_dim)
    attn_weights = softmax(scores)
    output = attn_weights @ V

    For multi-head attention, the input is expected to be already split into heads:
    Input shape: (batch, seq_len, d_model)

    If Q/K/V projections are provided (w_q, w_k, w_v), applies:
      Q = input @ w_q^T
      K = input @ w_k^T  
      V = input @ w_v^T
    Otherwise uses self-attention (Q = K = V = input)

    Outputs concatenated heads, optionally applies output projection (w_o).

    Note: num_heads and head_dim are inferred from weight dimensions to avoid
    TableGen BytecodeOpInterface issues in LLVM 19.
  }];

  let arguments = (ins 
    AnyMemRef:$input,
    AnyMemRef:$output,
    AnyMemRef:$w_q,
    AnyMemRef:$w_k,
    AnyMemRef:$w_v,
    AnyMemRef:$w_o
  );

  let assemblyFormat = [{
    `(` $input `,` $output `,` $w_q `,` $w_k `,` $w_v `,` $w_o `)`
    attr-dict `:` type($input) `,` type($output) `,` type($w_q) `,` type($w_k) `,` type($w_v) `,` type($w_o)
  }];
}

#endif // TRANSFORMER_OPS