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

#endif // TRANSFORMER_OPS