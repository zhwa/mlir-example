//===- NNOps.td - NN dialect operations -------------------*- tablegen -*-===//
//
// Chapter 9: Neural Network Operations
//
//===----------------------------------------------------------------------===//

#ifndef NN_OPS
#define NN_OPS

include "NNDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Element-wise Operations
//===----------------------------------------------------------------------===//

def NN_AddOp : NN_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "element-wise addition";
  let description = [{
    Performs element-wise addition of two tensors: `result = lhs + rhs`

    Example:
    ```mlir
    %result = nn.add %lhs, %rhs : tensor<4xf32>
    ```
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def NN_MulOp : NN_Op<"mul", [Pure, SameOperandsAndResultType]> {
  let summary = "element-wise multiplication";
  let description = [{
    Performs element-wise multiplication: `result = lhs * rhs`

    Example:
    ```mlir
    %result = nn.mul %lhs, %rhs : tensor<4xf32>
    ```
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Matrix Operations
//===----------------------------------------------------------------------===//

def NN_MatMulOp : NN_Op<"matmul", [Pure]> {
  let summary = "matrix multiplication";
  let description = [{
    Performs matrix multiplication: `C = A @ B`

    Supports 2D matrices with compatible shapes:
    - A: [M, K]
    - B: [K, N]
    - C: [M, N]

    Example:
    ```mlir
    %c = nn.matmul %a, %b : tensor<2x3xf32>, tensor<3x4xf32> -> tensor<2x4xf32>
    ```
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Activation Functions
//===----------------------------------------------------------------------===//

def NN_ReLUOp : NN_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation function";
  let description = [{
    Applies the Rectified Linear Unit activation function:
    `result = max(0, input)`

    Example:
    ```mlir
    %output = nn.relu %input : tensor<2x4xf32>
    ```
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($result)";
}

def NN_SoftmaxOp : NN_Op<"softmax", [Pure, SameOperandsAndResultType]> {
  let summary = "Softmax activation function";
  let description = [{
    Applies the softmax function along the last dimension:
    `output[i] = exp(input[i]) / sum(exp(input[j]))`

    Example:
    ```mlir
    %probs = nn.softmax %logits : tensor<2x4xf32>
    ```
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Linear Layer
//===----------------------------------------------------------------------===//

def NN_LinearOp : NN_Op<"linear", [Pure]> {
  let summary = "Linear layer (fully connected)";
  let description = [{
    Performs a linear transformation with optional bias:
    `output = input @ weight^T + bias`

    Arguments:
    - input: [batch_size, in_features]
    - weight: [out_features, in_features]
    - bias: [out_features] (optional)

    Example:
    ```mlir
    %output = nn.linear %input, %weight, %bias 
        : tensor<2x3xf32>, tensor<4x3xf32>, tensor<4xf32> -> tensor<2x4xf32>
    ```
  }];

  let arguments = (ins 
    AnyTensor:$input,
    AnyTensor:$weight,
    Optional<AnyTensor>:$bias
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict 
    `:` type($input) `,` type($weight) (`,` type($bias)^)? `->` type($result)
  }];
}

#endif // NN_OPS