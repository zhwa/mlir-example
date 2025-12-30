//===- ir.cpp - MLIR IR Generation for Softmax --------------------------===//
//
// This file demonstrates Softmax using tensor-first architecture with Linalg
// dialect for reductions and element-wise operations.
//
// Operation: output[i] = exp(input[i] - max) / sum(exp(input[j] - max))
//
// Key Learning Points:
//   - Tensor-first architecture with functional semantics
//   - linalg.reduce for finding maximum and computing sum
//   - linalg.generic for element-wise operations (exp, normalization)
//   - Numerical stability (subtract max before exp)
//   - Multi-pass algorithm with tensors (not imperative loops)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include <limits>

namespace mlir {

/// Creates a Softmax module using tensor-first architecture.
///
/// Algorithm (numerically stable, using tensors):
///   Pass 1: Find maximum value using linalg.reduce
///   Pass 2: Compute exp(x - max) using linalg.generic
///   Pass 3: Compute sum of exp values using linalg.reduce
///   Pass 4: Normalize by dividing by sum using linalg.generic
///
/// Generated MLIR:
///   func.func @softmax(%input: tensor<?xf32>) -> tensor<?xf32> {
///     // Pass 1: Find max using linalg.reduce with arith.maximumf
///     %init_max = tensor.empty() : tensor<f32>
///     %max_tensor = linalg.reduce { arith.maximumf }
///                   ins(%input : tensor<?xf32>)
///                   outs(%init_max : tensor<f32>)
///     %max = tensor.extract %max_tensor[] : tensor<f32>
///
///     // Pass 2: Compute exp(x - max) using linalg.generic
///     %exp_tensor = linalg.generic {
///       %val = linalg.index 0
///       %shifted = arith.subf %val, %max
///       %exp_val = math.exp %shifted
///       linalg.yield %exp_val
///     } ins(%input) outs(%empty)
///
///     // Pass 3: Sum exp values using linalg.reduce with arith.addf
///     %init_sum = tensor.empty() : tensor<f32>
///     %sum_tensor = linalg.reduce { arith.addf }
///                   ins(%exp_tensor : tensor<?xf32>)
///                   outs(%init_sum : tensor<f32>)
///     %sum = tensor.extract %sum_tensor[] : tensor<f32>
///
///     // Pass 4: Normalize using linalg.generic
///     %result = linalg.generic {
///       %exp_val = linalg.index 0
///       %normalized = arith.divf %exp_val, %sum
///       linalg.yield %normalized
///     } ins(%exp_tensor) outs(%empty)
///
///     return %result
///   }
OwningOpRef<ModuleOp> createSoftmaxModule(MLIRContext& context) {
  // Load required dialects
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();
  context.getOrLoadDialect<linalg::LinalgDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<math::MathDialect>();

  // Create builder and module
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module.getBody());

  // Define types
  auto f32Type = builder.getF32Type();
  auto indexType = builder.getIndexType();

  // Dynamic 1D tensor: tensor<?xf32>
  auto dynamicTensorType = RankedTensorType::get({ShapedType::kDynamic}, f32Type);
  
  // Scalar tensor for reductions: tensor<f32>
  auto scalarTensorType = RankedTensorType::get({}, f32Type);

  // Function type: (tensor<?xf32>) -> tensor<?xf32>
  auto funcType = builder.getFunctionType({dynamicTensorType}, {dynamicTensorType});

  // Create function
  auto funcOp = builder.create<func::FuncOp>(loc, "softmax", funcType);
  funcOp.setPublic();

  // Create function body
  auto& entryBlock = *funcOp.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // Get function argument
  Value input = entryBlock.getArgument(0);

  // Get size of input tensor
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value size = builder.create<tensor::DimOp>(loc, input, c0);

  //===--------------------------------------------------------------------===//
  // Pass 1: Find maximum value using linalg.reduce
  //===--------------------------------------------------------------------===//

  // Initialize scalar tensor with negative infinity
  Value negInf = builder.create<arith::ConstantOp>(
      loc, builder.getFloatAttr(f32Type, 
          APFloat::getInf(f32Type.getFloatSemantics(), /*Negative=*/true)));
  
  // Create tensor<f32> initialized with -inf
  Value initMaxTensor = builder.create<tensor::FromElementsOp>(
      loc, scalarTensorType, ValueRange{negInf});

  // Create linalg.reduce to find maximum
  auto reduceMaxOp = builder.create<linalg::ReduceOp>(
      loc, input, initMaxTensor,
      SmallVector<int64_t>{0},  // reduce along dimension 0
      [&](OpBuilder& b, Location loc, ValueRange args) {
        // args[0]: current element from input
        // args[1]: current accumulator (max so far)
        Value newMax = b.create<arith::MaximumFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, newMax);
      }
  );
  Value maxTensor = reduceMaxOp.getResult(0);
  
  // Extract scalar value from tensor<f32>
  Value maxVal = builder.create<tensor::ExtractOp>(loc, maxTensor, ValueRange{});

  //===--------------------------------------------------------------------===//
  // Pass 2: Compute exp(x - max) using linalg.generic
  //===--------------------------------------------------------------------===//

  // Create empty output tensor
  SmallVector<OpFoldResult> dynamicSizes = {
      builder.create<tensor::DimOp>(loc, input, c0).getResult()
  };
  Value emptyTensor = builder.create<tensor::EmptyOp>(
      loc, dynamicSizes, f32Type);

  // Affine maps for element-wise operation
  auto identityMap = builder.getMultiDimIdentityMap(1);
  SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes = {utils::IteratorType::parallel};

  // linalg.generic: compute exp(input[i] - max)
  auto expOp = builder.create<linalg::GenericOp>(
      loc, dynamicTensorType, input, emptyTensor,
      indexingMaps, iteratorTypes,
      [&](OpBuilder& b, Location loc, ValueRange args) {
        // args[0]: input element
        Value shifted = b.create<arith::SubFOp>(loc, args[0], maxVal);
        Value expVal = b.create<math::ExpOp>(loc, shifted);
        b.create<linalg::YieldOp>(loc, expVal);
      }
  );
  Value expTensor = expOp.getResult(0);

  //===--------------------------------------------------------------------===//
  // Pass 3: Compute sum of exp values using linalg.reduce
  //===--------------------------------------------------------------------===//

  // Initialize scalar tensor with 0.0
  Value zeroFloat = builder.create<arith::ConstantOp>(
      loc, builder.getFloatAttr(f32Type, APFloat(0.0f)));
  
  Value initSumTensor = builder.create<tensor::FromElementsOp>(
      loc, scalarTensorType, ValueRange{zeroFloat});

  // Create linalg.reduce to compute sum
  auto reduceSumOp = builder.create<linalg::ReduceOp>(
      loc, expTensor, initSumTensor,
      SmallVector<int64_t>{0},  // reduce along dimension 0
      [&](OpBuilder& b, Location loc, ValueRange args) {
        // args[0]: current element from expTensor
        // args[1]: current accumulator (sum so far)
        Value newSum = b.create<arith::AddFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, newSum);
      }
  );
  Value sumTensor = reduceSumOp.getResult(0);
  
  // Extract scalar value from tensor<f32>
  Value sumVal = builder.create<tensor::ExtractOp>(loc, sumTensor, ValueRange{});

  //===--------------------------------------------------------------------===//
  // Pass 4: Normalize by dividing by sum using linalg.generic
  //===--------------------------------------------------------------------===//

  // Create empty output tensor
  Value outputEmpty = builder.create<tensor::EmptyOp>(
      loc, dynamicSizes, f32Type);

  // linalg.generic: normalize exp_tensor[i] / sum
  auto normalizeOp = builder.create<linalg::GenericOp>(
      loc, dynamicTensorType, expTensor, outputEmpty,
      indexingMaps, iteratorTypes,
      [&](OpBuilder& b, Location loc, ValueRange args) {
        // args[0]: exp value
        Value normalized = b.create<arith::DivFOp>(loc, args[0], sumVal);
        b.create<linalg::YieldOp>(loc, normalized);
      }
  );
  Value result = normalizeOp.getResult(0);

  // Return result tensor
  builder.create<func::ReturnOp>(loc, result);

  return module;
}

/// Helper function to print generated IR (for debugging/testing)
std::string printSoftmaxModule(MLIRContext& context) {
  auto module = createSoftmaxModule(context);
  std::string result;
  llvm::raw_string_ostream os(result);
  module->print(os);
  return result;
}

} // namespace mlir