//===- TransformerPasses.cpp - Lower Transformer to Standard -----*- C++ -*-===//
#include "TransformerPasses.h"
#include "TransformerOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::transformer;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static Value createConstantFloat(OpBuilder &builder, Location loc, float value) {
  return builder.create<arith::ConstantOp>(
      loc, builder.getF32Type(), builder.getF32FloatAttr(value));
}

static Value createConstantIndex(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

//===----------------------------------------------------------------------===//
// LayerNormOp Lowering
//===----------------------------------------------------------------------===//

struct LayerNormOpLowering : public OpRewritePattern<LayerNormOp> {
  using OpRewritePattern<LayerNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LayerNormOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value gamma = op.getGamma();
    Value beta = op.getBeta();
    float epsilon = 1e-5f;

    // Get result type
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    ArrayRef<int64_t> shape = resultType.getShape();
    int rank = shape.size();

    // Create reduced shape for mean and variance (reduce along last dim)
    SmallVector<int64_t> reducedShape(shape.begin(), shape.end() - 1);
    SmallVector<Value> reducedDynamicDims;
    for (int i = 0; i < rank - 1; ++i) {
      if (resultType.isDynamicDim(i)) {
        Value dim = rewriter.create<tensor::DimOp>(loc, input, i);
        reducedDynamicDims.push_back(dim);
      }
    }
    auto reducedType = RankedTensorType::get(reducedShape, rewriter.getF32Type());

    Value zero = createConstantFloat(rewriter, loc, 0.0f);
    Value epsilonVal = createConstantFloat(rewriter, loc, epsilon);

    // Step 1: Compute mean using linalg.reduce (sum along last dim)
    Value meanBufferEmpty = rewriter.create<tensor::EmptyOp>(loc, reducedType, reducedDynamicDims);
    Value meanBuffer = rewriter.create<linalg::FillOp>(loc, zero, meanBufferEmpty).getResult(0);
    
    Value meanSum = rewriter.create<linalg::ReduceOp>(
        loc,
        ValueRange{input},
        ValueRange{meanBuffer},
        SmallVector<int64_t>{rank - 1},  // reduce along last dimension
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        }
    ).getResult(0);

    // Divide by d_model to get mean
    Value dModel = createConstantFloat(rewriter, loc, static_cast<float>(shape[rank - 1]));
    SmallVector<AffineMap> meanNormalizeMaps;
    auto reducedIdentityMap = AffineMap::getMultiDimIdentityMap(rank - 1, rewriter.getContext());
    meanNormalizeMaps.push_back(reducedIdentityMap);  // meanSum
    meanNormalizeMaps.push_back(reducedIdentityMap);  // output

    SmallVector<utils::IteratorType> reducedIteratorTypes(rank - 1, utils::IteratorType::parallel);

    Value meanBufferEmptyNorm = rewriter.create<tensor::EmptyOp>(loc, reducedType, reducedDynamicDims);
    Value meanResult = rewriter.create<linalg::GenericOp>(
        loc,
        reducedType,
        ValueRange{meanSum},
        ValueRange{meanBufferEmptyNorm},
        meanNormalizeMaps,
        reducedIteratorTypes,
        [dModel](OpBuilder &b, Location loc, ValueRange args) {
          Value normalized = b.create<arith::DivFOp>(loc, args[0], dModel);
          b.create<linalg::YieldOp>(loc, normalized);
        }
    ).getResult(0);

    // Step 2: Compute variance: sum((input - mean)^2) / d_model
    // Extract dynamic dimensions for full tensors
    SmallVector<Value> dynamicDims;
    for (int i = 0; i < rank; ++i) {
      if (resultType.isDynamicDim(i)) {
        Value dim = rewriter.create<tensor::DimOp>(loc, input, i);
        dynamicDims.push_back(dim);
      }
    }

    // Compute centered = input - mean (broadcast mean)
    SmallVector<AffineMap> centeringMaps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<AffineExpr> reducedExprs;
    for (int i = 0; i < rank - 1; i++) {
      reducedExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    auto broadcastMap = AffineMap::get(rank, 0, reducedExprs, rewriter.getContext());

    centeringMaps.push_back(identityMap);     // input
    centeringMaps.push_back(broadcastMap);    // meanResult (broadcasted)
    centeringMaps.push_back(identityMap);     // centeredBuffer

    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    Value centeredBufferEmpty = rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicDims);
    Value centeredBuffer = rewriter.create<linalg::GenericOp>(
        loc,
        resultType,
        ValueRange{input, meanResult},
        ValueRange{centeredBufferEmpty},
        centeringMaps,
        iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value centered = b.create<arith::SubFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, centered);
        }
    ).getResult(0);

    // Compute variance: sum(centered^2) / d_model
    Value varianceBufferEmpty = rewriter.create<tensor::EmptyOp>(loc, reducedType, reducedDynamicDims);
    Value varianceBuffer = rewriter.create<linalg::FillOp>(loc, zero, varianceBufferEmpty).getResult(0);
    
    Value varianceSum = rewriter.create<linalg::ReduceOp>(
        loc,
        ValueRange{centeredBuffer},
        ValueRange{varianceBuffer},
        SmallVector<int64_t>{rank - 1},
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value squared = b.create<arith::MulFOp>(loc, args[0], args[0]);
          Value sum = b.create<arith::AddFOp>(loc, squared, args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        }
    ).getResult(0);

    // Divide by d_model and compute rsqrt(variance + epsilon)
    Value invStdBufferEmpty = rewriter.create<tensor::EmptyOp>(loc, reducedType, reducedDynamicDims);
    Value invStdResult = rewriter.create<linalg::GenericOp>(
        loc,
        reducedType,
        ValueRange{varianceSum},
        ValueRange{invStdBufferEmpty},
        meanNormalizeMaps,
        reducedIteratorTypes,
        [dModel, epsilonVal](OpBuilder &b, Location loc, ValueRange args) {
          Value variance = b.create<arith::DivFOp>(loc, args[0], dModel);
          Value variancePlusEps = b.create<arith::AddFOp>(loc, variance, epsilonVal);
          Value invStd = b.create<math::RsqrtOp>(loc, variancePlusEps);
          b.create<linalg::YieldOp>(loc, invStd);
        }
    ).getResult(0);

    // Step 3: Normalize and apply scale/shift: output = ((input - mean) * invStd) * gamma + beta
    // Using centered values, multiply by invStd, then scale by gamma and shift by beta
    // Create broadcast map for gamma/beta (1D tensors indexing only last dimension)
    SmallVector<AffineExpr> gammaBetaExprs;
    gammaBetaExprs.push_back(rewriter.getAffineDimExpr(rank - 1));  // only last dimension
    auto gammaBetaBroadcastMap = AffineMap::get(rank, 0, gammaBetaExprs, rewriter.getContext());

    SmallVector<AffineMap> normalizeMaps;
    normalizeMaps.push_back(identityMap);              // centeredBuffer
    normalizeMaps.push_back(broadcastMap);             // invStdResult (broadcasted)
    normalizeMaps.push_back(gammaBetaBroadcastMap);    // gamma (broadcasted)
    normalizeMaps.push_back(gammaBetaBroadcastMap);    // beta (broadcasted)
    normalizeMaps.push_back(identityMap);              // output

    Value resultEmpty = rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicDims);
    Value result = rewriter.create<linalg::GenericOp>(
        loc,
        resultType,
        ValueRange{centeredBuffer, invStdResult, gamma, beta},
        ValueRange{resultEmpty},
        normalizeMaps,
        iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = centered, args[1] = invStd, args[2] = gamma, args[3] = beta
          Value normalized = b.create<arith::MulFOp>(loc, args[0], args[1]);
          Value scaled = b.create<arith::MulFOp>(loc, normalized, args[2]);
          Value finalResult = b.create<arith::AddFOp>(loc, scaled, args[3]);
          b.create<linalg::YieldOp>(loc, finalResult);
        }
    ).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LinearOp Lowering
//===----------------------------------------------------------------------===//

struct LinearOpLowering : public OpRewritePattern<LinearOp> {
  using OpRewritePattern<LinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LinearOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value weight = op.getWeight();
    Value bias = op.getBias();

    // Get result type
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto weightType = mlir::cast<RankedTensorType>(weight.getType());

    // Linear: output = input @ weight^T + bias
    // input: (seq_len, in_features), weight: (out_features, in_features)
    // Need to transpose weight to (in_features, out_features) for matmul

    int64_t seqLen = inputType.getShape()[0];
    int64_t inFeatures = inputType.getShape()[1];   // from input's second dim
    int64_t outFeatures = weightType.getShape()[0]; // from weight's first dim

    // Extract dynamic dimensions for transposed weight
    SmallVector<int64_t> transposedWeightShape = {inFeatures, outFeatures};
    SmallVector<Value> transposedDynamicDims;
    auto transposedWeightType = RankedTensorType::get(transposedWeightShape, rewriter.getF32Type());
    for (int i = 0; i < 2; ++i) {
      if (transposedWeightType.isDynamicDim(i)) {
        Value dim = rewriter.create<tensor::DimOp>(loc, weight, i == 0 ? 1 : 0);
        transposedDynamicDims.push_back(dim);
      }
    }

    // Step 1: Transpose weight (out_features, in_features) -> (in_features, out_features)
    Value transposedWeightEmpty = rewriter.create<tensor::EmptyOp>(loc, transposedWeightType, transposedDynamicDims);
    Value transposedWeight = rewriter.create<linalg::TransposeOp>(
        loc,
        weight,
        transposedWeightEmpty,
        SmallVector<int64_t>{1, 0}  // swap dimensions
    ).getResult()[0];

    // Extract dynamic dimensions for output
    SmallVector<Value> dynamicDims;
    for (int i = 0; i < resultType.getRank(); ++i) {
      if (resultType.isDynamicDim(i)) {
        Value dim = rewriter.create<tensor::DimOp>(loc, input, i);
        dynamicDims.push_back(dim);
      }
    }

    // Step 2: Initialize output to zero and perform matmul
    Value zero = createConstantFloat(rewriter, loc, 0.0f);
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicDims);
    Value filledTensor = rewriter.create<linalg::FillOp>(loc, zero, emptyTensor).getResult(0);

    // Matmul: (seq_len, in_features) @ (in_features, out_features) -> (seq_len, out_features)
    Value matmulResult = rewriter.create<linalg::MatmulOp>(
        loc,
        ValueRange{input, transposedWeight},
        ValueRange{filledTensor}
    ).getResult(0);

    // Step 3: Add bias using linalg.generic (broadcast bias across seq_len)
    int rank = 2;
    SmallVector<AffineMap> indexingMaps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());

    // Broadcast map for bias (only last dimension)
    SmallVector<AffineExpr> biasExprs;
    biasExprs.push_back(rewriter.getAffineDimExpr(1));  // only use second dimension
    auto broadcastMap = AffineMap::get(rank, 0, biasExprs, rewriter.getContext());

    indexingMaps.push_back(identityMap);     // matmulResult
    indexingMaps.push_back(broadcastMap);    // bias (broadcasted)
    indexingMaps.push_back(identityMap);     // output

    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    Value resultEmpty = rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicDims);
    Value result = rewriter.create<linalg::GenericOp>(
        loc,
        resultType,
        ValueRange{matmulResult, bias},
        ValueRange{resultEmpty},
        indexingMaps,
        iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        }
    ).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GeluOp Lowering
//===----------------------------------------------------------------------===//
// GeluOp Lowering (Tensor-Based)
//===----------------------------------------------------------------------===//

struct GeluOpLowering : public OpRewritePattern<GeluOp> {
  using OpRewritePattern<GeluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GeluOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();

    // Get result type
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    int rank = resultType.getRank();

    // GELU approximation constants: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Value c0_5 = createConstantFloat(rewriter, loc, 0.5f);
    Value c1 = createConstantFloat(rewriter, loc, 1.0f);
    Value cSqrt2OverPi = createConstantFloat(rewriter, loc, 0.7978845608f); // sqrt(2/pi)
    Value c0_044715 = createConstantFloat(rewriter, loc, 0.044715f);

    // Create empty output tensor
    SmallVector<Value> dynamicDims;
    for (int i = 0; i < rank; ++i) {
      if (ShapedType::isDynamic(resultType.getShape()[i])) {
        dynamicDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
      }
    }
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicDims);

    // Use linalg.generic for element-wise GELU activation
    SmallVector<AffineMap> indexingMaps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    indexingMaps.push_back(identityMap);  // input
    indexingMaps.push_back(identityMap);  // result

    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    // Capture constants for use in body builder
    Value capturedC0_5 = c0_5;
    Value capturedC1 = c1;
    Value capturedCSqrt2OverPi = cSqrt2OverPi;
    Value capturedC0_044715 = c0_044715;

    Value result = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/resultType,
        /*inputs=*/ValueRange{input},
        /*outputs=*/ValueRange{emptyTensor},
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*bodyBuilder=*/[capturedC0_5, capturedC1, capturedCSqrt2OverPi, capturedC0_044715]
        (OpBuilder &b, Location loc, ValueRange args) {
          Value x = args[0];

          // Compute x^3
          Value x2 = b.create<arith::MulFOp>(loc, x, x);
          Value x3 = b.create<arith::MulFOp>(loc, x2, x);

          // Compute 0.044715 * x^3
          Value term = b.create<arith::MulFOp>(loc, capturedC0_044715, x3);

          // Compute x + 0.044715 * x^3
          Value inner = b.create<arith::AddFOp>(loc, x, term);

          // Compute sqrt(2/pi) * (x + 0.044715 * x^3)
          Value scaled = b.create<arith::MulFOp>(loc, capturedCSqrt2OverPi, inner);

          // Compute tanh(...)
          Value tanhVal = b.create<math::TanhOp>(loc, scaled);

          // Compute 1 + tanh(...)
          Value onePlusTanh = b.create<arith::AddFOp>(loc, capturedC1, tanhVal);

          // Compute 0.5 * x * (1 + tanh(...))
          Value halfX = b.create<arith::MulFOp>(loc, capturedC0_5, x);
          Value geluResult = b.create<arith::MulFOp>(loc, halfX, onePlusTanh);

          b.create<linalg::YieldOp>(loc, geluResult);
        }
    ).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AddOp Lowering
//===----------------------------------------------------------------------===//
// AddOp Lowering (Tensor-Based)
//===----------------------------------------------------------------------===//

struct AddOpLowering : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // Get result type
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    int rank = resultType.getRank();

    // Create indexing maps: all identity (parallel element-wise operation)
    SmallVector<AffineMap> indexingMaps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    indexingMaps.push_back(identityMap);  // lhs
    indexingMaps.push_back(identityMap);  // rhs
    indexingMaps.push_back(identityMap);  // result

    // All dimensions are parallel (no reductions)
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    // Create empty tensor for output
    SmallVector<Value> dynamicDims;
    for (int i = 0; i < rank; ++i) {
      if (ShapedType::isDynamic(resultType.getShape()[i])) {
        dynamicDims.push_back(rewriter.create<tensor::DimOp>(loc, lhs, i));
      }
    }
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicDims);

    // Create linalg.generic operation returning tensor
    Value result = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/resultType,
        /*inputs=*/ValueRange{lhs, rhs},
        /*outputs=*/ValueRange{emptyTensor},
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*bodyBuilder=*/[](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = lhs element, args[1] = rhs element
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        }
    ).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MatmulOp Lowering (Tensor-Based)
//===----------------------------------------------------------------------===//

struct MatmulOpLowering : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // Get result type
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());

    // Create empty tensor for output
    SmallVector<Value> dynamicDims;
    for (int i = 0; i < resultType.getRank(); ++i) {
      if (ShapedType::isDynamic(resultType.getShape()[i])) {
        dynamicDims.push_back(rewriter.create<tensor::DimOp>(loc, lhs, i));
      }
    }
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicDims);

    // Zero-initialize with linalg.fill (returns tensor)
    Value zero = createConstantFloat(rewriter, loc, 0.0f);
    Value filledTensor = rewriter.create<linalg::FillOp>(loc, zero, emptyTensor).getResult(0);

    // Perform matrix multiplication (returns tensor)
    Value result = rewriter.create<linalg::MatmulOp>(
        loc,
        ValueRange{lhs, rhs},  // inputs
        ValueRange{filledTensor}     // output tensor
    ).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TransposeOp Lowering (Tensor-Based)
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();

    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    int rank = inputType.getRank();

    // Create permutation: swap last two dimensions
    // For rank=2: [0,1] -> [1,0]
    // For rank=3: [0,1,2] -> [0,2,1]
    SmallVector<int64_t> permutation;
    for (int i = 0; i < rank - 2; i++) {
      permutation.push_back(i);
    }
    permutation.push_back(rank - 1);  // Swap: last dimension first
    permutation.push_back(rank - 2);  // Swap: second-to-last dimension second

    // Create empty output tensor with transposed shape
    SmallVector<Value> dynamicDims;
    for (int i = 0; i < rank; ++i) {
      if (ShapedType::isDynamic(resultType.getShape()[i])) {
        int64_t inputDim = permutation[i];
        dynamicDims.push_back(rewriter.create<tensor::DimOp>(loc, input, inputDim));
      }
    }
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicDims);

    // Use linalg.transpose (returns tensor)
    Value result = rewriter.create<linalg::TransposeOp>(
        loc,
        input,
        emptyTensor,
        permutation
    ).getResult()[0];

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SoftmaxOp Lowering
//===----------------------------------------------------------------------===//

struct SoftmaxOpLowering : public OpRewritePattern<SoftmaxOp> {
  using OpRewritePattern<SoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SoftmaxOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();

    // Get result type
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    ArrayRef<int64_t> shape = resultType.getShape();
    int rank = shape.size();

    // Create reduced shape for max/sum buffers
    SmallVector<int64_t> reducedShape(shape.begin(), shape.end() - 1);
    SmallVector<Value> reducedDynamicDims;
    for (int i = 0; i < rank - 1; ++i) {
      if (resultType.isDynamicDim(i)) {
        Value dim = rewriter.create<tensor::DimOp>(loc, input, i);
        reducedDynamicDims.push_back(dim);
      }
    }
    auto reducedType = RankedTensorType::get(reducedShape, rewriter.getF32Type());

    // Extract dynamic dimensions for output tensor
    SmallVector<Value> dynamicDims;
    for (int i = 0; i < rank; ++i) {
      if (resultType.isDynamicDim(i)) {
        Value dim = rewriter.create<tensor::DimOp>(loc, input, i);
        dynamicDims.push_back(dim);
      }
    }

    // Step 1: Find max along last dimension using linalg.reduce
    Value negInf = createConstantFloat(rewriter, loc, -1e9f);
    Value maxBufferEmpty = rewriter.create<tensor::EmptyOp>(loc, reducedType, reducedDynamicDims);
    Value maxBuffer = rewriter.create<linalg::FillOp>(loc, negInf, maxBufferEmpty).getResult(0);

    Value maxResult = rewriter.create<linalg::ReduceOp>(
        loc,
        ValueRange{input},
        ValueRange{maxBuffer},
        SmallVector<int64_t>{rank - 1},  // reduce along last dimension
        [](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = input element, args[1] = current max
          Value newMax = b.create<arith::MaximumFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, newMax);
        }
    ).getResult(0);

    // Step 2: Compute exp(input - max) using linalg.generic
    SmallVector<AffineMap> indexingMaps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<AffineExpr> reducedExprs;
    for (int i = 0; i < rank - 1; i++) {
      reducedExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    auto broadcastMap = AffineMap::get(rank, 0, reducedExprs, rewriter.getContext());

    indexingMaps.push_back(identityMap);     // input
    indexingMaps.push_back(broadcastMap);    // maxResult (broadcasted)
    indexingMaps.push_back(identityMap);     // output

    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicDims);
    Value expResult = rewriter.create<linalg::GenericOp>(
        loc,
        resultType,
        ValueRange{input, maxResult},
        ValueRange{emptyTensor},
        indexingMaps,
        iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = input element, args[1] = max value (broadcasted)
          Value shifted = b.create<arith::SubFOp>(loc, args[0], args[1]);
          Value expVal = b.create<math::ExpOp>(loc, shifted);
          b.create<linalg::YieldOp>(loc, expVal);
        }
    ).getResult(0);

    // Step 3: Sum exp values along last dimension using linalg.reduce
    Value zero = createConstantFloat(rewriter, loc, 0.0f);
    Value sumBufferEmpty = rewriter.create<tensor::EmptyOp>(loc, reducedType, reducedDynamicDims);
    Value sumBuffer = rewriter.create<linalg::FillOp>(loc, zero, sumBufferEmpty).getResult(0);

    Value sumResult = rewriter.create<linalg::ReduceOp>(
        loc,
        ValueRange{expResult},  // input is now the exp values
        ValueRange{sumBuffer},
        SmallVector<int64_t>{rank - 1},  // reduce along last dimension
        [](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = exp element, args[1] = current sum
          Value newSum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, newSum);
        }
    ).getResult(0);

    // Step 4: Normalize by dividing exp values by sum using linalg.generic
    Value emptyResult = rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicDims);
    Value result = rewriter.create<linalg::GenericOp>(
        loc,
        resultType,
        ValueRange{expResult, sumResult},  // exp values and sum
        ValueRange{emptyResult},
        SmallVector<AffineMap>{identityMap, broadcastMap, identityMap},
        iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = exp element, args[1] = sum (broadcasted)
          Value normalized = b.create<arith::DivFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, normalized);
        }
    ).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ScaleOp Lowering
//===----------------------------------------------------------------------===//

struct ScaleOpLowering : public OpRewritePattern<ScaleOp> {
  using OpRewritePattern<ScaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScaleOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value scale = op.getScale();

    // Get result type
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    int rank = resultType.getRank();

    // Extract dynamic dimensions
    SmallVector<Value> dynamicDims;
    for (int i = 0; i < rank; ++i) {
      if (resultType.isDynamicDim(i)) {
        Value dim = rewriter.create<tensor::DimOp>(loc, input, i);
        dynamicDims.push_back(dim);
      }
    }

    // Create empty tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicDims);

    // Load scalar scale value from 1D tensor
    Value zeroIdx = createConstantIndex(rewriter, loc, 0);
    Value scalarValue = rewriter.create<tensor::ExtractOp>(loc, scale, ValueRange{zeroIdx});

    // Use linalg.generic for element-wise multiplication
    SmallVector<AffineMap> indexingMaps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    indexingMaps.push_back(identityMap);  // input
    indexingMaps.push_back(identityMap);  // output

    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    // Capture the scalar value for use in body builder
    Value capturedScalar = scalarValue;

    Value result = rewriter.create<linalg::GenericOp>(
        loc,
        resultType,
        ValueRange{input},
        ValueRange{emptyTensor},
        indexingMaps,
        iteratorTypes,
        [capturedScalar](OpBuilder &b, Location loc, ValueRange args) {
          Value product = b.create<arith::MulFOp>(loc, args[0], capturedScalar);
          b.create<linalg::YieldOp>(loc, product);
        }
    ).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};


//===----------------------------------------------------------------------===//
// EmbeddingOp Lowering (Chapter 13 - tensor-based)
//===----------------------------------------------------------------------===//

struct EmbeddingOpLowering : public OpRewritePattern<EmbeddingOp> {
  using OpRewritePattern<EmbeddingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(EmbeddingOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value indices = op.getIndices();
    Value table = op.getTable();

    auto indicesType = mlir::cast<RankedTensorType>(indices.getType());
    auto tableType = mlir::cast<RankedTensorType>(table.getType());
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());

    SmallVector<Value> dynamicDims;
    for (int i = 0; i < resultType.getRank(); ++i) {
      if (resultType.isDynamicDim(i)) {
        Value dim = rewriter.create<tensor::DimOp>(loc, i == 0 ? indices : table, i);
        dynamicDims.push_back(dim);
      }
    }
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicDims);

    SmallVector<AffineMap> indexingMaps;
    auto outputMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
    indexingMaps.push_back(outputMap);

    SmallVector<utils::IteratorType> iteratorTypes(2, utils::IteratorType::parallel);

    Value capturedIndices = indices;
    Value capturedTable = table;

    Value result = rewriter.create<linalg::GenericOp>(
        loc,
        resultType,
        ValueRange{},
        ValueRange{emptyTensor},
        indexingMaps,
        iteratorTypes,
        [capturedIndices, capturedTable](OpBuilder &b, Location loc, ValueRange args) {
          Value posIdx = b.create<linalg::IndexOp>(loc, 0);
          Value dimIdx = b.create<linalg::IndexOp>(loc, 1);
          
          Value tokenId32 = b.create<tensor::ExtractOp>(loc, capturedIndices, ValueRange{posIdx});
          Value tokenIdx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), tokenId32);
          
          Value embVal = b.create<tensor::ExtractOp>(loc, capturedTable, ValueRange{tokenIdx, dimIdx});
          b.create<linalg::YieldOp>(loc, embVal);
        }
    ).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CreateCausalMaskOp Lowering (Chapter 13 - tensor-based)
//===----------------------------------------------------------------------===//

struct CreateCausalMaskOpLowering : public OpRewritePattern<CreateCausalMaskOp> {
  using OpRewritePattern<CreateCausalMaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CreateCausalMaskOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());

    Value zeroFloat = createConstantFloat(rewriter, loc, 0.0f);
    Value negInf = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(rewriter.getF32Type(),
            llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(), /*Negative=*/true)));

    SmallVector<Value> dynamicDims;
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicDims);

    SmallVector<AffineMap> indexingMaps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
    indexingMaps.push_back(identityMap);

    SmallVector<utils::IteratorType> iteratorTypes(2, utils::IteratorType::parallel);

    Value capturedZero = zeroFloat;
    Value capturedNegInf = negInf;

    Value result = rewriter.create<linalg::GenericOp>(
        loc,
        resultType,
        ValueRange{},
        ValueRange{emptyTensor},
        indexingMaps,
        iteratorTypes,
        [capturedZero, capturedNegInf](OpBuilder &b, Location loc, ValueRange args) {
          Value i = b.create<linalg::IndexOp>(loc, 0);
          Value j = b.create<linalg::IndexOp>(loc, 1);
          Value cmp = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, j, i);
          Value maskValue = b.create<arith::SelectOp>(loc, cmp, capturedZero, capturedNegInf);
          b.create<linalg::YieldOp>(loc, maskValue);
        }
    ).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MaskedSoftmaxOp Lowering (Chapter 13 - tensor-based)
//===----------------------------------------------------------------------===//

struct MaskedSoftmaxOpLowering : public OpRewritePattern<MaskedSoftmaxOp> {
  using OpRewritePattern<MaskedSoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MaskedSoftmaxOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value mask = op.getMask();

    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    int rank = inputType.getRank();

    SmallVector<Value> dynamicDims;
    for (int i = 0; i < rank; ++i) {
      if (resultType.isDynamicDim(i)) {
        Value dim = rewriter.create<tensor::DimOp>(loc, input, i);
        dynamicDims.push_back(dim);
      }
    }

    // Step 1: Add mask to input
    SmallVector<AffineMap> addMaskMaps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<AffineExpr> maskExprs;
    for (int i = rank - 2; i < rank; i++) {
      maskExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    auto maskBroadcastMap = AffineMap::get(rank, 0, maskExprs, rewriter.getContext());

    addMaskMaps.push_back(identityMap);
    addMaskMaps.push_back(maskBroadcastMap);
    addMaskMaps.push_back(identityMap);

    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    Value emptyMasked = rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicDims);
    Value maskedInput = rewriter.create<linalg::GenericOp>(
        loc,
        resultType,
        ValueRange{input, mask},
        ValueRange{emptyMasked},
        addMaskMaps,
        iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value masked = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, masked);
        }
    ).getResult(0);

    // Step 2-4: Apply standard softmax to masked input
    ArrayRef<int64_t> shape = resultType.getShape();
    SmallVector<int64_t> reducedShape(shape.begin(), shape.end() - 1);
    SmallVector<Value> reducedDynamicDims;
    for (int i = 0; i < rank - 1; ++i) {
      if (resultType.isDynamicDim(i)) {
        Value dim = rewriter.create<tensor::DimOp>(loc, maskedInput, i);
        reducedDynamicDims.push_back(dim);
      }
    }
    auto reducedType = RankedTensorType::get(reducedShape, rewriter.getF32Type());

    Value negInf = createConstantFloat(rewriter, loc, -1e9f);
    Value maxBufferEmpty = rewriter.create<tensor::EmptyOp>(loc, reducedType, reducedDynamicDims);
    Value maxBuffer = rewriter.create<linalg::FillOp>(loc, negInf, maxBufferEmpty).getResult(0);

    Value maxResult = rewriter.create<linalg::ReduceOp>(
        loc,
        ValueRange{maskedInput},
        ValueRange{maxBuffer},
        SmallVector<int64_t>{rank - 1},
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value newMax = b.create<arith::MaximumFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, newMax);
        }
    ).getResult(0);

    SmallVector<AffineExpr> reducedExprs;
    for (int i = 0; i < rank - 1; i++) {
      reducedExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    auto broadcastMap = AffineMap::get(rank, 0, reducedExprs, rewriter.getContext());

    SmallVector<AffineMap> expMaps;
    expMaps.push_back(identityMap);
    expMaps.push_back(broadcastMap);
    expMaps.push_back(identityMap);

    Value emptyExp = rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicDims);
    Value expResult = rewriter.create<linalg::GenericOp>(
        loc,
        resultType,
        ValueRange{maskedInput, maxResult},
        ValueRange{emptyExp},
        expMaps,
        iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value shifted = b.create<arith::SubFOp>(loc, args[0], args[1]);
          Value expVal = b.create<math::ExpOp>(loc, shifted);
          b.create<linalg::YieldOp>(loc, expVal);
        }
    ).getResult(0);

    Value zero = createConstantFloat(rewriter, loc, 0.0f);
    Value sumBufferEmpty = rewriter.create<tensor::EmptyOp>(loc, reducedType, reducedDynamicDims);
    Value sumBuffer = rewriter.create<linalg::FillOp>(loc, zero, sumBufferEmpty).getResult(0);

    Value sumResult = rewriter.create<linalg::ReduceOp>(
        loc,
        ValueRange{expResult},
        ValueRange{sumBuffer},
        SmallVector<int64_t>{rank - 1},
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value newSum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, newSum);
        }
    ).getResult(0);

    Value emptyResult = rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicDims);
    Value result = rewriter.create<linalg::GenericOp>(
        loc,
        resultType,
        ValueRange{expResult, sumResult},
        ValueRange{emptyResult},
        SmallVector<AffineMap>{identityMap, broadcastMap, identityMap},
        iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value normalized = b.create<arith::DivFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, normalized);
        }
    ).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RoPEOp Lowering (Chapter 13 - tensor-based, simplified)
//===----------------------------------------------------------------------===//

struct RoPEOpLowering : public OpRewritePattern<RoPEOp> {
  using OpRewritePattern<RoPEOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RoPEOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();

    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    int rank = inputType.getRank();

    ArrayRef<int64_t> shape = inputType.getShape();
    int64_t dModel = shape[rank - 1];

    Value base = createConstantFloat(rewriter, loc, 10000.0f);
    Value two = createConstantFloat(rewriter, loc, 2.0f);
    Value dModelFloat = createConstantFloat(rewriter, loc, static_cast<float>(dModel));

    SmallVector<Value> dynamicDims;
    for (int i = 0; i < rank; ++i) {
      if (resultType.isDynamicDim(i)) {
        Value dim = rewriter.create<tensor::DimOp>(loc, input, i);
        dynamicDims.push_back(dim);
      }
    }
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType, dynamicDims);

    SmallVector<AffineMap> indexingMaps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    indexingMaps.push_back(identityMap);
    indexingMaps.push_back(identityMap);

    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    Value capturedBase = base;
    Value capturedTwo = two;
    Value capturedDModel = dModelFloat;
    Value capturedInput = input;

    Value result = rewriter.create<linalg::GenericOp>(
        loc,
        resultType,
        ValueRange{input},
        ValueRange{emptyTensor},
        indexingMaps,
        iteratorTypes,
        [capturedBase, capturedTwo, capturedDModel, capturedInput, rank](OpBuilder &b, Location loc, ValueRange args) {
          Value x = args[0];
          
          Value posIdx = b.create<linalg::IndexOp>(loc, rank - 2);
          Value dimIdx = b.create<linalg::IndexOp>(loc, rank - 1);

          Value posI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), posIdx);
          Value posFloat = b.create<arith::SIToFPOp>(loc, b.getF32Type(), posI64);

          Value dimI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), dimIdx);
          
          // Compute pair index j = dim / 2 (integer division)
          Value two_i64 = b.create<arith::ConstantOp>(loc, b.getI64Type(), b.getI64IntegerAttr(2));
          Value j_i64 = b.create<arith::DivSIOp>(loc, dimI64, two_i64);
          Value j_float = b.create<arith::SIToFPOp>(loc, b.getF32Type(), j_i64);
          
          // Check if dimension is even
          Value dimIdxMod2 = b.create<arith::RemSIOp>(loc, dimI64, two_i64);
          Value zero_i64 = b.create<arith::ConstantOp>(loc, b.getI64Type(), b.getI64IntegerAttr(0));
          Value isEven = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, dimIdxMod2, zero_i64);

          // Compute theta_j = 10000^(-2j/d_model)
          Value twoJ = b.create<arith::MulFOp>(loc, capturedTwo, j_float);
          Value exponent = b.create<arith::DivFOp>(loc, twoJ, capturedDModel);
          Value negExponent = b.create<arith::NegFOp>(loc, exponent);
          Value theta = b.create<math::PowFOp>(loc, capturedBase, negExponent);
          Value angle = b.create<arith::MulFOp>(loc, posFloat, theta);

          Value cosAngle = b.create<math::CosOp>(loc, angle);
          Value sinAngle = b.create<math::SinOp>(loc, angle);

          Value one = b.create<arith::ConstantIndexOp>(loc, 1);
          Value nextDim = b.create<arith::AddIOp>(loc, dimIdx, one);
          Value prevDim = b.create<arith::SubIOp>(loc, dimIdx, one);
          
          Value pairDimIdx = b.create<arith::SelectOp>(loc, isEven, nextDim, prevDim);

          SmallVector<Value> pairIndices;
          for (int i = 0; i < rank - 1; i++) {
            pairIndices.push_back(b.create<linalg::IndexOp>(loc, i));
          }
          pairIndices.push_back(pairDimIdx);
          Value xPair = b.create<tensor::ExtractOp>(loc, capturedInput, pairIndices);

          Value xCos = b.create<arith::MulFOp>(loc, x, cosAngle);
          Value xPairSin = b.create<arith::MulFOp>(loc, xPair, sinAngle);
          
          Value evenResult = b.create<arith::SubFOp>(loc, xCos, xPairSin);
          Value oddResult = b.create<arith::AddFOp>(loc, xPairSin, xCos);
          Value rotated = b.create<arith::SelectOp>(loc, isEven, evenResult, oddResult);

          b.create<linalg::YieldOp>(loc, rotated);
        }
    ).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower Transformer to Standard Pass (Chapter 13)
//===----------------------------------------------------------------------===//

namespace {
struct LowerTransformerToStandardPass
    : public PassWrapper<LowerTransformerToStandardPass, OperationPass<func::FuncOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect,
                    memref::MemRefDialect, math::MathDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LayerNormOpLowering, LinearOpLowering, GeluOpLowering, AddOpLowering,
                 MatmulOpLowering, TransposeOpLowering, SoftmaxOpLowering, ScaleOpLowering,
                 EmbeddingOpLowering, CreateCausalMaskOpLowering, MaskedSoftmaxOpLowering,
                 RoPEOpLowering>(
        &getContext());

    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createLowerTransformerToStandardPass() {
  return std::make_unique<LowerTransformerToStandardPass>();
}
