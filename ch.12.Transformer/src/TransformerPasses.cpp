//===- TransformerPasses.cpp - Lower Transformer to Standard -----*- C++ -*-===//
#include "TransformerPasses.h"
#include "TransformerOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
    Value output = op.getOutput();
    float epsilon = 1e-5f;

    auto inputType = cast<MemRefType>(input.getType());
    ArrayRef<int64_t> shape = inputType.getShape();
    int rank = shape.size();

    // Create temporary buffers for mean and variance (reduce along last dim)
    SmallVector<int64_t> reducedShape(shape.begin(), shape.end() - 1);
    auto reducedType = MemRefType::get(reducedShape, rewriter.getF32Type());
    Value meanBuffer = rewriter.create<memref::AllocOp>(loc, reducedType);
    Value varianceBuffer = rewriter.create<memref::AllocOp>(loc, reducedType);

    Value zero = createConstantFloat(rewriter, loc, 0.0f);
    Value epsilonVal = createConstantFloat(rewriter, loc, epsilon);

    // Step 1: Compute mean using linalg.reduce (sum along last dim)
    rewriter.create<linalg::FillOp>(loc, zero, meanBuffer);
    rewriter.create<linalg::ReduceOp>(
        loc,
        ValueRange{input},
        ValueRange{meanBuffer},
        SmallVector<int64_t>{rank - 1},  // reduce along last dimension
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        }
    );

    // Divide by d_model to get mean
    Value dModel = createConstantFloat(rewriter, loc, static_cast<float>(shape[rank - 1]));
    SmallVector<AffineMap> meanNormalizeMaps;
    auto reducedIdentityMap = AffineMap::getMultiDimIdentityMap(rank - 1, rewriter.getContext());
    meanNormalizeMaps.push_back(reducedIdentityMap);  // meanBuffer
    meanNormalizeMaps.push_back(reducedIdentityMap);  // output (in-place)

    SmallVector<utils::IteratorType> reducedIteratorTypes(rank - 1, utils::IteratorType::parallel);

    rewriter.create<linalg::GenericOp>(
        loc,
        ValueRange{meanBuffer},
        ValueRange{meanBuffer},  // in-place
        meanNormalizeMaps,
        reducedIteratorTypes,
        [dModel](OpBuilder &b, Location loc, ValueRange args) {
          Value normalized = b.create<arith::DivFOp>(loc, args[0], dModel);
          b.create<linalg::YieldOp>(loc, normalized);
        }
    );

    // Step 2: Compute variance: sum((input - mean)^2) / d_model
    // Create temporary buffer for centered values
    Value centeredBuffer = rewriter.create<memref::AllocOp>(loc, inputType);

    // Compute centered = input - mean (broadcast mean)
    SmallVector<AffineMap> centeringMaps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<AffineExpr> reducedExprs;
    for (int i = 0; i < rank - 1; i++) {
      reducedExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    auto broadcastMap = AffineMap::get(rank, 0, reducedExprs, rewriter.getContext());

    centeringMaps.push_back(identityMap);     // input
    centeringMaps.push_back(broadcastMap);    // meanBuffer (broadcasted)
    centeringMaps.push_back(identityMap);     // centeredBuffer

    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    rewriter.create<linalg::GenericOp>(
        loc,
        ValueRange{input, meanBuffer},
        ValueRange{centeredBuffer},
        centeringMaps,
        iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value centered = b.create<arith::SubFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, centered);
        }
    );

    // Compute variance: sum(centered^2) / d_model
    rewriter.create<linalg::FillOp>(loc, zero, varianceBuffer);
    rewriter.create<linalg::ReduceOp>(
        loc,
        ValueRange{centeredBuffer},
        ValueRange{varianceBuffer},
        SmallVector<int64_t>{rank - 1},
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value squared = b.create<arith::MulFOp>(loc, args[0], args[0]);
          Value sum = b.create<arith::AddFOp>(loc, squared, args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        }
    );

    // Divide by d_model and compute rsqrt(variance + epsilon)
    rewriter.create<linalg::GenericOp>(
        loc,
        ValueRange{varianceBuffer},
        ValueRange{varianceBuffer},  // in-place
        meanNormalizeMaps,
        reducedIteratorTypes,
        [dModel, epsilonVal](OpBuilder &b, Location loc, ValueRange args) {
          Value variance = b.create<arith::DivFOp>(loc, args[0], dModel);
          Value variancePlusEps = b.create<arith::AddFOp>(loc, variance, epsilonVal);
          Value invStd = b.create<math::RsqrtOp>(loc, variancePlusEps);
          b.create<linalg::YieldOp>(loc, invStd);
        }
    );

    // Step 3: Normalize and apply scale/shift: output = ((input - mean) * invStd) * gamma + beta
    // Using centered values, multiply by invStd, then scale by gamma and shift by beta
    // Create broadcast map for gamma/beta (1D tensors indexing only last dimension)
    SmallVector<AffineExpr> gammaBetaExprs;
    gammaBetaExprs.push_back(rewriter.getAffineDimExpr(rank - 1));  // only last dimension
    auto gammaBetaBroadcastMap = AffineMap::get(rank, 0, gammaBetaExprs, rewriter.getContext());

    SmallVector<AffineMap> normalizeMaps;
    normalizeMaps.push_back(identityMap);              // centeredBuffer
    normalizeMaps.push_back(broadcastMap);             // varianceBuffer (invStd, broadcasted)
    normalizeMaps.push_back(gammaBetaBroadcastMap);    // gamma (broadcasted)
    normalizeMaps.push_back(gammaBetaBroadcastMap);    // beta (broadcasted)
    normalizeMaps.push_back(identityMap);              // output

    rewriter.create<linalg::GenericOp>(
        loc,
        ValueRange{centeredBuffer, varianceBuffer, gamma, beta},
        ValueRange{output},
        normalizeMaps,
        iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = centered, args[1] = invStd, args[2] = gamma, args[3] = beta
          Value normalized = b.create<arith::MulFOp>(loc, args[0], args[1]);
          Value scaled = b.create<arith::MulFOp>(loc, normalized, args[2]);
          Value result = b.create<arith::AddFOp>(loc, scaled, args[3]);
          b.create<linalg::YieldOp>(loc, result);
        }
    );

    // Clean up temporary buffers
    rewriter.create<memref::DeallocOp>(loc, meanBuffer);
    rewriter.create<memref::DeallocOp>(loc, varianceBuffer);
    rewriter.create<memref::DeallocOp>(loc, centeredBuffer);

    rewriter.eraseOp(op);
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
    Value output = op.getOutput();

    auto inputType = cast<MemRefType>(input.getType());
    auto weightType = cast<MemRefType>(weight.getType());
    auto outputType = cast<MemRefType>(output.getType());

    // Linear: output = input @ weight^T + bias
    // input: (seq_len, in_features), weight: (out_features, in_features)
    // Need to transpose weight to (in_features, out_features) for matmul

    int64_t seqLen = inputType.getShape()[0];
    int64_t inFeatures = inputType.getShape()[1];
    int64_t outFeatures = weightType.getShape()[0];

    // Step 1: Transpose weight (out_features, in_features) -> (in_features, out_features)
    SmallVector<int64_t> transposedWeightShape = {inFeatures, outFeatures};
    auto transposedWeightType = MemRefType::get(transposedWeightShape, rewriter.getF32Type());
    Value transposedWeight = rewriter.create<memref::AllocOp>(loc, transposedWeightType);

    rewriter.create<linalg::TransposeOp>(
        loc,
        weight,
        transposedWeight,
        SmallVector<int64_t>{1, 0}  // swap dimensions
    );

    // Step 2: Initialize output to zero and perform matmul
    Value zero = createConstantFloat(rewriter, loc, 0.0f);
    rewriter.create<linalg::FillOp>(loc, zero, output);

    // Matmul: (seq_len, in_features) @ (in_features, out_features) -> (seq_len, out_features)
    rewriter.create<linalg::MatmulOp>(
        loc,
        ValueRange{input, transposedWeight},
        ValueRange{output}
    );

    // Step 3: Add bias using linalg.generic (broadcast bias across seq_len)
    int rank = 2;
    SmallVector<AffineMap> indexingMaps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());

    // Broadcast map for bias (only last dimension)
    SmallVector<AffineExpr> biasExprs;
    biasExprs.push_back(rewriter.getAffineDimExpr(1));  // only use second dimension
    auto broadcastMap = AffineMap::get(rank, 0, biasExprs, rewriter.getContext());

    indexingMaps.push_back(identityMap);     // output (read)
    indexingMaps.push_back(broadcastMap);    // bias (broadcasted)
    indexingMaps.push_back(identityMap);     // output (write)

    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    rewriter.create<linalg::GenericOp>(
        loc,
        ValueRange{output, bias},
        ValueRange{output},  // in-place
        indexingMaps,
        iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, result);
        }
    );

    // Clean up
    rewriter.create<memref::DeallocOp>(loc, transposedWeight);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GeluOp Lowering
//===----------------------------------------------------------------------===//

struct GeluOpLowering : public OpRewritePattern<GeluOp> {
  using OpRewritePattern<GeluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GeluOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();

    auto outputType = cast<MemRefType>(output.getType());
    int rank = outputType.getRank();

    // GELU approximation constants: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Value c0_5 = createConstantFloat(rewriter, loc, 0.5f);
    Value c1 = createConstantFloat(rewriter, loc, 1.0f);
    Value cSqrt2OverPi = createConstantFloat(rewriter, loc, 0.7978845608f); // sqrt(2/pi)
    Value c0_044715 = createConstantFloat(rewriter, loc, 0.044715f);

    // Use linalg.generic for element-wise GELU activation
    SmallVector<AffineMap> indexingMaps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    indexingMaps.push_back(identityMap);  // input
    indexingMaps.push_back(identityMap);  // output

    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    // Capture constants for use in body builder
    Value capturedC0_5 = c0_5;
    Value capturedC1 = c1;
    Value capturedCSqrt2OverPi = cSqrt2OverPi;
    Value capturedC0_044715 = c0_044715;

    rewriter.create<linalg::GenericOp>(
        loc,
        ValueRange{input},
        ValueRange{output},
        indexingMaps,
        iteratorTypes,
        [capturedC0_5, capturedC1, capturedCSqrt2OverPi, capturedC0_044715]
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
          Value result = b.create<arith::MulFOp>(loc, halfX, onePlusTanh);

          b.create<linalg::YieldOp>(loc, result);
        }
    );

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AddOp Lowering
//===----------------------------------------------------------------------===//

struct AddOpLowering : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value output = op.getOutput();

    auto outputType = cast<MemRefType>(output.getType());
    int rank = outputType.getRank();

    // Create indexing maps: all identity (parallel element-wise operation)
    SmallVector<AffineMap> indexingMaps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    indexingMaps.push_back(identityMap);  // lhs
    indexingMaps.push_back(identityMap);  // rhs
    indexingMaps.push_back(identityMap);  // output

    // All dimensions are parallel (no reductions)
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    // Create linalg.generic operation
    rewriter.create<linalg::GenericOp>(
        loc,
        /*inputs=*/ValueRange{lhs, rhs},
        /*outputs=*/ValueRange{output},
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*bodyBuilder=*/[](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = lhs element, args[1] = rhs element
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        }
    );

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MatmulOp Lowering
//===----------------------------------------------------------------------===//

struct MatmulOpLowering : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value output = op.getOutput();

    auto outputType = cast<MemRefType>(output.getType());

    // Step 1: Initialize output to zero using linalg.fill
    // linalg.matmul uses accumulation semantics (C += A @ B), so we need C = 0 first
    Value zero = createConstantFloat(rewriter, loc, 0.0f);
    rewriter.create<linalg::FillOp>(loc, zero, output);

    // Step 2: Perform matrix multiplication using linalg.matmul
    // For both 2D and 3D (batched), linalg.matmul handles it automatically
    rewriter.create<linalg::MatmulOp>(
        loc,
        ValueRange{lhs, rhs},  // inputs
        ValueRange{output}     // output (accumulates into initialized buffer)
    );

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TransposeOp Lowering
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();

    auto inputType = cast<MemRefType>(input.getType());
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

    // Use linalg.transpose with the permutation
    rewriter.create<linalg::TransposeOp>(
        loc,
        input,
        output,
        permutation
    );

    rewriter.eraseOp(op);
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
    Value output = op.getOutput();

    auto inputType = cast<MemRefType>(input.getType());
    ArrayRef<int64_t> shape = inputType.getShape();
    int rank = shape.size();

    // Create temporary buffers for intermediate results
    SmallVector<int64_t> reducedShape(shape.begin(), shape.end() - 1);
    auto reducedType = MemRefType::get(reducedShape, rewriter.getF32Type());
    Value maxBuffer = rewriter.create<memref::AllocOp>(loc, reducedType);
    Value sumBuffer = rewriter.create<memref::AllocOp>(loc, reducedType);

    // Step 1: Find max along last dimension using linalg.reduce
    Value negInf = createConstantFloat(rewriter, loc, -1e9f);
    rewriter.create<linalg::FillOp>(loc, negInf, maxBuffer);

    rewriter.create<linalg::ReduceOp>(
        loc,
        ValueRange{input},
        ValueRange{maxBuffer},
        SmallVector<int64_t>{rank - 1},  // reduce along last dimension
        [](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = input element, args[1] = current max
          Value newMax = b.create<arith::MaximumFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, newMax);
        }
    );

    // Step 2: Compute exp(input - max) using linalg.generic
    // This broadcasts max across the last dimension and computes exp
    SmallVector<AffineMap> indexingMaps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<AffineExpr> reducedExprs;
    for (int i = 0; i < rank - 1; i++) {
      reducedExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    auto broadcastMap = AffineMap::get(rank, 0, reducedExprs, rewriter.getContext());

    indexingMaps.push_back(identityMap);     // input
    indexingMaps.push_back(broadcastMap);    // maxBuffer (broadcasted)
    indexingMaps.push_back(identityMap);     // output

    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    rewriter.create<linalg::GenericOp>(
        loc,
        ValueRange{input, maxBuffer},
        ValueRange{output},
        indexingMaps,
        iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = input element, args[1] = max value (broadcasted)
          Value shifted = b.create<arith::SubFOp>(loc, args[0], args[1]);
          Value expVal = b.create<math::ExpOp>(loc, shifted);
          b.create<linalg::YieldOp>(loc, expVal);
        }
    );

    // Step 3: Sum exp values along last dimension using linalg.reduce
    Value zero = createConstantFloat(rewriter, loc, 0.0f);
    rewriter.create<linalg::FillOp>(loc, zero, sumBuffer);

    rewriter.create<linalg::ReduceOp>(
        loc,
        ValueRange{output},  // input is now the exp values
        ValueRange{sumBuffer},
        SmallVector<int64_t>{rank - 1},  // reduce along last dimension
        [](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = exp element, args[1] = current sum
          Value newSum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, newSum);
        }
    );

    // Step 4: Normalize by dividing exp values by sum using linalg.generic
    rewriter.create<linalg::GenericOp>(
        loc,
        ValueRange{output, sumBuffer},  // exp values and sum
        ValueRange{output},              // in-place normalization
        SmallVector<AffineMap>{identityMap, broadcastMap, identityMap},
        iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = exp element, args[1] = sum (broadcasted)
          Value normalized = b.create<arith::DivFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, normalized);
        }
    );

    // Clean up temporary buffers
    rewriter.create<memref::DeallocOp>(loc, maxBuffer);
    rewriter.create<memref::DeallocOp>(loc, sumBuffer);

    rewriter.eraseOp(op);
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
    Value output = op.getOutput();

    auto outputType = cast<MemRefType>(output.getType());
    int rank = outputType.getRank();

    // Load scalar scale value from 1D memref
    Value zeroIdx = createConstantIndex(rewriter, loc, 0);
    Value scalarValue = rewriter.create<memref::LoadOp>(loc, scale, ValueRange{zeroIdx});

    // Use linalg.generic for element-wise multiplication
    SmallVector<AffineMap> indexingMaps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    indexingMaps.push_back(identityMap);  // input
    indexingMaps.push_back(identityMap);  // output

    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    // Capture the scalar value for use in body builder
    Value capturedScalar = scalarValue;

    rewriter.create<linalg::GenericOp>(
        loc,
        ValueRange{input},
        ValueRange{output},
        indexingMaps,
        iteratorTypes,
        [capturedScalar](OpBuilder &b, Location loc, ValueRange args) {
          Value product = b.create<arith::MulFOp>(loc, args[0], capturedScalar);
          b.create<linalg::YieldOp>(loc, product);
        }
    );

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower Transformer to Standard Pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerTransformerToStandardPass
    : public PassWrapper<LowerTransformerToStandardPass, OperationPass<func::FuncOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect,
                    memref::MemRefDialect, math::MathDialect,
                    linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LayerNormOpLowering, LinearOpLowering, GeluOpLowering, AddOpLowering,
                 MatmulOpLowering, TransposeOpLowering, SoftmaxOpLowering, ScaleOpLowering>(
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