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

    // Linear: output = input @ weight + bias
    // Ch13 convention: weight is already (in_features, out_features)
    // input: (seq_len, in_features), weight: (in_features, out_features)

    int64_t seqLen = inputType.getShape()[0];
    int64_t inFeatures = inputType.getShape()[1];
    int64_t outFeatures = weightType.getShape()[1];  // Ch13: weight is [in, out]

    // Step 1: Initialize output to zero and perform matmul
    Value zero = createConstantFloat(rewriter, loc, 0.0f);
    rewriter.create<linalg::FillOp>(loc, zero, output);

    // Matmul: (seq_len, in_features) @ (in_features, out_features) -> (seq_len, out_features)
    rewriter.create<linalg::MatmulOp>(
        loc,
        ValueRange{input, weight},  // No transpose needed for Ch13
        ValueRange{output}
    );

    // Step 2: Add bias using linalg.generic (broadcast bias across seq_len)
    int rank = 2;
    SmallVector<AffineMap> indexingMaps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());

    // Broadcast map for bias (only last dimension)
    SmallVector<AffineExpr> biasExprs;
    biasExprs.push_back(rewriter.getAffineDimExpr(1));  // only use second dimension
    AffineMap biasMap = AffineMap::get(rank, 0, biasExprs, rewriter.getContext());

    indexingMaps.push_back(biasMap);       // bias input (1D)
    indexingMaps.push_back(identityMap);   // output (2D)

    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    rewriter.create<linalg::GenericOp>(
        loc,
        TypeRange{},           // result types (output is updated in-place)
        ValueRange{bias},      // inputs
        ValueRange{output},    // outputs
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value biasVal = args[0];
          Value outputVal = args[1];
          Value sum = b.create<arith::AddFOp>(loc, outputVal, biasVal);
          b.create<linalg::YieldOp>(loc, sum);
        }
    );

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

    auto inputType = cast<MemRefType>(input.getType());
    auto shape = inputType.getShape();

    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Value c0_5 = createConstantFloat(rewriter, loc, 0.5f);
    Value c1 = createConstantFloat(rewriter, loc, 1.0f);
    Value cSqrt2OverPi = createConstantFloat(rewriter, loc, 0.7978845608f); // sqrt(2/pi)
    Value c0_044715 = createConstantFloat(rewriter, loc, 0.044715f);

    Value dim0Val = createConstantIndex(rewriter, loc, shape[0]);
    Value dim1Val = createConstantIndex(rewriter, loc, shape[1]);
    Value zeroIdx = createConstantIndex(rewriter, loc, 0);
    Value oneIdx = createConstantIndex(rewriter, loc, 1);

    rewriter.create<scf::ForOp>(
        loc, zeroIdx, dim0Val, oneIdx, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          builder.create<scf::ForOp>(
              loc, zeroIdx, dim1Val, oneIdx, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {
                Value x = builder.create<memref::LoadOp>(loc, input, ValueRange{i, j});

                // Compute x^3
                Value x2 = builder.create<arith::MulFOp>(loc, x, x);
                Value x3 = builder.create<arith::MulFOp>(loc, x2, x);

                // Compute 0.044715 * x^3
                Value term = builder.create<arith::MulFOp>(loc, c0_044715, x3);

                // Compute x + 0.044715 * x^3
                Value inner = builder.create<arith::AddFOp>(loc, x, term);

                // Compute sqrt(2/pi) * (x + 0.044715 * x^3)
                Value scaled = builder.create<arith::MulFOp>(loc, cSqrt2OverPi, inner);

                // Compute tanh(...)
                Value tanhVal = builder.create<math::TanhOp>(loc, scaled);

                // Compute 1 + tanh(...)
                Value onePlusTanh = builder.create<arith::AddFOp>(loc, c1, tanhVal);

                // Compute 0.5 * x * (1 + tanh(...))
                Value halfX = builder.create<arith::MulFOp>(loc, c0_5, x);
                Value result = builder.create<arith::MulFOp>(loc, halfX, onePlusTanh);

                builder.create<memref::StoreOp>(loc, result, output, ValueRange{i, j});
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });

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

    auto lhsType = cast<MemRefType>(lhs.getType());
    auto shape = lhsType.getShape();

    Value dim0Val = createConstantIndex(rewriter, loc, shape[0]);
    Value dim1Val = createConstantIndex(rewriter, loc, shape[1]);
    Value zeroIdx = createConstantIndex(rewriter, loc, 0);
    Value oneIdx = createConstantIndex(rewriter, loc, 1);

    rewriter.create<scf::ForOp>(
        loc, zeroIdx, dim0Val, oneIdx, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          builder.create<scf::ForOp>(
              loc, zeroIdx, dim1Val, oneIdx, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {
                Value lhsVal = builder.create<memref::LoadOp>(loc, lhs, ValueRange{i, j});
                Value rhsVal = builder.create<memref::LoadOp>(loc, rhs, ValueRange{i, j});
                Value result = builder.create<arith::AddFOp>(loc, lhsVal, rhsVal);
                builder.create<memref::StoreOp>(loc, result, output, ValueRange{i, j});
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });

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

    auto lhsType = cast<MemRefType>(lhs.getType());
    auto rhsType = cast<MemRefType>(rhs.getType());

    // Support 2D matmul: (M, K) @ (K, N) -> (M, N)
    int64_t M = lhsType.getShape()[0];
    int64_t K = lhsType.getShape()[1];
    int64_t N = rhsType.getShape()[1];

    Value zero = createConstantFloat(rewriter, loc, 0.0f);
    Value MVal = createConstantIndex(rewriter, loc, M);
    Value NVal = createConstantIndex(rewriter, loc, N);
    Value KVal = createConstantIndex(rewriter, loc, K);
    Value zeroIdx = createConstantIndex(rewriter, loc, 0);
    Value oneIdx = createConstantIndex(rewriter, loc, 1);

    // output[i, j] = sum_k(lhs[i, k] * rhs[k, j])
    rewriter.create<scf::ForOp>(
        loc, zeroIdx, MVal, oneIdx, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          builder.create<scf::ForOp>(
              loc, zeroIdx, NVal, oneIdx, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {
                Value sum = builder.create<scf::ForOp>(
                    loc, zeroIdx, KVal, oneIdx, ValueRange{zero},
                    [&](OpBuilder &builder, Location loc, Value k, ValueRange iterArgs) {
                      Value currentSum = iterArgs[0];
                      Value lhsVal = builder.create<memref::LoadOp>(loc, lhs, ValueRange{i, k});
                      Value rhsVal = builder.create<memref::LoadOp>(loc, rhs, ValueRange{k, j});
                      Value prod = builder.create<arith::MulFOp>(loc, lhsVal, rhsVal);
                      Value newSum = builder.create<arith::AddFOp>(loc, currentSum, prod);
                      builder.create<scf::YieldOp>(loc, newSum);
                    }).getResult(0);

                builder.create<memref::StoreOp>(loc, sum, output, ValueRange{i, j});
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });

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
    auto shape = inputType.getShape();

    // Transpose: input is (dim0, dim1), output is (dim1, dim0)
    int64_t inputDim0 = shape[0];
    int64_t inputDim1 = shape[1];

    Value inputDim0Val = createConstantIndex(rewriter, loc, inputDim0);
    Value inputDim1Val = createConstantIndex(rewriter, loc, inputDim1);
    Value zeroIdx = createConstantIndex(rewriter, loc, 0);
    Value oneIdx = createConstantIndex(rewriter, loc, 1);

    // output[i, j] = input[j, i]
    // Output shape is (inputDim1, inputDim0), so iterate accordingly
    rewriter.create<scf::ForOp>(
        loc, zeroIdx, inputDim1Val, oneIdx, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          builder.create<scf::ForOp>(
              loc, zeroIdx, inputDim0Val, oneIdx, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {
                Value val = builder.create<memref::LoadOp>(loc, input, ValueRange{j, i});
                builder.create<memref::StoreOp>(loc, val, output, ValueRange{i, j});
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });

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
    auto shape = inputType.getShape();

    int64_t rows = shape[0];
    int64_t cols = shape[1];

    Value negInf = createConstantFloat(rewriter, loc, -std::numeric_limits<float>::infinity());
    Value zero = createConstantFloat(rewriter, loc, 0.0f);
    Value rowsVal = createConstantIndex(rewriter, loc, rows);
    Value colsVal = createConstantIndex(rewriter, loc, cols);
    Value zeroIdx = createConstantIndex(rewriter, loc, 0);
    Value oneIdx = createConstantIndex(rewriter, loc, 1);

    // For each row: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    rewriter.create<scf::ForOp>(
        loc, zeroIdx, rowsVal, oneIdx, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          // Step 1: Find max
          Value maxVal = builder.create<scf::ForOp>(
              loc, zeroIdx, colsVal, oneIdx, ValueRange{negInf},
              [&](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {
                Value currentMax = iterArgs[0];
                Value val = builder.create<memref::LoadOp>(loc, input, ValueRange{i, j});
                Value newMax = builder.create<arith::MaximumFOp>(loc, currentMax, val);
                builder.create<scf::YieldOp>(loc, newMax);
              }).getResult(0);

          // Step 2: Compute sum of exp(x - max)
          Value sumExp = builder.create<scf::ForOp>(
              loc, zeroIdx, colsVal, oneIdx, ValueRange{zero},
              [&](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {
                Value currentSum = iterArgs[0];
                Value val = builder.create<memref::LoadOp>(loc, input, ValueRange{i, j});
                Value shifted = builder.create<arith::SubFOp>(loc, val, maxVal);
                Value expVal = builder.create<math::ExpOp>(loc, shifted);
                Value newSum = builder.create<arith::AddFOp>(loc, currentSum, expVal);
                builder.create<scf::YieldOp>(loc, newSum);
              }).getResult(0);

          // Step 3: Normalize
          builder.create<scf::ForOp>(
              loc, zeroIdx, colsVal, oneIdx, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {
                Value val = builder.create<memref::LoadOp>(loc, input, ValueRange{i, j});
                Value shifted = builder.create<arith::SubFOp>(loc, val, maxVal);
                Value expVal = builder.create<math::ExpOp>(loc, shifted);
                Value result = builder.create<arith::DivFOp>(loc, expVal, sumExp);
                builder.create<memref::StoreOp>(loc, result, output, ValueRange{i, j});
                builder.create<scf::YieldOp>(loc);
              });

          builder.create<scf::YieldOp>(loc);
        });

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

    auto inputType = cast<MemRefType>(input.getType());
    auto shape = inputType.getShape();

    // Load scalar scale value
    Value zeroIdx = createConstantIndex(rewriter, loc, 0);
    Value scaleVal = rewriter.create<memref::LoadOp>(loc, scale, ValueRange{zeroIdx});

    Value dim0Val = createConstantIndex(rewriter, loc, shape[0]);
    Value dim1Val = createConstantIndex(rewriter, loc, shape[1]);
    Value oneIdx = createConstantIndex(rewriter, loc, 1);

    rewriter.create<scf::ForOp>(
        loc, zeroIdx, dim0Val, oneIdx, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          builder.create<scf::ForOp>(
              loc, zeroIdx, dim1Val, oneIdx, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {
                Value val = builder.create<memref::LoadOp>(loc, input, ValueRange{i, j});
                Value result = builder.create<arith::MulFOp>(loc, val, scaleVal);
                builder.create<memref::StoreOp>(loc, result, output, ValueRange{i, j});
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Chapter 13: GPT-specific Operations Lowering
//===----------------------------------------------------------------------===//

struct EmbeddingOpLowering : public OpRewritePattern<EmbeddingOp> {
  using OpRewritePattern<EmbeddingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(EmbeddingOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value indices = op.getIndices();
    Value table = op.getTable();
    Value output = op.getOutput();

    auto indicesType = cast<MemRefType>(indices.getType());
    auto tableType = cast<MemRefType>(table.getType());

    int64_t seqLen = indicesType.getShape()[0];
    int64_t dModel = tableType.getShape()[1];

    Value seqLenVal = createConstantIndex(rewriter, loc, seqLen);
    Value dModelVal = createConstantIndex(rewriter, loc, dModel);
    Value zeroIdx = createConstantIndex(rewriter, loc, 0);
    Value oneIdx = createConstantIndex(rewriter, loc, 1);

    // For each position in sequence
    rewriter.create<scf::ForOp>(
        loc, zeroIdx, seqLenVal, oneIdx, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          // Load token ID (int32)
          Value tokenId32 = builder.create<memref::LoadOp>(loc, indices, ValueRange{i});

          // Convert int32 token ID to index type
          Value tokenIdx = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), tokenId32);

          // Copy embedding vector from table to output
          builder.create<scf::ForOp>(
              loc, zeroIdx, dModelVal, oneIdx, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {
                Value embVal = builder.create<memref::LoadOp>(loc, table, ValueRange{tokenIdx, j});
                builder.create<memref::StoreOp>(loc, embVal, output, ValueRange{i, j});
                builder.create<scf::YieldOp>(loc);
              });

          builder.create<scf::YieldOp>(loc);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Create Causal Mask Lowering (Chapter 13)
//===----------------------------------------------------------------------===//

struct CreateCausalMaskOpLowering : public OpRewritePattern<CreateCausalMaskOp> {
  using OpRewritePattern<CreateCausalMaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CreateCausalMaskOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value output = op.getOutput();

    // Get mask dimensions [seq_len, seq_len]
    auto outputType = cast<MemRefType>(output.getType());
    auto shape = outputType.getShape();

    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    Value seqLenVal;
    if (shape[0] == ShapedType::kDynamic) {
      seqLenVal = rewriter.create<memref::DimOp>(loc, output, 0);
    } else {
      seqLenVal = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);
    }

    // Constants for mask values
    Value zeroFloat = rewriter.create<arith::ConstantFloatOp>(
        loc, llvm::APFloat(0.0f), rewriter.getF32Type());
    Value negInf = rewriter.create<arith::ConstantFloatOp>(
        loc, llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(), /*Negative=*/true),
        rewriter.getF32Type());

    // Generate mask: mask[i][j] = (j <= i) ? 0.0 : -inf
    rewriter.create<scf::ForOp>(
        loc, zeroIdx, seqLenVal, oneIdx, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          builder.create<scf::ForOp>(
              loc, zeroIdx, seqLenVal, oneIdx, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {
                // Compare j <= i
                Value cmp = builder.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::sle, j, i);
                // Select: j <= i ? 0.0 : -inf
                Value maskValue = builder.create<arith::SelectOp>(
                    loc, cmp, zeroFloat, negInf);
                builder.create<memref::StoreOp>(loc, maskValue, output, ValueRange{i, j});
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Masked Softmax Lowering (Chapter 13)
//===----------------------------------------------------------------------===//

struct MaskedSoftmaxOpLowering : public OpRewritePattern<MaskedSoftmaxOp> {
  using OpRewritePattern<MaskedSoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MaskedSoftmaxOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value mask = op.getMask();
    Value output = op.getOutput();

    // Get dimensions
    auto inputType = cast<MemRefType>(input.getType());
    auto shape = inputType.getShape();
    size_t ndim = shape.size();

    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Handle both 2D [seq_len, seq_len] and 3D [batch, seq_len, seq_len]
    Value dim0Val, dim1Val, dim2Val;
    bool is2D = (ndim == 2);

    if (is2D) {
      // 2D case: treat as batch=1, process [seq_len, seq_len]
      dim0Val = oneIdx;  // batch size = 1
      if (shape[0] == ShapedType::kDynamic) {
        dim1Val = rewriter.create<memref::DimOp>(loc, input, 0);
      } else {
        dim1Val = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);
      }
      if (shape[1] == ShapedType::kDynamic) {
        dim2Val = rewriter.create<memref::DimOp>(loc, input, 1);
      } else {
        dim2Val = rewriter.create<arith::ConstantIndexOp>(loc, shape[1]);
      }
    } else {
      // 3D case
      if (shape[0] == ShapedType::kDynamic) {
        dim0Val = rewriter.create<memref::DimOp>(loc, input, 0);
      } else {
        dim0Val = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);
      }
      if (shape[1] == ShapedType::kDynamic) {
        dim1Val = rewriter.create<memref::DimOp>(loc, input, 1);
      } else {
        dim1Val = rewriter.create<arith::ConstantIndexOp>(loc, shape[1]);
      }
      if (shape[2] == ShapedType::kDynamic) {
        dim2Val = rewriter.create<memref::DimOp>(loc, input, 2);
      } else {
        dim2Val = rewriter.create<arith::ConstantIndexOp>(loc, shape[2]);
      }
    }

    Value zero = rewriter.create<arith::ConstantFloatOp>(
        loc, llvm::APFloat(0.0f), rewriter.getF32Type());
    Value negInfInit = rewriter.create<arith::ConstantFloatOp>(
        loc, llvm::APFloat::getLargest(llvm::APFloat::IEEEsingle(), /*Negative=*/true),
        rewriter.getF32Type());

    // For each position in batch and row
    rewriter.create<scf::ForOp>(
        loc, zeroIdx, dim0Val, oneIdx, ValueRange{},
        [&, is2D](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          builder.create<scf::ForOp>(
              loc, zeroIdx, dim1Val, oneIdx, ValueRange{},
              [&, is2D](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {

                // Step 1: Find max for numerical stability
                Value maxVal = builder.create<scf::ForOp>(
                    loc, zeroIdx, dim2Val, oneIdx, 
                    ValueRange{negInfInit},
                    [&, is2D](OpBuilder &builder, Location loc, Value k, ValueRange iterArgs) {
                      Value currentMax = iterArgs[0];
                      // Load input: 2D uses [j,k], 3D uses [i,j,k]
                      Value logit = is2D 
                          ? builder.create<memref::LoadOp>(loc, input, ValueRange{j, k})
                          : builder.create<memref::LoadOp>(loc, input, ValueRange{i, j, k});
                      Value maskVal = builder.create<memref::LoadOp>(loc, mask, ValueRange{j, k});
                      Value maskedLogit = builder.create<arith::AddFOp>(loc, logit, maskVal);
                      Value newMax = builder.create<arith::MaximumFOp>(loc, currentMax, maskedLogit);
                      builder.create<scf::YieldOp>(loc, newMax);
                    }).getResult(0);

                // Step 2: Compute exp sum
                Value expSum = builder.create<scf::ForOp>(
                    loc, zeroIdx, dim2Val, oneIdx, ValueRange{zero},
                    [&, is2D](OpBuilder &builder, Location loc, Value k, ValueRange iterArgs) {
                      Value currentSum = iterArgs[0];
                      Value logit = is2D
                          ? builder.create<memref::LoadOp>(loc, input, ValueRange{j, k})
                          : builder.create<memref::LoadOp>(loc, input, ValueRange{i, j, k});
                      Value maskVal = builder.create<memref::LoadOp>(loc, mask, ValueRange{j, k});
                      Value maskedLogit = builder.create<arith::AddFOp>(loc, logit, maskVal);
                      Value shifted = builder.create<arith::SubFOp>(loc, maskedLogit, maxVal);
                      Value expVal = builder.create<math::ExpOp>(loc, shifted);
                      Value newSum = builder.create<arith::AddFOp>(loc, currentSum, expVal);
                      builder.create<scf::YieldOp>(loc, newSum);
                    }).getResult(0);

                // Step 3: Compute softmax
                builder.create<scf::ForOp>(
                    loc, zeroIdx, dim2Val, oneIdx, ValueRange{},
                    [&, is2D](OpBuilder &builder, Location loc, Value k, ValueRange iterArgs) {
                      Value logit = is2D
                          ? builder.create<memref::LoadOp>(loc, input, ValueRange{j, k})
                          : builder.create<memref::LoadOp>(loc, input, ValueRange{i, j, k});
                      Value maskVal = builder.create<memref::LoadOp>(loc, mask, ValueRange{j, k});
                      Value maskedLogit = builder.create<arith::AddFOp>(loc, logit, maskVal);
                      Value shifted = builder.create<arith::SubFOp>(loc, maskedLogit, maxVal);
                      Value expVal = builder.create<math::ExpOp>(loc, shifted);
                      Value result = builder.create<arith::DivFOp>(loc, expVal, expSum);
                      // Store output: 2D uses [j,k], 3D uses [i,j,k]
                      if (is2D) {
                        builder.create<memref::StoreOp>(loc, result, output, ValueRange{j, k});
                      } else {
                        builder.create<memref::StoreOp>(loc, result, output, ValueRange{i, j, k});
                      }
                      builder.create<scf::YieldOp>(loc);
                    });

                builder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RoPE (Rotary Position Embeddings) Lowering (Chapter 13)
//===----------------------------------------------------------------------===//

struct RoPEOpLowering : public OpRewritePattern<RoPEOp> {
  using OpRewritePattern<RoPEOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RoPEOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();

    // Get dimensions
    auto inputType = cast<MemRefType>(input.getType());
    auto shape = inputType.getShape();
    size_t ndim = shape.size();

    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value twoIdx = rewriter.create<arith::ConstantIndexOp>(loc, 2);

    // Get dimension values
    Value seqLenVal, dModelVal;
    if (ndim == 2) {
      // Shape: [seq_len, d_model]
      if (shape[0] == ShapedType::kDynamic) {
        seqLenVal = rewriter.create<memref::DimOp>(loc, input, 0);
      } else {
        seqLenVal = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);
      }
      if (shape[1] == ShapedType::kDynamic) {
        dModelVal = rewriter.create<memref::DimOp>(loc, input, 1);
      } else {
        dModelVal = rewriter.create<arith::ConstantIndexOp>(loc, shape[1]);
      }
    } else if (ndim == 3) {
      // Shape: [batch, seq_len, d_model] - handle later if needed
      return failure();
    } else {
      return failure();
    }

    // Constants for RoPE computation
    Value base = rewriter.create<arith::ConstantFloatOp>(
        loc, llvm::APFloat(10000.0f), rewriter.getF32Type());

    // For each position and dimension pair
    rewriter.create<scf::ForOp>(
        loc, zeroIdx, seqLenVal, oneIdx, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value pos, ValueRange iterArgs) {
          // pos is the position index (i) - convert index → i64 → f32
          Value posI64 = builder.create<arith::IndexCastOp>(
              loc, builder.getI64Type(), pos);
          Value posFloat = builder.create<arith::SIToFPOp>(
              loc, builder.getF32Type(), posI64);

          // Process dimension pairs: (0,1), (2,3), (4,5), ...
          builder.create<scf::ForOp>(
              loc, zeroIdx, dModelVal, twoIdx, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value dimIdx, ValueRange iterArgs) {
                // dimIdx is 2j (even dimension index)
                Value dimIdxPlus1 = builder.create<arith::AddIOp>(loc, dimIdx, oneIdx);

                // Compute θ = base^(-2j/d_model)
                // First: 2j / d_model
                Value dimIdxI64 = builder.create<arith::IndexCastOp>(
                    loc, builder.getI64Type(), dimIdx);
                Value dimIdxFloat = builder.create<arith::SIToFPOp>(
                    loc, builder.getF32Type(), dimIdxI64);
                Value dModelI64 = builder.create<arith::IndexCastOp>(
                    loc, builder.getI64Type(), dModelVal);
                Value dModelFloat = builder.create<arith::SIToFPOp>(
                    loc, builder.getF32Type(), dModelI64);
                Value ratio = builder.create<arith::DivFOp>(loc, dimIdxFloat, dModelFloat);

                // Compute: base^(-ratio) = exp(-ratio * log(base))
                Value logBase = builder.create<math::LogOp>(loc, base);
                Value negRatio = builder.create<arith::NegFOp>(loc, ratio);
                Value exponent = builder.create<arith::MulFOp>(loc, negRatio, logBase);
                Value theta = builder.create<math::ExpOp>(loc, exponent);

                // Compute: angle = pos * theta
                Value angle = builder.create<arith::MulFOp>(loc, posFloat, theta);

                // Compute: cos(angle), sin(angle)
                Value cosAngle = builder.create<math::CosOp>(loc, angle);
                Value sinAngle = builder.create<math::SinOp>(loc, angle);

                // Load input values: x0 = input[pos, 2j], x1 = input[pos, 2j+1]
                Value x0 = builder.create<memref::LoadOp>(loc, input, ValueRange{pos, dimIdx});
                Value x1 = builder.create<memref::LoadOp>(loc, input, ValueRange{pos, dimIdxPlus1});

                // Apply rotation:
                // output[pos, 2j]   = x0 * cos(angle) - x1 * sin(angle)
                // output[pos, 2j+1] = x0 * sin(angle) + x1 * cos(angle)
                Value x0_cos = builder.create<arith::MulFOp>(loc, x0, cosAngle);
                Value x1_sin = builder.create<arith::MulFOp>(loc, x1, sinAngle);
                Value out0 = builder.create<arith::SubFOp>(loc, x0_cos, x1_sin);

                Value x0_sin = builder.create<arith::MulFOp>(loc, x0, sinAngle);
                Value x1_cos = builder.create<arith::MulFOp>(loc, x1, cosAngle);
                Value out1 = builder.create<arith::AddFOp>(loc, x0_sin, x1_cos);

                // Store output values
                builder.create<memref::StoreOp>(loc, out0, output, ValueRange{pos, dimIdx});
                builder.create<memref::StoreOp>(loc, out1, output, ValueRange{pos, dimIdxPlus1});

                builder.create<scf::YieldOp>(loc);
              });

          builder.create<scf::YieldOp>(loc);
        });

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
                    memref::MemRefDialect, math::MathDialect>();
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