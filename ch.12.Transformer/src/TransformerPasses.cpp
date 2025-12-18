//===- TransformerPasses.cpp - Lower Transformer to Standard -----*- C++ -*-===//
#include "TransformerPasses.h"
#include "TransformerOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
    float epsilon = 1e-5f; // Hardcoded for now (avoid BytecodeOpInterface issues)

    auto inputType = cast<MemRefType>(input.getType());

    int64_t seqLen = inputType.getShape()[0];
    int64_t dModel = inputType.getShape()[1];

    Value epsilonVal = createConstantFloat(rewriter, loc, epsilon);
    Value zero = createConstantFloat(rewriter, loc, 0.0f);
    Value dModelFloat = createConstantFloat(rewriter, loc, static_cast<float>(dModel));

    // For each sequence position
    Value seqLenVal = createConstantIndex(rewriter, loc, seqLen);
    Value dModelVal = createConstantIndex(rewriter, loc, dModel);
    Value zeroIdx = createConstantIndex(rewriter, loc, 0);
    Value oneIdx = createConstantIndex(rewriter, loc, 1);

    rewriter.create<scf::ForOp>(
        loc, zeroIdx, seqLenVal, oneIdx, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          // Step 1: Compute mean
          // mean = sum(input[i, :]) / d_model
          Value sum = zero;
          builder.create<scf::ForOp>(
              loc, zeroIdx, dModelVal, oneIdx, ValueRange{sum},
              [&](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {
                Value currentSum = iterArgs[0];
                Value val = builder.create<memref::LoadOp>(loc, input, ValueRange{i, j});
                Value newSum = builder.create<arith::AddFOp>(loc, currentSum, val);
                builder.create<scf::YieldOp>(loc, newSum);
              });

          Value sumResult = builder.create<scf::ForOp>(
              loc, zeroIdx, dModelVal, oneIdx, ValueRange{zero},
              [&](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {
                Value currentSum = iterArgs[0];
                Value val = builder.create<memref::LoadOp>(loc, input, ValueRange{i, j});
                Value newSum = builder.create<arith::AddFOp>(loc, currentSum, val);
                builder.create<scf::YieldOp>(loc, newSum);
              }).getResult(0);

          Value mean = builder.create<arith::DivFOp>(loc, sumResult, dModelFloat);

          // Step 2: Compute variance
          // variance = sum((input[i, :] - mean)^2) / d_model
          Value varianceSum = builder.create<scf::ForOp>(
              loc, zeroIdx, dModelVal, oneIdx, ValueRange{zero},
              [&](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {
                Value currentSum = iterArgs[0];
                Value val = builder.create<memref::LoadOp>(loc, input, ValueRange{i, j});
                Value diff = builder.create<arith::SubFOp>(loc, val, mean);
                Value diffSq = builder.create<arith::MulFOp>(loc, diff, diff);
                Value newSum = builder.create<arith::AddFOp>(loc, currentSum, diffSq);
                builder.create<scf::YieldOp>(loc, newSum);
              }).getResult(0);

          Value variance = builder.create<arith::DivFOp>(loc, varianceSum, dModelFloat);

          // Step 3: Compute rsqrt(variance + epsilon)
          Value variancePlusEps = builder.create<arith::AddFOp>(loc, variance, epsilonVal);
          Value invStd = builder.create<math::RsqrtOp>(loc, variancePlusEps);

          // Step 4: Normalize and apply scale/shift
          // output[i, j] = ((input[i, j] - mean) * invStd) * gamma[j] + beta[j]
          builder.create<scf::ForOp>(
              loc, zeroIdx, dModelVal, oneIdx, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {
                Value val = builder.create<memref::LoadOp>(loc, input, ValueRange{i, j});
                Value gammaVal = builder.create<memref::LoadOp>(loc, gamma, ValueRange{j});
                Value betaVal = builder.create<memref::LoadOp>(loc, beta, ValueRange{j});

                // Normalize: (val - mean) * invStd
                Value centered = builder.create<arith::SubFOp>(loc, val, mean);
                Value normalized = builder.create<arith::MulFOp>(loc, centered, invStd);

                // Scale and shift: normalized * gamma + beta
                Value scaled = builder.create<arith::MulFOp>(loc, normalized, gammaVal);
                Value result = builder.create<arith::AddFOp>(loc, scaled, betaVal);

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

    int64_t seqLen = inputType.getShape()[0];
    int64_t inFeatures = inputType.getShape()[1];
    int64_t outFeatures = weightType.getShape()[0];

    Value zero = createConstantFloat(rewriter, loc, 0.0f);
    Value seqLenVal = createConstantIndex(rewriter, loc, seqLen);
    Value outFeaturesVal = createConstantIndex(rewriter, loc, outFeatures);
    Value inFeaturesVal = createConstantIndex(rewriter, loc, inFeatures);
    Value zeroIdx = createConstantIndex(rewriter, loc, 0);
    Value oneIdx = createConstantIndex(rewriter, loc, 1);

    // output[i, j] = sum_k(input[i, k] * weight[j, k]) + bias[j]
    rewriter.create<scf::ForOp>(
        loc, zeroIdx, seqLenVal, oneIdx, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          builder.create<scf::ForOp>(
              loc, zeroIdx, outFeaturesVal, oneIdx, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value j, ValueRange iterArgs) {
                // Compute matmul: sum_k(input[i, k] * weight[j, k])
                Value sum = builder.create<scf::ForOp>(
                    loc, zeroIdx, inFeaturesVal, oneIdx, ValueRange{zero},
                    [&](OpBuilder &builder, Location loc, Value k, ValueRange iterArgs) {
                      Value currentSum = iterArgs[0];
                      Value inputVal = builder.create<memref::LoadOp>(loc, input, ValueRange{i, k});
                      Value weightVal = builder.create<memref::LoadOp>(loc, weight, ValueRange{j, k});
                      Value prod = builder.create<arith::MulFOp>(loc, inputVal, weightVal);
                      Value newSum = builder.create<arith::AddFOp>(loc, currentSum, prod);
                      builder.create<scf::YieldOp>(loc, newSum);
                    }).getResult(0);

                // Add bias
                Value biasVal = builder.create<memref::LoadOp>(loc, bias, ValueRange{j});
                Value result = builder.create<arith::AddFOp>(loc, sum, biasVal);

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
                 MatmulOpLowering, TransposeOpLowering, SoftmaxOpLowering, ScaleOpLowering>(
        &getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createLowerTransformerToStandardPass() {
  return std::make_unique<LowerTransformerToStandardPass>();
}