//===- NNToStandard.cpp - Convert NN dialect to standard ------------------===//
//
// Chapter 9: Lowering patterns for NN dialect operations
//
//===----------------------------------------------------------------------===//
#include "NNToStandard.h"
#include "NNDialect.h"
#include "NNOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <limits>

using namespace mlir;
using namespace mlir::nn;

//===----------------------------------------------------------------------===//
// NN Add Lowering
//===----------------------------------------------------------------------===//

struct NNAddOpLowering : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                  PatternRewriter &rewriter) const override {
    // nn.add -> linalg.generic with arith.addf
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto rank = resultType.getRank();

    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    // Create linalg.generic for element-wise addition
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
    SmallVector<AffineMap> indexingMaps(3, rewriter.getMultiDimIdentityMap(rank));

    Value result = rewriter.create<linalg::GenericOp>(
        loc, 
        resultType,
        ValueRange{op.getLhs(), op.getRhs()},  // inputs
        ValueRange{emptyTensor},                // outputs
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// NN Mul Lowering
//===----------------------------------------------------------------------===//

struct NNMulOpLowering : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                  PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto rank = resultType.getRank();

    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    // Create linalg.generic for element-wise multiplication
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
    SmallVector<AffineMap> indexingMaps(3, rewriter.getMultiDimIdentityMap(rank));

    Value result = rewriter.create<linalg::GenericOp>(
        loc,
        resultType,
        ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{emptyTensor},
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value product = b.create<arith::MulFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, product);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// NN MatMul Lowering
//===----------------------------------------------------------------------===//

struct NNMatMulOpLowering : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern<MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op,
                                  PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    // Create zero constant for initialization
    auto zeroAttr = rewriter.getFloatAttr(resultType.getElementType(), 0.0);
    Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);

    // Create empty tensor and fill with zeros
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    Value zeroTensor = rewriter.create<linalg::FillOp>(
        loc, zero, emptyTensor).result();

    // Create linalg.matmul
    Value result = rewriter.create<linalg::MatmulOp>(
        loc, resultType,
        ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{zeroTensor}).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// NN ReLU Lowering
//===----------------------------------------------------------------------===//

struct NNReLUOpLowering : public OpRewritePattern<ReLUOp> {
  using OpRewritePattern<ReLUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReLUOp op,
                                  PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto rank = resultType.getRank();

    // Create zero constant
    auto zeroAttr = rewriter.getFloatAttr(resultType.getElementType(), 0.0);
    auto zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);

    // Create empty output tensor
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    // Create linalg.generic for ReLU
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
    SmallVector<AffineMap> indexingMaps(2, rewriter.getMultiDimIdentityMap(rank));

    Value result = rewriter.create<linalg::GenericOp>(
        loc,
        resultType,
        ValueRange{op.getInput()},
        ValueRange{emptyTensor},
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // max(0, x)
          Value maxVal = b.create<arith::MaximumFOp>(loc, zero, args[0]);
          b.create<linalg::YieldOp>(loc, maxVal);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// NN Softmax Lowering
//===----------------------------------------------------------------------===//

struct NNSoftmaxOpLowering : public OpRewritePattern<SoftmaxOp> {
  using OpRewritePattern<SoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto type = cast<RankedTensorType>(input.getType());
    auto shape = type.getShape();
    auto rank = type.getRank();
    auto elemType = type.getElementType();

    // We assume the reduction is on the last dimension
    int64_t reductionDim = rank - 1;

    // 1. Compute Max along last dim
    // Shape for reduction: drop last dim
    SmallVector<int64_t> reducedShape;
    for (int i = 0; i < rank - 1; ++i) reducedShape.push_back(shape[i]);

    auto reducedType = RankedTensorType::get(reducedShape, elemType);

    // Init tensor for reduction (filled with min float)
    Value minVal = rewriter.create<arith::ConstantOp>(loc, 
        rewriter.getFloatAttr(elemType, -std::numeric_limits<float>::infinity()));
    Value emptyReduced = rewriter.create<tensor::EmptyOp>(loc, reducedShape, elemType);
    Value initMax = rewriter.create<linalg::FillOp>(loc, minVal, emptyReduced).result();

    // Maps for reduction: 
    // Input: (d0, d1) -> (d0, d1)
    // Output: (d0, d1) -> (d0)
    SmallVector<AffineMap> reductionMaps;
    reductionMaps.push_back(rewriter.getMultiDimIdentityMap(rank)); // Input

    // Output map drops the last dim
    SmallVector<AffineExpr> exprs;
    for (int i = 0; i < rank - 1; ++i) exprs.push_back(rewriter.getAffineDimExpr(i));
    reductionMaps.push_back(AffineMap::get(rank, 0, exprs, rewriter.getContext()));

    SmallVector<utils::IteratorType> reductionIterators(rank, utils::IteratorType::parallel);
    reductionIterators[reductionDim] = utils::IteratorType::reduction;

    Value maxVal = rewriter.create<linalg::GenericOp>(
        loc, reducedType, ValueRange{input}, ValueRange{initMax},
        reductionMaps, reductionIterators,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value max = b.create<arith::MaximumFOp>(loc, args[0], args[1]);
            b.create<linalg::YieldOp>(loc, max);
        }).getResult(0);

    // 2. Compute Exp(x - max)
    // We need to broadcast max back to (d0, d1)
    // Maps: Input (d0, d1), Max (d0), Output (d0, d1)
    SmallVector<AffineMap> broadcastMaps = {
        rewriter.getMultiDimIdentityMap(rank), // Input
        reductionMaps[1],                      // Max (broadcast)
        rewriter.getMultiDimIdentityMap(rank)  // Output
    };
    SmallVector<utils::IteratorType> parallelIterators(rank, utils::IteratorType::parallel);

    Value emptyOutput = rewriter.create<tensor::EmptyOp>(loc, shape, elemType);

    Value expVals = rewriter.create<linalg::GenericOp>(
        loc, type, ValueRange{input, maxVal}, ValueRange{emptyOutput},
        broadcastMaps, parallelIterators,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value sub = b.create<arith::SubFOp>(loc, args[0], args[1]);
            Value exp = b.create<math::ExpOp>(loc, sub);
            b.create<linalg::YieldOp>(loc, exp);
        }).getResult(0);

    // 3. Compute Sum of Exp
    Value zeroVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 0.0));
    Value initSum = rewriter.create<linalg::FillOp>(loc, zeroVal, emptyReduced).result();

    Value sumVal = rewriter.create<linalg::GenericOp>(
        loc, reducedType, ValueRange{expVals}, ValueRange{initSum},
        reductionMaps, reductionIterators,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value add = b.create<arith::AddFOp>(loc, args[0], args[1]);
            b.create<linalg::YieldOp>(loc, add);
        }).getResult(0);

    // 4. Compute Div (Exp / Sum)
    Value result = rewriter.create<linalg::GenericOp>(
        loc, type, ValueRange{expVals, sumVal}, ValueRange{emptyOutput},
        broadcastMaps, parallelIterators,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value div = b.create<arith::DivFOp>(loc, args[0], args[1]);
            b.create<linalg::YieldOp>(loc, div);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// NN Linear Lowering
//===----------------------------------------------------------------------===//

struct NNLinearOpLowering : public OpRewritePattern<LinearOp> {
  using OpRewritePattern<LinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LinearOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = op.getInput();
    Value weight = op.getWeight();
    Value bias = op.getBias();

    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto elemType = resultType.getElementType();

    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), elemType);

    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 0.0));
    Value zeroTensor = rewriter.create<linalg::FillOp>(loc, zero, emptyTensor).result();

    // Maps for MatMul with Transpose B:
    // C[m, n] += A[m, k] * B[n, k]
    // A: (m, k) -> (d0, d2)
    // B: (n, k) -> (d1, d2)
    // C: (m, n) -> (d0, d1)

    SmallVector<AffineMap> matmulMaps;
    auto context = rewriter.getContext();
    matmulMaps.push_back(AffineMap::get(3, 0, {getAffineDimExpr(0, context), getAffineDimExpr(2, context)}, context)); // A
    matmulMaps.push_back(AffineMap::get(3, 0, {getAffineDimExpr(1, context), getAffineDimExpr(2, context)}, context)); // B
    matmulMaps.push_back(AffineMap::get(3, 0, {getAffineDimExpr(0, context), getAffineDimExpr(1, context)}, context)); // C

    SmallVector<utils::IteratorType> matmulIterators = {
        utils::IteratorType::parallel, 
        utils::IteratorType::parallel, 
        utils::IteratorType::reduction
    };

    Value matmulResult = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{input, weight}, ValueRange{zeroTensor},
        matmulMaps, matmulIterators,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value mul = b.create<arith::MulFOp>(loc, args[0], args[1]);
            Value add = b.create<arith::AddFOp>(loc, mul, args[2]);
            b.create<linalg::YieldOp>(loc, add);
        }).getResult(0);

    // 2. Add Bias if present
    if (bias) {
        // Bias is [N]. Broadcast to [M, N]
        // Maps:
        // MatMulResult: (d0, d1) -> (d0, d1)
        // Bias: (d1) -> (d1)
        // Output: (d0, d1) -> (d0, d1)

        SmallVector<AffineMap> biasMaps;
        biasMaps.push_back(rewriter.getMultiDimIdentityMap(2)); // Input
        biasMaps.push_back(AffineMap::get(2, 0, {getAffineDimExpr(1, context)}, context)); // Bias
        biasMaps.push_back(rewriter.getMultiDimIdentityMap(2)); // Output

        SmallVector<utils::IteratorType> biasIterators = {
            utils::IteratorType::parallel, utils::IteratorType::parallel
        };

        Value biasResult = rewriter.create<linalg::GenericOp>(
            loc, resultType, ValueRange{matmulResult, bias}, ValueRange{emptyTensor},
            biasMaps, biasIterators,
            [&](OpBuilder &b, Location loc, ValueRange args) {
                Value add = b.create<arith::AddFOp>(loc, args[0], args[1]);
                b.create<linalg::YieldOp>(loc, add);
            }).getResult(0);

        rewriter.replaceOp(op, biasResult);
    } else {
        rewriter.replaceOp(op, matmulResult);
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertNNToStandardPass
    : public PassWrapper<ConvertNNToStandardPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertNNToStandardPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, 
                    linalg::LinalgDialect,
                    math::MathDialect,
                    tensor::TensorDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());

    // Mark NN dialect as illegal (needs to be lowered)
    target.addIllegalDialect<NNDialect>();

    // Mark standard dialects as legal
    target.addLegalDialect<arith::ArithDialect, 
                          linalg::LinalgDialect,
                          math::MathDialect,
                          tensor::TensorDialect,
                          func::FuncDialect,
                          scf::SCFDialect>();

    // Setup rewrite patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<NNAddOpLowering,
                 NNMulOpLowering,
                 NNMatMulOpLowering,
                 NNReLUOpLowering,
                 NNSoftmaxOpLowering,
                 NNLinearOpLowering>(&getContext());

    // Apply conversion
    if (failed(applyPartialConversion(getOperation(), target, 
                                       std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override { return "convert-nn-to-standard"; }
  StringRef getDescription() const override {
    return "Convert NN dialect to standard MLIR dialects";
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertNNToStandardPass() {
  return std::make_unique<ConvertNNToStandardPass>();
}