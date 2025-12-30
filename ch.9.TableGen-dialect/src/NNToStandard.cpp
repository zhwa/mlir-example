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
                 NNReLUOpLowering>(&getContext());

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