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
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    SmallVector<utils::IteratorType> iteratorTypes(outputType.getRank(), utils::IteratorType::parallel);

    rewriter.create<linalg::GenericOp>(
        loc, 
        ValueRange{op.getLhs(), op.getRhs()},  // inputs
        ValueRange{op.getOutput()},             // outputs
        ArrayRef<AffineMap>{
            rewriter.getMultiDimIdentityMap(outputType.getRank()),
            rewriter.getMultiDimIdentityMap(outputType.getRank()),
            rewriter.getMultiDimIdentityMap(outputType.getRank())
        },
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.eraseOp(op);
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
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    SmallVector<utils::IteratorType> iteratorTypes(outputType.getRank(), utils::IteratorType::parallel);

    rewriter.create<linalg::GenericOp>(
        loc,
        ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{op.getOutput()},
        ArrayRef<AffineMap>{
            rewriter.getMultiDimIdentityMap(outputType.getRank()),
            rewriter.getMultiDimIdentityMap(outputType.getRank()),
            rewriter.getMultiDimIdentityMap(outputType.getRank())
        },
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value product = b.create<arith::MulFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, product);
        });

    rewriter.eraseOp(op);
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
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Create zero constant for initialization
    auto zeroAttr = rewriter.getFloatAttr(outputType.getElementType(), 0.0);
    auto zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);

    // Fill output with zeros
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{op.getOutput()});

    // Create linalg.matmul
    rewriter.create<linalg::MatmulOp>(
        loc, ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{op.getOutput()});

    rewriter.eraseOp(op);
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
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Create zero constant
    auto zeroAttr = rewriter.getFloatAttr(outputType.getElementType(), 0.0);
    auto zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);

    SmallVector<utils::IteratorType> iteratorTypes(
        outputType.getRank(),
        utils::IteratorType::parallel);

    rewriter.create<linalg::GenericOp>(
        loc,
        ValueRange{op.getInput()},
        ValueRange{op.getOutput()},
        ArrayRef<AffineMap>{
            rewriter.getMultiDimIdentityMap(outputType.getRank()),
            rewriter.getMultiDimIdentityMap(outputType.getRank())
        },
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // max(0, x)
          Value maxVal = b.create<arith::MaximumFOp>(loc, zero, args[0]);
          b.create<linalg::YieldOp>(loc, maxVal);
        });

    rewriter.eraseOp(op);
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