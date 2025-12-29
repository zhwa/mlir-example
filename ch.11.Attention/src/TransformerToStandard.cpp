#include "TransformerToStandard.h"
#include "TransformerDialect.h"
#include "TransformerOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::transformer {

//===----------------------------------------------------------------------===//
// Lowering Patterns
//===----------------------------------------------------------------------===//

// Helper: Create constants
static Value createConstantFloat(OpBuilder &builder, Location loc,
                                  float value) {
  return builder.create<arith::ConstantOp>(
      loc, builder.getF32Type(), builder.getF32FloatAttr(value));
}

static Value createConstantIndex(OpBuilder &builder, Location loc,
                                  int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

// Lower transformer.matmul to linalg.matmul
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

// Lower transformer.add to linalg.generic for element-wise addition
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

// Lower transformer.mul to linalg.generic for element-wise multiplication
struct MulOpLowering : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
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

    // All dimensions are parallel
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
          Value prod = b.create<arith::MulFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, prod);
        }
    );

    rewriter.eraseOp(op);
    return success();
  }
};

// Lower transformer.softmax using linalg operations for numerically stable softmax
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

// Lower transformer.transpose to linalg.transpose (swaps last two dimensions)
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
// Pass Definition
//===----------------------------------------------------------------------===//

struct LowerTransformerToStandardPass
    : public PassWrapper<LowerTransformerToStandardPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTransformerToStandardPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, math::MathDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulOpLowering, AddOpLowering, MulOpLowering,
                 SoftmaxOpLowering, TransposeOpLowering>(
        &getContext());

    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createLowerTransformerToStandardPass() {
  return std::make_unique<LowerTransformerToStandardPass>();
}

} // namespace mlir::transformer