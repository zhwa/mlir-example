#include "TransformerToStandard.h"
#include "TransformerDialect.h"
#include "TransformerOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

// Lower transformer.matmul to nested loops
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
    (void)cast<MemRefType>(output.getType()); // Unused but validates type

    int rank = lhsType.getRank();

    if (rank == 2) {
      // 2D: (M, K) x (K, N) -> (M, N)
      Value M = createConstantIndex(rewriter, loc, lhsType.getShape()[0]);
      Value N = createConstantIndex(rewriter, loc, rhsType.getShape()[1]);
      Value K = createConstantIndex(rewriter, loc, lhsType.getShape()[1]);
      Value zero = createConstantIndex(rewriter, loc, 0);
      Value one = createConstantIndex(rewriter, loc, 1);

      // for i in [0, M)
      rewriter.create<scf::ForOp>(
          loc, zero, M, one, std::nullopt,
          [&](OpBuilder &builder, Location loc, Value i, ValueRange) {
            // for j in [0, N)
            builder.create<scf::ForOp>(
                loc, zero, N, one, std::nullopt,
                [&](OpBuilder &builder, Location loc, Value j, ValueRange) {
                  Value sum = createConstantFloat(builder, loc, 0.0f);

                  // for k in [0, K)
                  auto forK = builder.create<scf::ForOp>(
                      loc, zero, K, one, ValueRange{sum},
                      [&](OpBuilder &builder, Location loc, Value k,
                          ValueRange iterArgs) {
                        Value accumulator = iterArgs[0];
                        Value lhsVal = builder.create<memref::LoadOp>(
                            loc, lhs, ValueRange{i, k});
                        Value rhsVal = builder.create<memref::LoadOp>(
                            loc, rhs, ValueRange{k, j});
                        Value prod = builder.create<arith::MulFOp>(
                            loc, lhsVal, rhsVal);
                        Value newSum =
                            builder.create<arith::AddFOp>(loc, accumulator, prod);
                        builder.create<scf::YieldOp>(loc, newSum);
                      });

                  builder.create<memref::StoreOp>(loc, forK.getResult(0),
                                                   output, ValueRange{i, j});
                  builder.create<scf::YieldOp>(loc);
                });
            builder.create<scf::YieldOp>(loc);
          });

    } else if (rank == 3) {
      // 3D batched: (B, M, K) x (B, K, N) -> (B, M, N)
      Value B = createConstantIndex(rewriter, loc, lhsType.getShape()[0]);
      Value M = createConstantIndex(rewriter, loc, lhsType.getShape()[1]);
      Value N = createConstantIndex(rewriter, loc, rhsType.getShape()[2]);
      Value K = createConstantIndex(rewriter, loc, lhsType.getShape()[1]);
      Value zero = createConstantIndex(rewriter, loc, 0);
      Value one = createConstantIndex(rewriter, loc, 1);

      // for b in [0, B)
      rewriter.create<scf::ForOp>(
          loc, zero, B, one, std::nullopt,
          [&](OpBuilder &builder, Location loc, Value b, ValueRange) {
            // for i in [0, M)
            builder.create<scf::ForOp>(
                loc, zero, M, one, std::nullopt,
                [&](OpBuilder &builder, Location loc, Value i, ValueRange) {
                  // for j in [0, N)
                  builder.create<scf::ForOp>(
                      loc, zero, N, one, std::nullopt,
                      [&](OpBuilder &builder, Location loc, Value j,
                          ValueRange) {
                        Value sum = createConstantFloat(builder, loc, 0.0f);

                        // for k in [0, K)
                        auto forK = builder.create<scf::ForOp>(
                            loc, zero, K, one, ValueRange{sum},
                            [&](OpBuilder &builder, Location loc, Value k,
                                ValueRange iterArgs) {
                              Value accumulator = iterArgs[0];
                              Value lhsVal = builder.create<memref::LoadOp>(
                                  loc, lhs, ValueRange{b, i, k});
                              Value rhsVal = builder.create<memref::LoadOp>(
                                  loc, rhs, ValueRange{b, k, j});
                              Value prod = builder.create<arith::MulFOp>(
                                  loc, lhsVal, rhsVal);
                              Value newSum = builder.create<arith::AddFOp>(
                                  loc, accumulator, prod);
                              builder.create<scf::YieldOp>(loc, newSum);
                            });

                        builder.create<memref::StoreOp>(
                            loc, forK.getResult(0), output, ValueRange{b, i, j});
                        builder.create<scf::YieldOp>(loc);
                      });
                  builder.create<scf::YieldOp>(loc);
                });
            builder.create<scf::YieldOp>(loc);
          });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// Lower transformer.add to element-wise addition
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

    // Build loop bounds for all dimensions
    SmallVector<Value> bounds;
    Value zero = createConstantIndex(rewriter, loc, 0);
    Value one = createConstantIndex(rewriter, loc, 1);

    for (int64_t dim : outputType.getShape()) {
      bounds.push_back(createConstantIndex(rewriter, loc, dim));
    }

    // Create nested loops
    std::function<void(OpBuilder &, Location, ValueRange, int)> buildLoops;
    buildLoops = [&](OpBuilder &builder, Location loc, ValueRange indices,
                     int depth) {
      if (depth == rank) {
        // Base case: load, add, store
        Value lhsVal = builder.create<memref::LoadOp>(loc, lhs, indices);
        Value rhsVal = builder.create<memref::LoadOp>(loc, rhs, indices);
        Value sum = builder.create<arith::AddFOp>(loc, lhsVal, rhsVal);
        builder.create<memref::StoreOp>(loc, sum, output, indices);
      } else {
        // Recursive case: create loop for this dimension
        builder.create<scf::ForOp>(
            loc, zero, bounds[depth], one, std::nullopt,
            [&](OpBuilder &builder, Location loc, Value iv, ValueRange) {
              SmallVector<Value> newIndices(indices.begin(), indices.end());
              newIndices.push_back(iv);
              buildLoops(builder, loc, newIndices, depth + 1);
              builder.create<scf::YieldOp>(loc);
            });
      }
    };

    buildLoops(rewriter, loc, {}, 0);
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower transformer.mul to element-wise multiplication
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

    // Build loop bounds
    SmallVector<Value> bounds;
    Value zero = createConstantIndex(rewriter, loc, 0);
    Value one = createConstantIndex(rewriter, loc, 1);

    for (int64_t dim : outputType.getShape()) {
      bounds.push_back(createConstantIndex(rewriter, loc, dim));
    }

    // Create nested loops
    std::function<void(OpBuilder &, Location, ValueRange, int)> buildLoops;
    buildLoops = [&](OpBuilder &builder, Location loc, ValueRange indices,
                     int depth) {
      if (depth == rank) {
        Value lhsVal = builder.create<memref::LoadOp>(loc, lhs, indices);
        Value rhsVal = builder.create<memref::LoadOp>(loc, rhs, indices);
        Value prod = builder.create<arith::MulFOp>(loc, lhsVal, rhsVal);
        builder.create<memref::StoreOp>(loc, prod, output, indices);
      } else {
        builder.create<scf::ForOp>(
            loc, zero, bounds[depth], one, std::nullopt,
            [&](OpBuilder &builder, Location loc, Value iv, ValueRange) {
              SmallVector<Value> newIndices(indices.begin(), indices.end());
              newIndices.push_back(iv);
              buildLoops(builder, loc, newIndices, depth + 1);
              builder.create<scf::YieldOp>(loc);
            });
      }
    };

    buildLoops(rewriter, loc, {}, 0);
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower transformer.softmax to numerically stable softmax
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
    int lastDim = shape[rank - 1];

    Value zero = createConstantIndex(rewriter, loc, 0);
    Value one = createConstantIndex(rewriter, loc, 1);
    Value negInf = createConstantFloat(rewriter, loc, -1e9f);

    // Build loops for all dimensions except the last
    SmallVector<Value> bounds;
    for (int i = 0; i < rank - 1; i++) {
      bounds.push_back(createConstantIndex(rewriter, loc, shape[i]));
    }
    Value lastDimBound = createConstantIndex(rewriter, loc, lastDim);

    // Create nested loops for outer dimensions
    std::function<void(OpBuilder &, Location, ValueRange, int)> buildOuterLoops;
    buildOuterLoops = [&](OpBuilder &builder, Location loc, ValueRange indices,
                          int depth) {
      if (depth == rank - 1) {
        // Process the last dimension: find max, compute exp, normalize

        // Step 1: Find max value
        auto maxLoop = builder.create<scf::ForOp>(
            loc, zero, lastDimBound, one, ValueRange{negInf},
            [&](OpBuilder &builder, Location loc, Value iv,
                ValueRange iterArgs) {
              SmallVector<Value> fullIndices(indices.begin(), indices.end());
              fullIndices.push_back(iv);
              Value val =
                  builder.create<memref::LoadOp>(loc, input, fullIndices);
              Value currentMax = iterArgs[0];
              Value newMax =
                  builder.create<arith::MaximumFOp>(loc, currentMax, val);
              builder.create<scf::YieldOp>(loc, newMax);
            });
        Value maxVal = maxLoop.getResult(0);

        // Step 2: Compute exp and sum
        auto sumLoop = builder.create<scf::ForOp>(
            loc, zero, lastDimBound, one,
            ValueRange{createConstantFloat(builder, loc, 0.0f)},
            [&](OpBuilder &builder, Location loc, Value iv,
                ValueRange iterArgs) {
              SmallVector<Value> fullIndices(indices.begin(), indices.end());
              fullIndices.push_back(iv);
              Value val =
                  builder.create<memref::LoadOp>(loc, input, fullIndices);
              Value shifted = builder.create<arith::SubFOp>(loc, val, maxVal);
              Value expVal = builder.create<math::ExpOp>(loc, shifted);
              builder.create<memref::StoreOp>(loc, expVal, output, fullIndices);

              Value currentSum = iterArgs[0];
              Value newSum =
                  builder.create<arith::AddFOp>(loc, currentSum, expVal);
              builder.create<scf::YieldOp>(loc, newSum);
            });
        Value sumVal = sumLoop.getResult(0);

        // Step 3: Normalize by sum
        builder.create<scf::ForOp>(
            loc, zero, lastDimBound, one, std::nullopt,
            [&](OpBuilder &builder, Location loc, Value iv, ValueRange) {
              SmallVector<Value> fullIndices(indices.begin(), indices.end());
              fullIndices.push_back(iv);
              Value expVal =
                  builder.create<memref::LoadOp>(loc, output, fullIndices);
              Value normalized =
                  builder.create<arith::DivFOp>(loc, expVal, sumVal);
              builder.create<memref::StoreOp>(loc, normalized, output,
                                               fullIndices);
              builder.create<scf::YieldOp>(loc);
            });

      } else {
        // Create loop for this outer dimension
        builder.create<scf::ForOp>(
            loc, zero, bounds[depth], one, std::nullopt,
            [&](OpBuilder &builder, Location loc, Value iv, ValueRange) {
              SmallVector<Value> newIndices(indices.begin(), indices.end());
              newIndices.push_back(iv);
              buildOuterLoops(builder, loc, newIndices, depth + 1);
              builder.create<scf::YieldOp>(loc);
            });
      }
    };

    buildOuterLoops(rewriter, loc, {}, 0);
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower transformer.transpose (swaps last two dimensions)
struct TransposeOpLowering : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();

    auto inputType = cast<MemRefType>(input.getType());
    ArrayRef<int64_t> shape = inputType.getShape();
    int rank = shape.size();

    Value zero = createConstantIndex(rewriter, loc, 0);
    Value one = createConstantIndex(rewriter, loc, 1);

    SmallVector<Value> bounds;
    for (int64_t dim : shape) {
      bounds.push_back(createConstantIndex(rewriter, loc, dim));
    }

    // Create nested loops
    std::function<void(OpBuilder &, Location, ValueRange, int)> buildLoops;
    buildLoops = [&](OpBuilder &builder, Location loc, ValueRange indices,
                     int depth) {
      if (depth == rank) {
        // Load with original indices
        Value val = builder.create<memref::LoadOp>(loc, input, indices);

        // Store with swapped last two indices
        SmallVector<Value> swappedIndices(indices.begin(), indices.end());
        std::swap(swappedIndices[rank - 2], swappedIndices[rank - 1]);
        builder.create<memref::StoreOp>(loc, val, output, swappedIndices);
      } else {
        builder.create<scf::ForOp>(
            loc, zero, bounds[depth], one, std::nullopt,
            [&](OpBuilder &builder, Location loc, Value iv, ValueRange) {
              SmallVector<Value> newIndices(indices.begin(), indices.end());
              newIndices.push_back(iv);
              buildLoops(builder, loc, newIndices, depth + 1);
              builder.create<scf::YieldOp>(loc);
            });
      }
    };

    buildLoops(rewriter, loc, {}, 0);
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower transformer.attention (this will be complex, but demonstrates the full pipeline)
struct AttentionOpLowering : public OpRewritePattern<AttentionOp> {
  using OpRewritePattern<AttentionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AttentionOp op,
                                PatternRewriter &rewriter) const override {
    // For now, leave attention as-is or implement as composition of other ops
    // The attention op is complex and would benefit from being implemented
    // as a combination of the primitives we've already defined

    // This is a placeholder - in a real implementation, you'd either:
    // 1. Decompose attention into matmul, softmax, etc. ops
    // 2. Lower directly to loops (very complex)
    // 3. Keep it high-level and implement in C++/Python

    return failure(); // Don't lower attention yet
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
    registry.insert<arith::ArithDialect, func::FuncDialect, math::MathDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulOpLowering, AddOpLowering, MulOpLowering,
                 SoftmaxOpLowering, TransposeOpLowering, AttentionOpLowering>(
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