//===- ir.cpp - MLIR IR Generation for Softmax --------------------------===//
//
// This file demonstrates Softmax using the Math dialect for mathematical
// functions and SCF dialect for explicit looping.
//
// Operation: output[i] = exp(input[i] - max) / sum(exp(input[j] - max))
//
// Key Learning Points:
//   - Using math.exp for exponential function
//   - Multi-pass algorithm with loop-carried variables
//   - Numerical stability (subtract max before exp)
//   - Reduction patterns (finding max, computing sum)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include <limits>

namespace mlir {

/// Creates a Softmax module using Math dialect and three-pass algorithm.
///
/// Algorithm (numerically stable):
///   Pass 1: Find maximum value
///   Pass 2: Compute exp(x - max) and accumulate sum
///   Pass 3: Normalize by dividing by sum
///
/// Generated MLIR:
///   func.func @softmax(%input: memref<?xf32>, %output: memref<?xf32>) {
///     // Pass 1: Find max
///     %max = scf.for %i = ... iter_args(%current_max) {
///       %val = memref.load %input[%i]
///       %new_max = arith.maximumf %current_max, %val
///       scf.yield %new_max
///     }
///
///     // Pass 2: Compute exp and sum
///     %sum = scf.for %i = ... iter_args(%current_sum) {
///       %val = memref.load %input[%i]
///       %shifted = arith.subf %val, %max
///       %exp_val = math.exp %shifted
///       memref.store %exp_val, %temp[%i]
///       %new_sum = arith.addf %current_sum, %exp_val
///       scf.yield %new_sum
///     }
///
///     // Pass 3: Normalize
///     scf.for %i = ... {
///       %exp_val = memref.load %temp[%i]
///       %normalized = arith.divf %exp_val, %sum
///       memref.store %normalized, %output[%i]
///     }
///   }
OwningOpRef<ModuleOp> createSoftmaxModule(MLIRContext& context) {
  // Load required dialects
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();
  context.getOrLoadDialect<math::MathDialect>();

  // Create builder and module
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module.getBody());

  // Define types
  auto f32Type = builder.getF32Type();

  // Dynamic 1D memref: memref<?xf32>
  auto dynamicMemRefType = MemRefType::get({ShapedType::kDynamic}, f32Type);

  // Function type: (memref<?xf32>, memref<?xf32>) -> ()
  auto funcType = builder.getFunctionType(
    {dynamicMemRefType, dynamicMemRefType},
    {}
  );

  // Create function
  auto funcOp = builder.create<func::FuncOp>(loc, "softmax", funcType);
  funcOp.setPublic();

  // Create function body
  auto& entryBlock = *funcOp.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // Get function arguments
  Value input = entryBlock.getArgument(0);
  Value output = entryBlock.getArgument(1);

  // Get size of input
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value size = builder.create<memref::DimOp>(loc, input, c0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  // Allocate temporary buffer for exp values
  Value tempBuffer = builder.create<memref::AllocaOp>(loc, dynamicMemRefType, ValueRange{size});

  //===--------------------------------------------------------------------===//
  // Pass 1: Find maximum value for numerical stability
  //===--------------------------------------------------------------------===//

  // Initialize with negative infinity
  Value negInf = builder.create<arith::ConstantFloatOp>(
      loc, APFloat::getInf(f32Type.getFloatSemantics(), /*Negative=*/true), f32Type);

  // Create loop to find max
  auto findMaxLoop = builder.create<scf::ForOp>(
      loc, c0, size, c1, ValueRange{negInf},
      [&](OpBuilder& b, Location loc, Value i, ValueRange iterArgs) {
        Value currentMax = iterArgs[0];
        Value val = b.create<memref::LoadOp>(loc, input, ValueRange{i});
        Value newMax = b.create<arith::MaximumFOp>(loc, currentMax, val);
        b.create<scf::YieldOp>(loc, ValueRange{newMax});
      }
  );
  Value maxVal = findMaxLoop.getResult(0);

  //===--------------------------------------------------------------------===//
  // Pass 2: Compute exp(x - max) and accumulate sum
  //===--------------------------------------------------------------------===//

  Value zeroFloat = builder.create<arith::ConstantFloatOp>(
      loc, APFloat(0.0f), f32Type);

  auto expSumLoop = builder.create<scf::ForOp>(
      loc, c0, size, c1, ValueRange{zeroFloat},
      [&](OpBuilder& b, Location loc, Value i, ValueRange iterArgs) {
        Value currentSum = iterArgs[0];

        // Load input value
        Value val = b.create<memref::LoadOp>(loc, input, ValueRange{i});

        // Subtract max for numerical stability
        Value shifted = b.create<arith::SubFOp>(loc, val, maxVal);

        // Compute exp(x - max)
        Value expVal = b.create<math::ExpOp>(loc, shifted);

        // Store exp value to temporary buffer
        b.create<memref::StoreOp>(loc, expVal, tempBuffer, ValueRange{i});

        // Accumulate sum
        Value newSum = b.create<arith::AddFOp>(loc, currentSum, expVal);

        b.create<scf::YieldOp>(loc, ValueRange{newSum});
      }
  );
  Value sumExp = expSumLoop.getResult(0);

  //===--------------------------------------------------------------------===//
  // Pass 3: Normalize by dividing by sum
  //===--------------------------------------------------------------------===//

  builder.create<scf::ForOp>(
      loc, c0, size, c1, std::nullopt,
      [&](OpBuilder& b, Location loc, Value i, ValueRange iterArgs) {
        // Load exp value from temp buffer
        Value expVal = b.create<memref::LoadOp>(loc, tempBuffer, ValueRange{i});

        // Normalize: divide by sum
        Value normalized = b.create<arith::DivFOp>(loc, expVal, sumExp);

        // Store to output
        b.create<memref::StoreOp>(loc, normalized, output, ValueRange{i});

        b.create<scf::YieldOp>(loc, ValueRange{});
      }
  );

  // Return
  builder.create<func::ReturnOp>(loc);

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