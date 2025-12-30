//===- ir.cpp - MLIR IR Generation for SAXPY -----------------------------===//
//
// This file demonstrates SAXPY (Single-Precision A·X Plus Y) using tensors
// and the Linalg dialect for high-level operations.
//
// Operation: C[i] = α · A[i] + B[i]
//
// Key Learning Points:
//   - Tensor-first architecture (modern MLIR best practice)
//   - Using linalg.generic with tensor types
//   - Functional semantics (immutable tensors)
//   - Bufferization happens later in the pipeline
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir {

/// Creates a SAXPY module using tensors and Linalg dialect.
///
/// Operation: C[i] = alpha * A[i] + B[i]
///
/// Generated MLIR (Tensor-based):
///   func.func @saxpy(%alpha: f32,
///                    %A: tensor<?xf32>,
///                    %B: tensor<?xf32>) -> tensor<?xf32> {
///     %c0 = arith.constant 0 : index
///     %size = tensor.dim %A, %c0 : tensor<?xf32>
///     %empty = tensor.empty(%size) : tensor<?xf32>
///
///     %result = linalg.generic {
///       indexing_maps = [affine_map<(d0) -> (d0)>,
///                        affine_map<(d0) -> (d0)>,
///                        affine_map<(d0) -> (d0)>],
///       iterator_types = ["parallel"]
///     } ins(%A, %B : tensor<?xf32>, tensor<?xf32>)
///       outs(%empty : tensor<?xf32>) {
///     ^bb0(%a: f32, %b: f32, %out: f32):
///       %scaled = arith.mulf %alpha, %a : f32
///       %sum = arith.addf %scaled, %b : f32
///       linalg.yield %sum : f32
///     } -> tensor<?xf32>
///
///     return %result : tensor<?xf32>
///   }
OwningOpRef<ModuleOp> createSaxpyModule(MLIRContext& context) {
  // Load required dialects
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<linalg::LinalgDialect>();

  // Create builder and module
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module.getBody());

  // Define types
  auto f32Type = builder.getF32Type();

  // Dynamic 1D tensor: tensor<?xf32>
  auto dynamicTensorType = RankedTensorType::get({ShapedType::kDynamic}, f32Type);

  // Function type: (f32, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  auto funcType = builder.getFunctionType(
    {f32Type, dynamicTensorType, dynamicTensorType},
    {dynamicTensorType}
  );

  // Create function
  auto funcOp = builder.create<func::FuncOp>(loc, "saxpy", funcType);
  funcOp.setPublic();

  // Create function body
  auto& entryBlock = *funcOp.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // Get function arguments
  Value alpha = entryBlock.getArgument(0);  // f32
  Value A = entryBlock.getArgument(1);      // tensor<?xf32>
  Value B = entryBlock.getArgument(2);      // tensor<?xf32>

  // Get dynamic size: %size = tensor.dim %A, %c0
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value size = builder.create<tensor::DimOp>(loc, A, c0);

  // Create empty output tensor: %empty = tensor.empty(%size)
  SmallVector<OpFoldResult> sizes;
  sizes.push_back(size);
  Value empty = builder.create<tensor::EmptyOp>(
    loc, 
    sizes,         // Dynamic dimensions as OpFoldResult
    f32Type        // Element type
  );

  // Create affine maps for linalg.generic
  // All three operands (A, B, C) use identity map: (d0) -> (d0)
  auto identityMap = AffineMap::get(
    /*dimCount=*/1,
    /*symbolCount=*/0,
    builder.getAffineDimExpr(0),
    &context
  );

  SmallVector<AffineMap> indexingMaps{identityMap, identityMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes{utils::IteratorType::parallel};

  // Create linalg.generic operation
  auto genericOp = builder.create<linalg::GenericOp>(
    loc,
    /*resultTensorTypes=*/TypeRange{dynamicTensorType},
    /*inputs=*/ValueRange{A, B},
    /*outputs=*/ValueRange{empty},
    indexingMaps,
    iteratorTypes,
    /*bodyBuilder=*/[&](OpBuilder& b, Location nestedLoc, ValueRange args) {
      // args[0] = a (from A), args[1] = b (from B), args[2] = out (unused)
      Value a = args[0];
      Value bVal = args[1];

      // Compute: scaled = alpha * a
      Value scaled = b.create<arith::MulFOp>(nestedLoc, alpha, a);

      // Compute: result = scaled + bVal
      Value sum = b.create<arith::AddFOp>(nestedLoc, scaled, bVal);

      // Yield the result
      b.create<linalg::YieldOp>(nestedLoc, sum);
    }
  );

  // Return the result tensor
  builder.create<func::ReturnOp>(loc, genericOp.getResult(0));

  return module;
}

} // namespace mlir