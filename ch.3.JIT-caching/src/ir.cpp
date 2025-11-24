//===- ir.cpp - MLIR IR Generation for GEMM ------------------------------===//
//
// This file demonstrates how to programmatically generate MLIR IR using the
// C++ API. It creates a high-level linalg.matmul operation with DYNAMIC shapes
// that supports matrices of any compatible size.
//
// Key Learning Points:
//   - Using OpBuilder to construct MLIR operations
//   - Working with the linalg dialect for linear algebra operations
//   - Dynamic shapes using ShapedType::kDynamic (memref<?x?xf32>)
//   - MemRef vs Tensor: We use memrefs (pointers) instead of tensors (values)
//     to avoid the complexity of bufferization
//   - The linalg.matmul operation is declarative and shape-polymorphic
//
//===----------------------------------------------------------------------====//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir {

/// Creates a high-level MLIR module containing a GEMM function with DYNAMIC shapes.
///
/// Operation: C = A @ B where A, B, C can be any compatible sizes (all float32)
///
/// Generated MLIR (what you'll see with test_ir_generation()):
///   func.func @gemm(%arg0: memref<?x?xf32>,  // Input A (MxK)
///                   %arg1: memref<?x?xf32>,  // Input B (KxN)
///                   %arg2: memref<?x?xf32>) {// Output C (MxN)
///     %cst = arith.constant 0.0 : f32
///     linalg.fill ins(%cst) outs(%arg2)
///     linalg.matmul ins(%arg0, %arg1) outs(%arg2)
///     return
///   }
///
/// Key features:
///   - Uses memref<?x?xf32> for dynamic dimensions
///   - The ? means "dimension unknown at compile time"
///   - Actual dimensions passed at runtime through memref descriptor
///   - linalg.matmul is shape-polymorphic - works with any size!
OwningOpRef<ModuleOp> createGemmModule(MLIRContext& context) {
  // Load required dialects (same as static version)
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<linalg::LinalgDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();

  // Create builder and module
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());

  // Define DYNAMIC memref types using ShapedType::kDynamic
  // kDynamic is a special constant (-1) meaning "unknown at compile time"
  auto f32Type = builder.getF32Type();
  auto matrixA_type = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic},  // ?x? (any size)
      f32Type
  );
  auto matrixB_type = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic},  // ?x? (any size)
      f32Type
  );
  auto matrixC_type = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic},  // ?x? (any size)
      f32Type
  );

  // Create function type (same signature structure, different types)
  auto funcType = builder.getFunctionType(
    {matrixA_type, matrixB_type, matrixC_type},
    {}
  );

  // Create the function (simpler name - no dimensions in the name)
  auto funcOp = builder.create<func::FuncOp>(loc, "gemm", funcType);
  funcOp.setPublic();

  // Create function body (identical to static version!)
  auto* entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Get function arguments
  Value argA = entryBlock->getArgument(0);
  Value argB = entryBlock->getArgument(1);
  Value argC = entryBlock->getArgument(2);

  // Fill output with zeros
  auto zeroAttr = builder.getFloatAttr(f32Type, 0.0);
  auto zeroConstant = builder.create<arith::ConstantOp>(loc, zeroAttr);
  builder.create<linalg::FillOp>(
      loc,
      ValueRange{zeroConstant.getResult()},
      ValueRange{argC}
  );

  // Create linalg.matmul (works with dynamic shapes - no changes needed!)
  builder.create<linalg::MatmulOp>(
      loc,
      TypeRange{},
      ValueRange{argA, argB},
      ValueRange{argC}
  );

  // Return
  builder.create<func::ReturnOp>(loc);

  return module;
}

} // namespace mlir