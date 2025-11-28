//===- ir.cpp - MLIR IR Generation for SAXPY -----------------------------===//
//
// This file demonstrates SAXPY (Single-Precision A·X Plus Y) using the SCF
// (Structured Control Flow) dialect for explicit looping.
//
// Operation: C[i] = α · A[i] + B[i]
//
// Key Learning Points:
//   - Using scf.for for explicit loops (vs linalg.generic)
//   - Working with dynamic shapes (memref<?xf32>)
//   - memref.dim to query runtime dimensions
//   - Loop induction variables and SSA form
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir {

/// Creates a SAXPY module using SCF dialect for explicit looping.
///
/// Operation: C[i] = alpha * A[i] + B[i]
///
/// Generated MLIR:
///   func.func @saxpy(%alpha: f32,
///                    %A: memref<?xf32>,
///                    %B: memref<?xf32>,
///                    %C: memref<?xf32>) {
///     %c0 = arith.constant 0 : index
///     %size = memref.dim %A, %c0 : memref<?xf32>
///     %c1 = arith.constant 1 : index
///
///     scf.for %i = %c0 to %size step %c1 {
///       %a = memref.load %A[%i] : memref<?xf32>
///       %b = memref.load %B[%i] : memref<?xf32>
///       %scaled = arith.mulf %alpha, %a : f32
///       %result = arith.addf %scaled, %b : f32
///       memref.store %result, %C[%i] : memref<?xf32>
///     }
///     return
///   }
OwningOpRef<ModuleOp> createSaxpyModule(MLIRContext& context) {
  // Load required dialects
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();

  // Create builder and module
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module.getBody());

  // Define types
  auto f32Type = builder.getF32Type();

  // Dynamic 1D memref: memref<?xf32>
  auto dynamicMemRefType = MemRefType::get({ShapedType::kDynamic}, f32Type);

  // Function type: (f32, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  auto funcType = builder.getFunctionType(
    {f32Type, dynamicMemRefType, dynamicMemRefType, dynamicMemRefType},
    {}
  );

  // Create function
  auto funcOp = builder.create<func::FuncOp>(loc, "saxpy", funcType);
  funcOp.setPublic();

  // Create function body
  auto& entryBlock = *funcOp.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // Get function arguments
  Value alpha = entryBlock.getArgument(0);  // f32
  Value A = entryBlock.getArgument(1);      // memref<?xf32>
  Value B = entryBlock.getArgument(2);      // memref<?xf32>
  Value C = entryBlock.getArgument(3);      // memref<?xf32>

  // Create constants
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  // Get dynamic size: %size = memref.dim %A, %c0
  Value size = builder.create<memref::DimOp>(loc, A, c0);

  // Create scf.for loop: for i = 0 to size step 1
  auto forOp = builder.create<scf::ForOp>(loc, c0, size, c1);

  // Build loop body
  builder.setInsertionPointToStart(forOp.getBody());
  Value i = forOp.getInductionVar();

  // Load A[i] and B[i]
  Value a = builder.create<memref::LoadOp>(loc, A, ValueRange{i});
  Value b = builder.create<memref::LoadOp>(loc, B, ValueRange{i});

  // Compute: scaled = alpha * a
  Value scaled = builder.create<arith::MulFOp>(loc, alpha, a);

  // Compute: result = scaled + b
  Value result = builder.create<arith::AddFOp>(loc, scaled, b);

  // Store result to C[i]
  builder.create<memref::StoreOp>(loc, result, C, ValueRange{i});

  // Return to function level
  builder.setInsertionPointAfter(forOp);
  builder.create<func::ReturnOp>(loc);

  return module;
}

} // namespace mlir