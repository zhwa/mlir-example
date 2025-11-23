//===- ir.cpp - MLIR IR Generation for GEMM ------------------------------===//
//
// This file demonstrates how to programmatically generate MLIR IR using the
// C++ API. It creates a high-level linalg.matmul operation for 8×32 × 32×16
// matrix multiplication.
//
// Key Learning Points:
//   - Using OpBuilder to construct MLIR operations
//   - Working with the linalg dialect for linear algebra operations
//   - MemRef vs Tensor: We use memrefs (pointers) instead of tensors (values)
//     to avoid the complexity of bufferization
//   - The linalg.matmul operation is declarative - it says "what" to compute,
//     not "how" (the optimization passes will decide "how")
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

/// Creates a high-level MLIR module containing a GEMM function.
///
/// Operation: C = A @ B where A is 8×32, B is 32×16, C is 8×16 (all float32)
///
/// Generated MLIR (what you'll see with test_ir_generation()):
///   func.func @gemm_8x16x32(%arg0: memref<8x32xf32>,  // Input A
///                           %arg1: memref<32x16xf32>, // Input B
///                           %arg2: memref<8x16xf32>) {// Output C (in/out)
///     %cst = arith.constant 0.0 : f32               // Zero constant
///     linalg.fill ins(%cst) outs(%arg2)             // Initialize C = 0
///     linalg.matmul ins(%arg0, %arg1) outs(%arg2)   // C += A @ B
///     return
///   }
///
/// This is the "high-level" representation. It declares WHAT we want (matrix
/// multiplication) but not HOW (no loops, no SIMD). The lowering passes will
/// transform this into efficient machine code.
OwningOpRef<ModuleOp> createGemmModule(MLIRContext& context) {
  // Step 1: Load required MLIR dialects
  // Dialects are like "vocabulary" - each provides a set of operations
  context.getOrLoadDialect<func::FuncDialect>();    // func.func, func.return
  context.getOrLoadDialect<linalg::LinalgDialect>(); // linalg.matmul, linalg.fill
  context.getOrLoadDialect<memref::MemRefDialect>(); // memref types
  context.getOrLoadDialect<arith::ArithDialect>();   // arith.constant

  // Step 2: Create a builder and module
  // OpBuilder is the tool for constructing MLIR operations
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();  // Source location (we don't track it)
  auto module = ModuleOp::create(loc); // Top-level container

  // Set insertion point to the module body
  builder.setInsertionPointToStart(module.getBody());

  // Define memref types for the GEMM operation
  // A: 8x32, B: 32x16, C: 8x16
  auto f32Type = builder.getF32Type();
  auto matrixA_type = MemRefType::get({8, 32}, f32Type);
  auto matrixB_type = MemRefType::get({32, 16}, f32Type);
  auto matrixC_type = MemRefType::get({8, 16}, f32Type);

  // Create function type: (A, B, C) with no return (memrefs are mutated in-place)
  auto funcType = builder.getFunctionType(
    {matrixA_type, matrixB_type, matrixC_type},   // inputs: A, B, C
    {}                                            // no return - C is modified in-place
  );

  // Create the function
  auto funcOp = builder.create<func::FuncOp>(loc, "gemm_8x16x32", funcType);
  funcOp.setPublic();

  // Create the function body
  auto* entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Get function arguments: A, B, C
  Value argA = entryBlock->getArgument(0);
  Value argB = entryBlock->getArgument(1);
  Value argC = entryBlock->getArgument(2);  // Output memref passed in

  // Fill the output memref C with zeros (required for matmul accumulation)
  auto zeroAttr = builder.getFloatAttr(f32Type, 0.0);
  auto zeroConstant = builder.create<arith::ConstantOp>(loc, zeroAttr);
  builder.create<linalg::FillOp>(
      loc,
      ValueRange{zeroConstant.getResult()},
      ValueRange{argC}  // Fill the memref in-place
  );

  // Create the linalg.matmul operation on memrefs
  // C = matmul(A, B) where C is pre-initialized with zeros
  builder.create<linalg::MatmulOp>(
      loc,
      TypeRange{},               // no result types for memref version
      ValueRange{argA, argB},    // inputs
      ValueRange{argC}           // outputs (operates in-place)
  );

  // Return (no value - memrefs are mutated in-place)
  builder.create<func::ReturnOp>(loc);

  return module;
}

} // namespace mlir