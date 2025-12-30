//===- lowering.cpp - MLIR Lowering Passes for SAXPY --------------------===//
//
// This file demonstrates the modern tensor-first lowering pipeline with
// bufferization.
//
// Pipeline:
//   tensor ops → bufferization → memref ops → scf.for → cf.br → llvm.*
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {

/// Applies the modern tensor-first lowering pipeline with bufferization
///
/// Steps:
///   1. Canonicalize (simplify high-level tensor IR)
///   2. One-Shot Bufferize (tensor → memref transformation)
///   3. Bufferization-To-MemRef (finalize bufferization dialect)
///   4. Linalg to Loops (linalg.generic → scf.for)
///   5. SCF to CF (scf.for → cf.br branches)
///   6. Convert to LLVM (memref, arith, func, cf → llvm.*)
///   7. Reconcile casts
LogicalResult applyLoweringPasses(ModuleOp module) {
  MLIRContext* context = module.getContext();
  PassManager pm(context);

  // Disable multi-threading for deterministic behavior
  context->disableMultithreading();

  // Register bufferization interface implementations
  // These tell One-Shot Bufferize how to handle tensor operations
  DialectRegistry registry;
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  context->appendDialectRegistry(registry);

  // === Phase 1: Canonicalization ===
  // Simplify the tensor IR (constant folding, dead code elimination, etc.)
  pm.addPass(createCanonicalizerPass());

  // === Phase 2: Bufferization ===
  // Convert tensors to memrefs using One-Shot Bufferize
  bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizationOptions));
  
  // Convert function results to out-parameters
  pm.addPass(bufferization::createBufferResultsToOutParamsPass());
  
  // Finalize bufferization dialect operations to memref operations
  pm.addPass(createConvertBufferizationToMemRefPass());
  pm.addPass(createCanonicalizerPass());

  // === Phase 3: Lower Linalg to Loops ===
  // Convert linalg.generic to explicit scf.for loops
  pm.addPass(createConvertLinalgToLoopsPass());

  // === Phase 4: SCF to Control Flow ===
  // Lower scf.for to cf.br (branch instructions)
  pm.addPass(createSCFToControlFlowPass());

  // === Phase 5: Convert to LLVM ===
  // Lower all remaining dialects to LLVM
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());

  // === Phase 6: Cleanup ===
  // Remove any unrealized casts introduced during conversion
  pm.addPass(createReconcileUnrealizedCastsPass());

  // Run the pipeline
  if (failed(pm.run(module))) {
    llvm::errs() << "Pass manager failed\n";
    return failure();
  }

  return success();
}

} // namespace mlir