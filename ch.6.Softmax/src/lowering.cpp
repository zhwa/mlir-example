//===- lowering.cpp - MLIR Lowering Passes for Softmax ------------------===//
//
// This file demonstrates lowering tensor-first Softmax through bufferization
// to LLVM IR.
//
// Pipeline:
//   Tensor IR → Bufferize → Linalg-to-Loops → Math-to-Libm → SCF-to-CF → LLVM
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {

/// Applies the lowering pipeline for tensor-first Softmax.
///
/// Steps:
///   1. Canonicalize (simplify IR)
///   2. One-Shot Bufferize (tensor → memref)
///   3. Buffer-Results-To-Out-Params (convert return values to out parameters)
///   4. Convert-Bufferization-To-MemRef (lower bufferization.* ops)
///   5. Linalg-To-Loops (linalg.generic → scf.for)
///   6. Math-To-Libm (math.exp → call to libm's expf)
///   7. SCF-To-CF (scf.for → cf.br branches)
///   8. Convert to LLVM (memref, arith, func, cf → llvm.*)
///   9. Reconcile casts
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
  pm.addPass(createCanonicalizerPass());

  // === Phase 2: Bufferization (Tensor → MemRef) ===
  // Configure One-Shot Bufferize options
  bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizationOptions));
  pm.addPass(bufferization::createBufferResultsToOutParamsPass());
  pm.addPass(createConvertBufferizationToMemRefPass());

  // === Phase 3: Lower Linalg to Loops ===
  pm.addPass(createConvertLinalgToLoopsPass());

  // === Phase 4: Lower Math operations ===
  // Convert math.exp to LLVM intrinsics or libm calls
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertMathToLibmPass());

  // === Phase 5: Lower SCF to Control Flow ===
  pm.addPass(createSCFToControlFlowPass());

  // === Phase 6: Convert everything to LLVM dialect ===
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());

  // === Phase 7: Cleanup ===
  pm.addPass(createReconcileUnrealizedCastsPass());

  // Run the pipeline
  if (failed(pm.run(module))) {
    llvm::errs() << "Pass manager failed\n";
    return failure();
  }

  return success();
}

} // namespace mlir
