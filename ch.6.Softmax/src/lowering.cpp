//===- lowering.cpp - MLIR Lowering Passes for Softmax ------------------===//
//
// This file demonstrates lowering Math dialect and SCF dialect to LLVM IR.
//
// Pipeline:
//   math.exp → libm calls → scf.for → cf.br → llvm.* instructions
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {

/// Applies the lowering pipeline: Math → Libm → SCF → CF → LLVM
///
/// Steps:
///   1. Canonicalize (simplify IR)
///   2. Math to Libm (math.exp → call to libm's expf)
///   3. SCF to CF (scf.for → cf.br branches)
///   4. Convert to LLVM (memref, arith, func, cf → llvm.*)
///   5. Reconcile casts
///
/// Note: We use math-to-libm instead of math-to-llvm because:
///   - Links to C standard library (libm) for exp, log, sin, etc.
///   - More accurate than polynomial approximations
///   - Simpler for learning purposes
LogicalResult applyLoweringPasses(ModuleOp module) {
  MLIRContext* context = module.getContext();
  PassManager pm(context);

  // Disable multi-threading for deterministic behavior
  context->disableMultithreading();

  // === Phase 1: Canonicalization ===
  pm.addPass(createCanonicalizerPass());

  // === Phase 2: Lower Math operations FIRST ===
  // Convert math.exp to LLVM intrinsics or libm calls
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertMathToLibmPass());

  // === Phase 3: Lower SCF to Control Flow ===
  pm.addPass(createSCFToControlFlowPass());

  // === Phase 4: Convert everything to LLVM dialect ===
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());

  // === Phase 5: Cleanup ===
  pm.addPass(createReconcileUnrealizedCastsPass());

  // Run the pipeline
  if (failed(pm.run(module))) {
    llvm::errs() << "Pass manager failed\n";
    return failure();
  }

  return success();
}

} // namespace mlir
