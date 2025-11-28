//===- lowering.cpp - MLIR Lowering Passes for SAXPY --------------------===//
//
// This file demonstrates lowering SCF (Structured Control Flow) dialect
// to lower-level representations and eventually to LLVM IR.
//
// Pipeline:
//   scf.for → cf.br (control flow) → llvm.* instructions
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {

/// Applies the lowering pipeline: SCF → CF → LLVM
///
/// Steps:
///   1. Canonicalize (simplify IR)
///   2. SCF to CF (scf.for → cf.br branches)
///   3. Convert to LLVM (memref, arith, func, cf → llvm.*)
///   4. Reconcile casts
LogicalResult applyLoweringPasses(ModuleOp module) {
  MLIRContext* context = module.getContext();
  PassManager pm(context);

  // Disable multi-threading for deterministic behavior
  context->disableMultithreading();

  // === Phase 1: Canonicalization ===
  // Simplify the IR (constant folding, dead code elimination, etc.)
  pm.addPass(createCanonicalizerPass());

  // === Phase 2: SCF to Control Flow ===
  // Lower scf.for to cf.br (branch instructions)
  pm.addPass(createConvertSCFToCFPass());

  // === Phase 3: Convert to LLVM ===
  // Lower all remaining dialects to LLVM
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());

  // === Phase 4: Cleanup ===
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