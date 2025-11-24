//===- lowering.cpp - MLIR Optimization and Lowering Passes --------------===//
//
// This file demonstrates MLIR's progressive lowering approach. We start with
// high-level declarative operations (linalg.matmul) and gradually transform
// them into low-level LLVM IR through a series of passes.
//
// Think of it as a pipeline:
//   linalg.matmul → scf.for loops → cf.br branches → llvm.* instructions
//
// Each pass handles one level of abstraction, making the compiler modular
// and easier to understand/debug.
//
//===----------------------------------------------------------------------====//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {

/// Applies the full optimization and lowering pipeline to the module.
///
/// This is where the magic happens! We transform:
///   High-level: "compute matrix multiplication" (linalg.matmul)
///   ↓
///   Mid-level: "for i, j, k: C[i,j] += A[i,k] * B[k,j]" (scf.for)
///   ↓
///   Low-level: "load, multiply, store" (llvm.load, llvm.fmul, llvm.store)
///
/// Each transformation is a separate "pass" that runs in sequence.
/// Think of it as a compiler assembly line.
LogicalResult applyOptimizationPasses(ModuleOp module) {
  MLIRContext* context = module.getContext();
  PassManager pm(context);

  // Disable multi-threading
  context->disableMultithreading();

  // === Phase 1: Canonicalization ===
  // Simplify the IR (e.g., constant folding, dead code elimination)
  // This is a general cleanup pass that makes subsequent passes easier
  pm.addPass(createCanonicalizerPass());

  // === Phase 2: Linalg → Loops ===
  // Transform: linalg.matmul → nested scf.for loops
  // Before: "compute C = A @ B" (declarative)
  // After: "for i=0..8: for j=0..16: for k=0..32: C[i,j] += A[i,k]*B[k,j]"
  pm.addPass(createConvertLinalgToLoopsPass());

  // === Phase 3: SCF → Control Flow ===
  // Transform: scf.for (structured loops) → cf.br (goto-style branches)
  // This converts high-level loops into basic blocks that CPUs understand
  pm.addPass(createConvertSCFToCFPass());

  // === Phase 4: MemRef → LLVM ===
  // Transform: memref<8x32xf32> → raw pointers + strides
  // Memrefs become {ptr, ptr, offset, size0, size1, stride0, stride1}
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());

  // === Phase 5: Everything → LLVM Dialect ===
  // Final conversion of all remaining operations to LLVM equivalents
  pm.addPass(createArithToLLVMConversionPass());     // arith.addf → llvm.fadd
  pm.addPass(createConvertFuncToLLVMPass());         // func.func → llvm.func
  pm.addPass(createConvertControlFlowToLLVMPass());  // cf.br → llvm.br

  // === Phase 6: Cleanup ===
  // Remove any temporary cast operations left over from conversions
  pm.addPass(createReconcileUnrealizedCastsPass());

  // Run the pass pipeline
  if (failed(pm.run(module))) {
    llvm::errs() << "Pass pipeline failed\n";
    return failure();
  }

  return success();
}

} // namespace mlir