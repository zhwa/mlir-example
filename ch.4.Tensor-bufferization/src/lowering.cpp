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
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
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
  
  // Enable IR printing for debugging (optional)
  // pm.enableIRPrinting();

  // Register bufferization interface implementations
  // These tell one-shot-bufferize how to handle tensor ops
  DialectRegistry registry;
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  context->appendDialectRegistry(registry);

  // === Phase 1: Canonicalization ===
  // Simplify the IR (e.g., constant folding, dead code elimination)
  // This is a general cleanup pass that makes subsequent passes easier
  pm.addPass(createCanonicalizerPass());

  // === Phase 2: Bufferization (Tensor → MemRef) ===
  // Transform: tensor<?x?xf32> → memref<?x?xf32>
  // This converts functional tensor semantics to imperative memory operations
  
  // Configure One-Shot Bufferize to handle function boundaries
  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  
  // "One-Shot Bufferize" converts all tensors to memrefs (including function args/results)
  pm.addPass(bufferization::createOneShotBufferizePass(options));
  
  // Convert memref function results to out-parameters
  // This transforms: func(memref, memref) -> memref
  // Into: func(memref, memref, memref) with output as 3rd parameter
  pm.addPass(bufferization::createBufferResultsToOutParamsPass());
  
  // Lower remaining bufferization operations to memref  
  pm.addPass(createBufferizationToMemRefPass());
  pm.addPass(createCanonicalizerPass());  // Clean up

  // === Phase 3: Linalg → Loops ===
  // Transform: linalg.matmul → nested scf.for loops
  // Before: "compute C = A @ B" (declarative)
  // After: "for i=0..8: for j=0..16: for k=0..32: C[i,j] += A[i,k]*B[k,j]"
  pm.addPass(createConvertLinalgToLoopsPass());

  // === Phase 4: SCF → Control Flow ===
  // Transform: scf.for (structured loops) → cf.br (goto-style branches)
  // This converts high-level loops into basic blocks that CPUs understand
  pm.addPass(createConvertSCFToCFPass());

  // === Phase 5: MemRef → LLVM ===
  // Transform: memref<8x32xf32> → raw pointers + strides
  // Memrefs become {ptr, ptr, offset, size0, size1, stride0, stride1}
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());

  // === Phase 6: Everything → LLVM Dialect ===
  // Final conversion of all remaining operations to LLVM equivalents
  pm.addPass(createArithToLLVMConversionPass());     // arith.addf → llvm.fadd
  pm.addPass(createConvertFuncToLLVMPass());         // func.func → llvm.func
  pm.addPass(createConvertControlFlowToLLVMPass());  // cf.br → llvm.br

  // === Phase 7: Cleanup ===
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