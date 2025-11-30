#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

LogicalResult lowerToLLVM(ModuleOp module) {
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
    pm.addPass(createConvertSCFToCFPass());

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