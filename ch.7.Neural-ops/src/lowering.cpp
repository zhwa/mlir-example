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
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

LogicalResult lowerToLLVM(ModuleOp module) {
    MLIRContext* context = module.getContext();
    PassManager pm(context);

    // Disable multi-threading for deterministic behavior
    context->disableMultithreading();

    // Register bufferization interface implementations
    DialectRegistry registry;
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    context->appendDialectRegistry(registry);

    // === Phase 1: Canonicalization ===
    pm.addPass(createCanonicalizerPass());

    // === Phase 2: Bufferization ===
    bufferization::OneShotBufferizePassOptions bufferizationOptions;
    bufferizationOptions.bufferizeFunctionBoundaries = true;
    
    pm.addPass(bufferization::createOneShotBufferizePass(bufferizationOptions));
    pm.addPass(bufferization::createBufferResultsToOutParamsPass());
    pm.addPass(createConvertBufferizationToMemRefPass());

    // === Phase 3: Lower Linalg to Loops ===
    pm.addPass(createConvertLinalgToLoopsPass());

    // === Phase 4: Lower Math operations ===
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