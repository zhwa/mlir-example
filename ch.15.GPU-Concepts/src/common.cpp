#include "common.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

// Dialects
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"

// Conversions
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

// LLVM IR Translation
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace ch15 {

mlir::MLIRContext* createContext() {
    auto context = new mlir::MLIRContext();

    // Register all required dialects for CPU execution
    // Note: We don't use GPU dialect - we directly build SCF loops for CPU
    context->getOrLoadDialect<mlir::arith::ArithDialect>();
    context->getOrLoadDialect<mlir::func::FuncDialect>();
    context->getOrLoadDialect<mlir::memref::MemRefDialect>();
    context->getOrLoadDialect<mlir::scf::SCFDialect>();
    context->getOrLoadDialect<mlir::math::MathDialect>();
    context->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();

    return context;
}

mlir::LogicalResult lowerToLLVMDialect(mlir::ModuleOp module) {
    mlir::PassManager pm(module.getContext());

    // Enable IR printing for debugging (optional)
    // pm.enableIRPrinting();

    // Lowering pipeline for CPU execution
    // 1. SCF → ControlFlow (convert loops to branches)
    pm.addPass(mlir::createConvertSCFToCFPass());

    // 2. Math → Libm (convert math ops to libm calls BEFORE lowering to LLVM)
    pm.addPass(mlir::createConvertMathToLibmPass());

    // 3. Convert all high-level dialects to LLVM
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());

    // 4. Cleanup unrealized casts
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    // Run the pipeline
    if (mlir::failed(pm.run(module))) {
        llvm::errs() << "Lowering to LLVM dialect failed\n";
        return mlir::failure();
    }

    return mlir::success();
}

std::unique_ptr<llvm::Module> translateToLLVMIR(
    mlir::ModuleOp module,
    llvm::LLVMContext& llvmContext
) {
    // Register LLVM IR translation for all dialects
    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());

    // The Math dialect ops are lowered to LLVM dialect in the lowering pass,
    // so we don't need separate translation registration

    // Translate MLIR to LLVM IR
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);

    if (!llvmModule) {
        llvm::errs() << "Failed to translate to LLVM IR\n";
        return nullptr;
    }

    return llvmModule;
}

} // namespace ch15