#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

// Dialect headers
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Math/IR/Math.h>

// Conversion passes
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MathToLibm/MathToLibm.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>

// Linalg passes
#include <mlir/Dialect/Linalg/Passes.h>

#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Support/CommandLine.h>
#include <string>
#include <memory>
#include <vector>

using namespace mlir;

namespace ch8 {

// Initialize LLVM once globally
static struct LLVMInitializer {
    LLVMInitializer() {
        // Reset command line parser to avoid "registered more than once" errors
        llvm::cl::ResetCommandLineParser();

        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
    }
} llvmInitializer;

class Compiler {
public:
    Compiler() {
        // Register dialects
        context_.getOrLoadDialect<func::FuncDialect>();
        context_.getOrLoadDialect<arith::ArithDialect>();
        context_.getOrLoadDialect<memref::MemRefDialect>();
        context_.getOrLoadDialect<linalg::LinalgDialect>();
        context_.getOrLoadDialect<scf::SCFDialect>();
        context_.getOrLoadDialect<math::MathDialect>();
        context_.getOrLoadDialect<LLVM::LLVMDialect>();
    }

    // Parse MLIR text
    OwningOpRef<ModuleOp> parseMLIR(const std::string& mlir_text) {
        return parseSourceString<ModuleOp>(mlir_text, &context_);
    }

    // Apply lowering passes (same as Chapter 7)
    bool lowerToLLVM(ModuleOp module) {
        PassManager pm(&context_);

        // Linalg optimizations
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgGeneralizationPass());

        // Lower Linalg to loops
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());

        // Lower SCF to CF
        // Lower to LLVM (same order as Chapter 7)
        pm.addPass(createConvertMathToLLVMPass());
        pm.addPass(createConvertMathToLibmPass());
        pm.addPass(createConvertSCFToCFPass());
        pm.addPass(createArithToLLVMConversionPass());
        pm.addPass(createConvertControlFlowToLLVMPass());
        pm.addPass(createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(createConvertFuncToLLVMPass());
        pm.addPass(createReconcileUnrealizedCastsPass());
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());

        return succeeded(pm.run(module));
    }
    // JIT compile module using LLJIT (same pattern as Chapter 7)
    void* compileAndGetFunctionPtr(ModuleOp module, const std::string& funcName) {
        // Register translations
        registerBuiltinDialectTranslation(*module.getContext());
        registerLLVMDialectTranslation(*module.getContext());

        // Lower to LLVM dialect
        if (!lowerToLLVM(module)) {
            llvm::errs() << "Failed to lower to LLVM dialect\n";
            return nullptr;
        }

        // Translate to LLVM IR
        llvm::LLVMContext llvmContext;
        auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
        if (!llvmModule) {
            llvm::errs() << "Failed to translate to LLVM IR\n";
            return nullptr;
        }

        // Create LLJIT
        auto jitOrErr = llvm::orc::LLJITBuilder().create();
        if (!jitOrErr) {
            llvm::errs() << "Failed to create LLJIT: " << jitOrErr.takeError() << "\n";
            return nullptr;
        }
        auto jit = std::move(*jitOrErr);

        // Add the module
        auto tsm = llvm::orc::ThreadSafeModule(std::move(llvmModule), 
                                               std::make_unique<llvm::LLVMContext>());
        if (auto err = jit->addIRModule(std::move(tsm))) {
            llvm::errs() << "Failed to add IR module: " << err << "\n";
            return nullptr;
        }

        // Lookup the function
        auto symOrErr = jit->lookup(funcName);
        if (!symOrErr) {
            llvm::errs() << "Failed to lookup function: " << symOrErr.takeError() << "\n";
            return nullptr;
        }

        // Store JIT instance to keep it alive
        jitInstances_.push_back(std::move(jit));

        return symOrErr->toPtr<void*>();
    }

    MLIRContext& getContext() { return context_; }

private:
    MLIRContext context_;
    std::vector<std::unique_ptr<llvm::orc::LLJIT>> jitInstances_;
};

} // namespace ch8