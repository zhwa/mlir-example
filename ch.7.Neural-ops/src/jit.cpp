#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include <memory>

using namespace mlir;

LogicalResult lowerToLLVM(ModuleOp module);

class JITCompiler {
public:
    JITCompiler() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
    }

    void* compile(ModuleOp module, const std::string& funcName) {
        // Register translations
        registerBuiltinDialectTranslation(*module.getContext());
        registerLLVMDialectTranslation(*module.getContext());

        // Lower to LLVM dialect
        if (failed(lowerToLLVM(module))) {
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

        // Store the JIT instance to keep it alive
        jitInstances.push_back(std::move(jit));

        return symOrErr->toPtr<void*>();
    }

private:
    std::vector<std::unique_ptr<llvm::orc::LLJIT>> jitInstances;
};

JITCompiler* createJIT() {
    return new JITCompiler();
}

void deleteJIT(JITCompiler* jit) {
    delete jit;
}

void* compileModule(JITCompiler* jit, ModuleOp module, const char* funcName) {
    return jit->compile(module, funcName);
}