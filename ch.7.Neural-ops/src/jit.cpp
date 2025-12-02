#include "graph.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"
#include <memory>
#include <vector>

using namespace mlir;

LogicalResult lowerToLLVM(ModuleOp module);

// Implementation details hidden from header
struct JITCompiler::Impl {
    std::vector<std::unique_ptr<mlir::ExecutionEngine>> engines;
};

// JITCompiler implementation
JITCompiler::JITCompiler() : pImpl(new Impl()) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
}

JITCompiler::~JITCompiler() {
    delete pImpl;
}

void* JITCompiler::compile(ModuleOp module, const std::string& funcName) {
    // Register translations (required for ExecutionEngine)
    registerBuiltinDialectTranslation(*module.getContext());
    registerLLVMDialectTranslation(*module.getContext());

    // Lower to LLVM dialect
    if (failed(lowerToLLVM(module))) {
        llvm::errs() << "Failed to lower to LLVM dialect\n";
        return nullptr;
    }

    // Create ExecutionEngine with optimization pipeline
    mlir::ExecutionEngineOptions options;
    options.transformer = mlir::makeOptimizingTransformer(
        /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

    auto maybeEngine = mlir::ExecutionEngine::create(module, options);
    if (!maybeEngine) {
        llvm::errs() << "Failed to create ExecutionEngine: " 
                     << maybeEngine.takeError() << "\n";
        return nullptr;
    }

    auto engine = std::move(*maybeEngine);

    // Lookup function
    auto expectedFPtr = engine->lookup(funcName);
    if (!expectedFPtr) {
        llvm::errs() << "Failed to lookup function: " << funcName << "\n";
        return nullptr;
    }

    // Store ExecutionEngine instance to keep it alive
    void* fnPtr = reinterpret_cast<void*>(*expectedFPtr);
    pImpl->engines.push_back(std::move(engine));

    return fnPtr;
}