//===- bindings.cpp - Python bindings for Chapter 9 -----------------------===//
//
// Chapter 9: Custom Dialect with TableGen - Python Interface
//
//===----------------------------------------------------------------------===//
#include "NN/NNDialect.h"
#include "NN/NNOps.h"
#include "NN/NNToStandard.h"

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MathToLibm/MathToLibm.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>

#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/CommandLine.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <ffi.h>

namespace py = pybind11;
using namespace mlir;
using namespace mlir::nn;

//===----------------------------------------------------------------------===//
// LLVM Initialization
//===----------------------------------------------------------------------===//

static struct LLVMInitializer {
    LLVMInitializer() {
        llvm::cl::ResetCommandLineParser();
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
    }
} llvmInitializer;

//===----------------------------------------------------------------------===//
// Compiler with NN Dialect
//===----------------------------------------------------------------------===//

class NNCompiler {
public:
    NNCompiler() {
        // Register all dialects
        context_.getOrLoadDialect<NNDialect>();
        context_.getOrLoadDialect<func::FuncDialect>();
        context_.getOrLoadDialect<arith::ArithDialect>();
        context_.getOrLoadDialect<memref::MemRefDialect>();
        context_.getOrLoadDialect<linalg::LinalgDialect>();
        context_.getOrLoadDialect<scf::SCFDialect>();
        context_.getOrLoadDialect<math::MathDialect>();
        context_.getOrLoadDialect<tensor::TensorDialect>();
        context_.getOrLoadDialect<LLVM::LLVMDialect>();
    }

    OwningOpRef<ModuleOp> parseMLIR(const std::string& mlir_text) {
        return parseSourceString<ModuleOp>(mlir_text, &context_);
    }

    bool lowerToLLVM(ModuleOp module) {
        PassManager pm(&context_);

        // 1. Lower NN dialect to standard dialects (memref-based)
        pm.addPass(createConvertNNToStandardPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // 2. Lower linalg to loops
        pm.addPass(mlir::createConvertLinalgToLoopsPass());

        // 3. Lower to LLVM
        pm.addPass(createConvertMathToLLVMPass());
        pm.addPass(createConvertMathToLibmPass());
        pm.addPass(createConvertSCFToCFPass());
        pm.addPass(createArithToLLVMConversionPass());
        pm.addPass(createConvertControlFlowToLLVMPass());
        pm.addPass(createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(createConvertFuncToLLVMPass());
        pm.addPass(createReconcileUnrealizedCastsPass());

        return succeeded(pm.run(module));
    }

    void* compileAndGetFunctionPtr(ModuleOp module, const std::string& funcName) {
        registerBuiltinDialectTranslation(*module.getContext());
        registerLLVMDialectTranslation(*module.getContext());

        if (!lowerToLLVM(module)) {
            llvm::errs() << "Failed to lower to LLVM dialect\n";
            return nullptr;
        }

        mlir::ExecutionEngineOptions options;
        options.transformer = mlir::makeOptimizingTransformer(3, 0, nullptr);

        auto maybeEngine = mlir::ExecutionEngine::create(module, options);
        if (!maybeEngine) {
            llvm::errs() << "Failed to create ExecutionEngine: " 
                         << maybeEngine.takeError() << "\n";
            return nullptr;
        }

        auto engine = std::move(*maybeEngine);
        auto expectedFPtr = engine->lookup(funcName);
        if (!expectedFPtr) {
            llvm::errs() << "Failed to lookup function: " << funcName << "\n";
            return nullptr;
        }

        // Use raw pointer to avoid destruction order issues
        engines_.push_back(engine.release());
        return reinterpret_cast<void*>(*expectedFPtr);
    }

private:
    MLIRContext context_;
    std::vector<mlir::ExecutionEngine*> engines_;  // Raw pointers
};

static NNCompiler& getCompiler() {
    static NNCompiler compiler;
    return compiler;
}

//===----------------------------------------------------------------------===//
// Memref Marshalling (from Chapter 8)
//===----------------------------------------------------------------------===//

void marshal_memref_1d(std::vector<void*>& args, py::array_t<float> arr) {
    auto buf = arr.request();
    float* data = static_cast<float*>(buf.ptr);
    args.push_back(data);
    args.push_back(data);
    args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(0)));
    args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[0])));
    args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(1)));
}

void marshal_memref_2d(std::vector<void*>& args, py::array_t<float> arr) {
    auto buf = arr.request();
    float* data = static_cast<float*>(buf.ptr);
    args.push_back(data);
    args.push_back(data);
    args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(0)));
    args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[0])));
    args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[1])));
    args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[1])));
    args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(1)));
}

//===----------------------------------------------------------------------===//
// Universal Execute with libffi
//===----------------------------------------------------------------------===//

py::array_t<float> execute(const std::string& mlir_text,
                            const std::string& func_name,
                            py::list inputs,
                            py::tuple output_shape) {
    NNCompiler& compiler = getCompiler();

    auto module = compiler.parseMLIR(mlir_text);
    if (!module) {
        throw std::runtime_error("Failed to parse MLIR");
    }

    void* fnPtr = compiler.compileAndGetFunctionPtr(module.get(), func_name);
    if (!fnPtr) {
        throw std::runtime_error("Failed to compile function");
    }

    std::vector<ssize_t> out_shape;
    for (auto item : output_shape) {
        out_shape.push_back(py::cast<ssize_t>(item));
    }
    auto output = py::array_t<float>(out_shape);

    std::vector<void*> args;

    // Marshal input arguments
    for (auto item : inputs) {
        auto arr = py::cast<py::array_t<float>>(item);
        auto buf = arr.request();
        if (buf.ndim == 1) {
            marshal_memref_1d(args, arr);
        } else if (buf.ndim == 2) {
            marshal_memref_2d(args, arr);
        } else {
            throw std::runtime_error("Only 1D and 2D arrays supported");
        }
    }

    // Marshal output argument (return value becomes last parameter)
    if (out_shape.size() == 1) {
        marshal_memref_1d(args, output);
    } else if (out_shape.size() == 2) {
        marshal_memref_2d(args, output);
    } else {
        throw std::runtime_error("Only 1D and 2D outputs supported");
    }

    size_t num_args = args.size();
    std::vector<ffi_type*> arg_types(num_args, &ffi_type_pointer);
    std::vector<void*> arg_values(num_args);
    for (size_t i = 0; i < num_args; ++i) {
        arg_values[i] = &args[i];
    }

    ffi_cif cif;
    if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, num_args, 
                     &ffi_type_void, arg_types.data()) != FFI_OK) {
        throw std::runtime_error("libffi ffi_prep_cif failed");
    }

    ffi_call(&cif, FFI_FN(fnPtr), nullptr, arg_values.data());
    return output;
}

//===----------------------------------------------------------------------===//
// Python Module
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(ch9, m) {
    m.doc() = "Chapter 9: Custom Dialect with TableGen";

    m.def("execute", &execute,
          "Compile and execute MLIR with NN dialect",
          py::arg("mlir_text"),
          py::arg("func_name"),
          py::arg("inputs"),
          py::arg("output_shape"));
}