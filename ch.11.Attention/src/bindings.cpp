//===- bindings.cpp - Python bindings for Chapter 11 ----------------------===//
//===----------------------------------------------------------------------===//
#include "TransformerDialect.h"
#include "TransformerOps.h"
#include "TransformerToStandard.h"

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>
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

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MathToLibm/MathToLibm.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>

#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/CommandLine.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <ffi.h>

#include <sstream>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cmath>

namespace py = pybind11;
using namespace mlir;
using namespace mlir::transformer;

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
// Compiler
//===----------------------------------------------------------------------===//

class TransformerCompiler {
public:
    TransformerCompiler() {
        context_.getOrLoadDialect<TransformerDialect>();
        context_.getOrLoadDialect<func::FuncDialect>();
        context_.getOrLoadDialect<arith::ArithDialect>();
        context_.getOrLoadDialect<memref::MemRefDialect>();
        context_.getOrLoadDialect<scf::SCFDialect>();
        context_.getOrLoadDialect<math::MathDialect>();
        context_.getOrLoadDialect<LLVM::LLVMDialect>();
    }

    MLIRContext& getContext() { return context_; }

    bool lowerToLLVM(ModuleOp module) {
        PassManager pm(&context_);

        // Lower transformer dialect to standard dialects
        pm.addPass(createLowerTransformerToStandardPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // Lower to LLVM
        pm.addPass(mlir::createConvertSCFToCFPass());
        pm.addPass(mlir::createConvertMathToLibmPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertMathToLLVMPass());
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());

        if (failed(pm.run(module))) {
            llvm::errs() << "Failed to run passes\n";
            return false;
        }
        return true;
    }

    std::unique_ptr<ExecutionEngine> createJIT(ModuleOp module) {
        llvm::SmallVector<llvm::StringRef, 4> sharedLibs;
        mlir::ExecutionEngineOptions opts;
        opts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;
        opts.sharedLibPaths = llvm::ArrayRef(sharedLibs);
        auto maybeEngine = mlir::ExecutionEngine::create(module, opts);

        if (!maybeEngine) {
            llvm::errs() << "Failed to create execution engine\n";
            return nullptr;
        }

        return std::move(*maybeEngine);
    }

private:
    MLIRContext context_;
};

//===----------------------------------------------------------------------===//
// C++ Attention Implementation (fallback)
//===----------------------------------------------------------------------===//

py::array_t<float> compute_attention_cpp(
    py::array_t<float> input_arr,
    py::array_t<float> wq_arr,
    py::array_t<float> wk_arr,
    py::array_t<float> wv_arr,
    py::array_t<float> wo_arr,
    int num_heads,
    int head_dim
) {
    auto input = input_arr.unchecked<2>();

    int seq_len = input.shape(0);
    int d_model = input.shape(1);

    // Allocate output array
    py::array_t<float> output_arr({seq_len, d_model});
    auto output = output_arr.mutable_unchecked<2>();

    // Get weight matrices
    auto wq = wq_arr.unchecked<2>();
    auto wk = wk_arr.unchecked<2>();
    auto wv = wv_arr.unchecked<2>();
    auto wo = wo_arr.unchecked<2>();

    // Allocate temporary buffers
    std::vector<float> Q(seq_len * d_model);
    std::vector<float> K(seq_len * d_model);
    std::vector<float> V(seq_len * d_model);
    std::vector<float> scores(seq_len * seq_len);
    std::vector<float> attn_weights(seq_len * seq_len);
    std::vector<float> context(seq_len * d_model);

    // Compute Q = input @ W_q
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d_model; k++) {
                sum += input(i, k) * wq(k, j);
            }
            Q[i * d_model + j] = sum;
        }
    }

    // Compute K = input @ W_k
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d_model; k++) {
                sum += input(i, k) * wk(k, j);
            }
            K[i * d_model + j] = sum;
        }
    }

    // Compute V = input @ W_v
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d_model; k++) {
                sum += input(i, k) * wv(k, j);
            }
            V[i * d_model + j] = sum;
        }
    }

    // Process each head
    for (int h = 0; h < num_heads; h++) {
        int head_offset = h * head_dim;

        // Compute attention scores for this head
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float score = 0.0f;
                for (int k = 0; k < head_dim; k++) {
                    score += Q[i * d_model + head_offset + k] * 
                             K[j * d_model + head_offset + k];
                }
                score /= std::sqrt(static_cast<float>(head_dim));
                scores[i * seq_len + j] = score;
            }
        }

        // Apply softmax
        for (int i = 0; i < seq_len; i++) {
            // Find max for numerical stability
            float max_score = scores[i * seq_len];
            for (int j = 1; j < seq_len; j++) {
                max_score = std::max(max_score, scores[i * seq_len + j]);
            }

            // Compute exp and sum
            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                float exp_val = std::exp(scores[i * seq_len + j] - max_score);
                attn_weights[i * seq_len + j] = exp_val;
                sum_exp += exp_val;
            }

            // Normalize
            for (int j = 0; j < seq_len; j++) {
                attn_weights[i * seq_len + j] /= sum_exp;
            }
        }

        // Compute weighted sum of values
        for (int i = 0; i < seq_len; i++) {
            for (int k = 0; k < head_dim; k++) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    sum += attn_weights[i * seq_len + j] * 
                           V[j * d_model + head_offset + k];
                }
                context[i * d_model + head_offset + k] = sum;
            }
        }
    }

    // Apply output projection: output = context @ W_o
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d_model; k++) {
                sum += context[i * d_model + k] * wo(k, j);
            }
            output(i, j) = sum;
        }
    }

    return output_arr;
}

//===----------------------------------------------------------------------===//
// Python Bindings
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(ch11, m) {
    m.doc() = "Chapter 11: Attention Mechanism with Transformer Dialect";

    py::class_<TransformerCompiler>(m, "Compiler")
        .def(py::init<>())
        .def("get_context", &TransformerCompiler::getContext,
             py::return_value_policy::reference);

    m.def("attention", &compute_attention_cpp,
          py::arg("input"),
          py::arg("wq"),
          py::arg("wk"),
          py::arg("wv"),
          py::arg("wo"),
          py::arg("num_heads"),
          py::arg("head_dim"),
          "Compute multi-head attention with linear projections (C++ implementation)");
}