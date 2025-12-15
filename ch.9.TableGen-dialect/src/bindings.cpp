//===- bindings.cpp - Python bindings for Chapter 9 -----------------------===//
//
// Chapter 9: Custom Dialect with TableGen - Python Interface
//
//===----------------------------------------------------------------------===//
#include "NNDialect.h"
#include "NNOps.h"
#include "NNToStandard.h"

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

#include <sstream>
#include <memory>
#include <unordered_map>
#include <functional>

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

    MLIRContext& getContext() { return context_; }

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
        engines_.emplace_back(engine.release());
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
    args.emplace_back(data);
    args.emplace_back(data);
    args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(0)));
    args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[0])));
    args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(1)));
}

void marshal_memref_2d(std::vector<void*>& args, py::array_t<float> arr) {
    auto buf = arr.request();
    float* data = static_cast<float*>(buf.ptr);
    args.emplace_back(data);
    args.emplace_back(data);
    args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(0)));
    args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[0])));
    args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[1])));
    args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[1])));
    args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(1)));
}

//===----------------------------------------------------------------------===//
// Pythonic Tensor API (Industrial Approach)
//===----------------------------------------------------------------------===//
// High-level API with operator overloading and automatic graph building.
// Uses OpBuilder to construct IR directly (same as Torch-MLIR, JAX, IREE).

class Tensor {
public:
    Tensor(py::array_t<float> data) 
        : data_(data), op_type_("input"), is_computed_(true) {
        auto buf = data.request();
        for (ssize_t i = 0; i < buf.ndim; ++i) {
            shape_.emplace_back(buf.shape[i]);
        }
    }

    Tensor(std::vector<ssize_t> shape, std::string op_type,
           std::shared_ptr<Tensor> input1 = nullptr,
           std::shared_ptr<Tensor> input2 = nullptr)
        : shape_(shape), op_type_(op_type), is_computed_(false),
          input1_(input1), input2_(input2) {}

    py::array_t<float> numpy() {
        if (!is_computed_) {
            throw std::runtime_error("Tensor not computed yet. Call forward() first.");
        }
        return data_;
    }

    const std::vector<ssize_t>& shape() const { return shape_; }
    const std::string& op_type() const { return op_type_; }
    bool is_input() const { return op_type_ == "input"; }
    std::shared_ptr<Tensor> input1() const { return input1_; }
    std::shared_ptr<Tensor> input2() const { return input2_; }

    void set_data(py::array_t<float> data) {
        data_ = data;
        is_computed_ = true;
    }

private:
    py::array_t<float> data_;
    std::vector<ssize_t> shape_;
    std::string op_type_;
    bool is_computed_;
    std::shared_ptr<Tensor> input1_;
    std::shared_ptr<Tensor> input2_;
};

// Graph compiler - Industrial approach with OpBuilder
class GraphCompiler {
public:
    static py::array_t<float> forward(std::shared_ptr<Tensor> output) {
        NNCompiler& compiler = getCompiler();

        // Topological sort to get computation order
        std::vector<std::shared_ptr<Tensor>> inputs;
        std::vector<std::shared_ptr<Tensor>> ops;
        std::unordered_map<Tensor*, int> tensor_ids;

        std::function<void(std::shared_ptr<Tensor>)> collect;
        collect = [&](std::shared_ptr<Tensor> t) {
            if (tensor_ids.find(t.get()) != tensor_ids.end()) return;

            if (t->is_input()) {
                tensor_ids[t.get()] = inputs.size();
                inputs.emplace_back(t);
            } else {
                if (t->input1()) collect(t->input1());
                if (t->input2()) collect(t->input2());
                tensor_ids[t.get()] = inputs.size() + ops.size();
                ops.emplace_back(t);
            }
        };
        collect(output);

        // Build IR directly using OpBuilder (industrial approach)
        auto module = buildModule(compiler.getContext(), inputs, ops, output);
        if (!module) {
            throw std::runtime_error("Failed to build MLIR module");
        }

        // Compile and execute
        void* fnPtr = compiler.compileAndGetFunctionPtr(module.get(), "compute");
        if (!fnPtr) {
            throw std::runtime_error("Failed to compile function");
        }

        // Collect input data
        py::list input_arrays;
        for (auto& inp : inputs) {
            input_arrays.append(inp->numpy());
        }

        // Execute
        py::tuple out_shape;
        std::vector<ssize_t> shape_vec = output->shape();
        if (shape_vec.size() == 1) {
            out_shape = py::make_tuple(shape_vec[0]);
        } else if (shape_vec.size() == 2) {
            out_shape = py::make_tuple(shape_vec[0], shape_vec[1]);
        }

        auto result = execute_direct(fnPtr, input_arrays, out_shape);
        output->set_data(result);
        return result;
    }

private:
    // Build MLIR module using OpBuilder (no strings!)
    static OwningOpRef<ModuleOp> buildModule(
        MLIRContext& context,
        const std::vector<std::shared_ptr<Tensor>>& inputs,
        const std::vector<std::shared_ptr<Tensor>>& ops,
        std::shared_ptr<Tensor> output) {

        OpBuilder builder(&context);
        auto loc = builder.getUnknownLoc();

        // Create module
        auto module = ModuleOp::create(loc);
        builder.setInsertionPointToEnd(module.getBody());

        // Build function signature types
        SmallVector<Type> inputTypes;
        for (auto& inp : inputs) {
            inputTypes.emplace_back(getMemRefType(builder, inp->shape()));
        }
        // Add output parameter
        inputTypes.emplace_back(getMemRefType(builder, output->shape()));

        auto funcType = builder.getFunctionType(inputTypes, {});

        // Create function
        auto func = builder.create<func::FuncOp>(loc, "compute", funcType);
        Block* entryBlock = func.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);

        // Map tensors to SSA values
        std::unordered_map<Tensor*, Value> valueMap;
        for (size_t i = 0; i < inputs.size(); ++i) {
            valueMap[inputs[i].get()] = entryBlock->getArgument(i);
        }
        Value outputVal = entryBlock->getArgument(inputs.size());

        // Build operations
        for (size_t i = 0; i < ops.size(); ++i) {
            auto& op = ops[i];
            Value result;

            // Allocate result buffer (unless it's the final output)
            if (op.get() == output.get()) {
                result = outputVal;
            } else {
                auto allocOp = builder.create<memref::AllocOp>(
                    loc, 
                    mlir::cast<MemRefType>(getMemRefType(builder, op->shape()))
                );
                result = allocOp.getResult();
            }

            valueMap[op.get()] = result;

            // Create the operation
            Value input1 = valueMap[op->input1().get()];
            std::string op_name = op->op_type();

            if (op_name == "add") {
                Value input2 = valueMap[op->input2().get()];
                builder.create<AddOp>(loc, input1, input2, result);
            } else if (op_name == "mul") {
                Value input2 = valueMap[op->input2().get()];
                builder.create<MulOp>(loc, input1, input2, result);
            } else if (op_name == "matmul") {
                Value input2 = valueMap[op->input2().get()];
                builder.create<MatMulOp>(loc, input1, input2, result);
            } else if (op_name == "relu") {
                builder.create<ReLUOp>(loc, input1, result);
            } else {
                throw std::runtime_error("Unknown operation: " + op_name);
            }
        }

        // Return from function
        builder.create<func::ReturnOp>(loc);

        return OwningOpRef<ModuleOp>(module);
    }

    // Helper: Create MemRefType from shape
    static Type getMemRefType(OpBuilder& builder, const std::vector<ssize_t>& shape) {
        SmallVector<int64_t> mlirShape(shape.begin(), shape.end());
        return MemRefType::get(mlirShape, builder.getF32Type());
    }

    // Execute without parsing MLIR text
    static py::array_t<float> execute_direct(void* fnPtr, py::list inputs, py::tuple output_shape)
    {
        std::vector<ssize_t> out_shape;
        for (auto item : output_shape) {
            out_shape.emplace_back(py::cast<ssize_t>(item));
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

        // Marshal output argument
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
        if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, num_args, &ffi_type_void, arg_types.data()) != FFI_OK) {
            throw std::runtime_error("libffi ffi_prep_cif failed");
        }

        ffi_call(&cif, FFI_FN(fnPtr), nullptr, arg_values.data());
        return output;
    }
};

// Operator functions
std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape() != b->shape()) {
        throw std::runtime_error("Shape mismatch for add");
    }
    return std::make_shared<Tensor>(a->shape(), "add", a, b);
}

std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape() != b->shape()) {
        throw std::runtime_error("Shape mismatch for mul");
    }
    return std::make_shared<Tensor>(a->shape(), "mul", a, b);
}

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape().size() != 2 || b->shape().size() != 2) {
        throw std::runtime_error("matmul requires 2D tensors");
    }
    if (a->shape()[1] != b->shape()[0]) {
        throw std::runtime_error("matmul shape mismatch");
    }
    std::vector<ssize_t> out_shape = {a->shape()[0], b->shape()[1]};
    return std::make_shared<Tensor>(out_shape, "matmul", a, b);
}

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a) {
    return std::make_shared<Tensor>(a->shape(), "relu", a, nullptr);
}

//===----------------------------------------------------------------------===//
// Python Module
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(ch9, m) {
    m.doc() = "Chapter 9: Custom Dialect with TableGen\n\n"
              "Industrial-strength MLIR dialect with Pythonic API.\n"
              "Uses OpBuilder for IR construction (same as Torch-MLIR/JAX).";

    // Pythonic Tensor API
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<py::array_t<float>>(), "Create tensor from numpy array")
        .def("numpy", &Tensor::numpy, "Convert to numpy array")
        .def("__add__", &add, "Element-wise addition")
        .def("__mul__", &mul, "Element-wise multiplication")
        .def("__repr__", [](const Tensor& t) {
            std::stringstream ss;
            ss << "Tensor(shape=[";
            for (size_t i = 0; i < t.shape().size(); ++i) {
                if (i > 0) ss << ", ";
                ss << t.shape()[i];
            }
            ss << "], op=" << t.op_type() << ")";
            return ss.str();
        });

    m.def("forward", &GraphCompiler::forward, "Forward pass: compile and execute tensor graph");
    m.def("matmul", &matmul, "Matrix multiplication");
    m.def("relu", &relu, "ReLU activation");
}