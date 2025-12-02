#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <ffi.h>
#include "graph.h"

namespace py = pybind11;
using namespace mlir;

class Graph {
public:
    Graph() : ctx(std::make_unique<MLIRContext>()), 
              graph(std::make_unique<ComputationGraph>(ctx.get())) {}

    int variable(const std::vector<int64_t>& shape) {
        return graph->addVariable(shape);
    }

    int add(int lhs, int rhs) {
        return graph->add(lhs, rhs);
    }

    int mul(int lhs, int rhs) {
        return graph->mul(lhs, rhs);
    }

    int matmul(int lhs, int rhs) {
        return graph->matmul(lhs, rhs);
    }

    int relu(int input) {
        return graph->relu(input);
    }

    int softmax(int input) {
        return graph->softmax(input);
    }

    std::string getMLIR(int outputId, const std::string& funcName = "main") {
        auto module = graph->generateMLIR(outputId, funcName);
        std::string result;
        llvm::raw_string_ostream os(result);
        module.print(os);
        return result;
    }

    py::object compile(int outputId, const std::string& funcName = "main") {
        auto module = graph->generateMLIR(outputId, funcName);

        if (!jit) {
            jit = std::make_unique<JITCompiler>();
        }

        void* fnPtr = jit->compile(module, funcName);
        if (!fnPtr) {
            throw std::runtime_error("Failed to compile function");
        }

        return py::cast(reinterpret_cast<uintptr_t>(fnPtr));
    }

private:
    std::unique_ptr<MLIRContext> ctx;
    std::unique_ptr<ComputationGraph> graph;
    std::unique_ptr<JITCompiler> jit;
};

// Helper to marshal memref arguments
void marshal_memref_1d(std::vector<void*>& args, py::array_t<float> arr) {
    auto buf = arr.request();
    float* data = static_cast<float*>(buf.ptr);

    args.push_back(data);  // allocated_ptr
    args.push_back(data);  // aligned_ptr
    args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(0)));  // offset
    args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[0])));  // size
    args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(1)));  // stride
}

void marshal_memref_2d(std::vector<void*>& args, py::array_t<float> arr) {
    auto buf = arr.request();
    float* data = static_cast<float*>(buf.ptr);

    args.push_back(data);  // allocated_ptr
    args.push_back(data);  // aligned_ptr
    args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(0)));  // offset
    args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[0])));  // size0
    args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[1])));  // size1
    args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[1])));  // stride0
    args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(1)));  // stride1
}

// libffi-based execute - handles ANY signature universally with LLJIT
py::array_t<float> execute_generic(uintptr_t fnPtr, py::list inputs, py::tuple output_shape) {
    // Prepare output array
    std::vector<ssize_t> out_shape;
    for (auto item : output_shape) {
        out_shape.push_back(py::cast<ssize_t>(item));
    }
    auto output = py::array_t<float>(out_shape);

    // Marshal all arguments (inputs + output) into a flat vector
    std::vector<void*> args;
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

    // Marshal output
    if (out_shape.size() == 1) {
        marshal_memref_1d(args, output);
    } else if (out_shape.size() == 2) {
        marshal_memref_2d(args, output);
    } else {
        throw std::runtime_error("Only 1D and 2D outputs supported");
    }

    // Use libffi for truly variadic calling (handles ANY parameter count)
    size_t num_args = args.size();

    // Setup FFI types (all arguments are pointers or intptr_t cast to pointers)
    std::vector<ffi_type*> arg_types(num_args, &ffi_type_pointer);

    // Prepare argument values
    std::vector<void*> arg_values(num_args);
    for (size_t i = 0; i < num_args; ++i) {
        arg_values[i] = &args[i];  // pointer to the void* value
    }

    // Setup FFI CIF (Call Interface)
    ffi_cif cif;
    ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, num_args, 
                                      &ffi_type_void, arg_types.data());

    if (status != FFI_OK) {
        throw std::runtime_error("libffi ffi_prep_cif failed");
    }

    // Execute the function using libffi
    ffi_call(&cif, FFI_FN(reinterpret_cast<void(*)()>(fnPtr)), nullptr, arg_values.data());

    return output;
}

PYBIND11_MODULE(ch7_neural_ops, m) {
    m.doc() = "MLIR Neural Operations with LLJIT + libffi";

    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("variable", &Graph::variable, "Add a variable/tensor to the graph")
        .def("add", &Graph::add, "Add element-wise addition")
        .def("mul", &Graph::mul, "Add element-wise multiplication")
        .def("matmul", &Graph::matmul, "Add matrix multiplication")
        .def("relu", &Graph::relu, "Add ReLU activation")
        .def("softmax", &Graph::softmax, "Add Softmax activation")
        .def("get_mlir", &Graph::getMLIR, "Get MLIR representation")
        .def("compile", &Graph::compile, "Compile the graph");

    m.def("execute_generic", &execute_generic,
          "libffi-based execute with LLJIT - handles ANY signature",
          py::arg("fnPtr"),
          py::arg("inputs"),
          py::arg("output_shape"));
}