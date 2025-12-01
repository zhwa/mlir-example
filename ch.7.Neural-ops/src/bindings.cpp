#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
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

// Generic execute helper - handles arbitrary inputs
py::array_t<float> execute_generic(uintptr_t fnPtr, py::list inputs, py::tuple output_shape) {
    // Prepare output array
    std::vector<ssize_t> out_shape;
    for (auto item : output_shape) {
        out_shape.push_back(py::cast<ssize_t>(item));
    }
    auto output = py::array_t<float>(out_shape);

    // Marshal all arguments
    std::vector<void*> args;
    
    // Helper lambdas for marshaling
    auto marshal_1d = [&args](py::array_t<float> arr) {
        auto buf = arr.request();
        float* data = static_cast<float*>(buf.ptr);
        args.push_back(data);
        args.push_back(data);
        args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(0)));
        args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[0])));
        args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(1)));
    };
    
    auto marshal_2d = [&args](py::array_t<float> arr) {
        auto buf = arr.request();
        float* data = static_cast<float*>(buf.ptr);
        args.push_back(data);
        args.push_back(data);
        args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(0)));
        args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[0])));
        args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[1])));
        args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[1])));
        args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(1)));
    };

    // Marshal inputs
    for (auto item : inputs) {
        auto arr = py::cast<py::array_t<float>>(item);
        auto buf = arr.request();
        if (buf.ndim == 1) {
            marshal_1d(arr);
        } else if (buf.ndim == 2) {
            marshal_2d(arr);
        } else {
            throw std::runtime_error("Only 1D and 2D arrays supported");
        }
    }

    // Marshal output
    if (out_shape.size() == 1) {
        marshal_1d(output);
    } else if (out_shape.size() == 2) {
        marshal_2d(output);
    } else {
        throw std::runtime_error("Only 1D and 2D outputs supported");
    }

    // Execute with appropriate calling convention based on arg count
    size_t num_args = args.size();
    
    if (num_args == 10) {  // 1 input + 1 output (both 1D): 5+5
        using FnPtr = void(*)(void*, void*, void*, void*, void*,
                               void*, void*, void*, void*, void*);
        auto fn = reinterpret_cast<FnPtr>(fnPtr);
        fn(args[0], args[1], args[2], args[3], args[4],
           args[5], args[6], args[7], args[8], args[9]);
    } else if (num_args == 14) {  // 1 input + 1 output (both 2D): 7+7
        using FnPtr = void(*)(void*, void*, void*, void*, void*, void*, void*,
                               void*, void*, void*, void*, void*, void*, void*);
        auto fn = reinterpret_cast<FnPtr>(fnPtr);
        fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
           args[7], args[8], args[9], args[10], args[11], args[12], args[13]);
    } else if (num_args == 15) {  // 2 inputs + 1 output (all 1D): 5+5+5
        using FnPtr = void(*)(void*, void*, void*, void*, void*,
                               void*, void*, void*, void*, void*,
                               void*, void*, void*, void*, void*);
        auto fn = reinterpret_cast<FnPtr>(fnPtr);
        fn(args[0], args[1], args[2], args[3], args[4],
           args[5], args[6], args[7], args[8], args[9],
           args[10], args[11], args[12], args[13], args[14]);
    } else if (num_args == 21) {  // 2 inputs + 1 output (all 2D): 7+7+7
        using FnPtr = void(*)(void*, void*, void*, void*, void*, void*, void*,
                               void*, void*, void*, void*, void*, void*, void*,
                               void*, void*, void*, void*, void*, void*, void*);
        auto fn = reinterpret_cast<FnPtr>(fnPtr);
        fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
           args[7], args[8], args[9], args[10], args[11], args[12], args[13],
           args[14], args[15], args[16], args[17], args[18], args[19], args[20]);
    } else if (num_args == 28) {  // 3 inputs + 1 output (all 2D): 7+7+7+7
        using FnPtr = void(*)(void*, void*, void*, void*, void*, void*, void*,
                               void*, void*, void*, void*, void*, void*, void*,
                               void*, void*, void*, void*, void*, void*, void*,
                               void*, void*, void*, void*, void*, void*, void*);
        auto fn = reinterpret_cast<FnPtr>(fnPtr);
        fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
           args[7], args[8], args[9], args[10], args[11], args[12], args[13],
           args[14], args[15], args[16], args[17], args[18], args[19], args[20],
           args[21], args[22], args[23], args[24], args[25], args[26], args[27]);
    } else {
        throw std::runtime_error("Unsupported argument count: " + std::to_string(num_args) + 
                                  ". Extend execute_generic() for this case.");
    }

    return output;
}

PYBIND11_MODULE(ch7_neural_ops, m) {
    m.doc() = "MLIR Neural Operations with Computation Graph";

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
          "Generic execute function - handles arbitrary inputs",
          py::arg("fnPtr"),
          py::arg("inputs"),
          py::arg("output_shape"));
}