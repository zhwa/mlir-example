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

    int input(const std::vector<int64_t>& shape) {
        return graph->addInput(shape);
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

// Helper to execute 1D functions
py::array_t<float> execute_1d(uintptr_t fnPtr, py::array_t<float> input) {
    auto inputBuf = input.request();
    if (inputBuf.ndim != 1) {
        throw std::runtime_error("Input must be 1D");
    }

    py::array_t<float> output(inputBuf.shape[0]);
    auto outputBuf = output.request();

    // For static 1D memrefs like memref<Nxf32>, MLIR uses 5 parameters:
    // (allocated, aligned, offset, size, stride)
    using FnType = void(*)(
        float*, float*, int64_t, int64_t, int64_t,  // input
        float*, float*, int64_t, int64_t, int64_t   // output
    );
    auto fn = reinterpret_cast<FnType>(fnPtr);

    fn(static_cast<float*>(inputBuf.ptr),
       static_cast<float*>(inputBuf.ptr),
       0, inputBuf.shape[0], 1,
       static_cast<float*>(outputBuf.ptr),
       static_cast<float*>(outputBuf.ptr),
       0, outputBuf.shape[0], 1);

    return output;
}

// Helper to execute 2D functions
py::array_t<float> execute_2d(uintptr_t fnPtr, py::array_t<float> input) {
    auto inputBuf = input.request();
    if (inputBuf.ndim != 2) {
        throw std::runtime_error("Input must be 2D");
    }

    py::array_t<float> output({inputBuf.shape[0], inputBuf.shape[1]});
    auto outputBuf = output.request();

    // For static 2D memrefs like memref<MxNxf32>, MLIR uses 7 parameters:
    // (allocated, aligned, offset, size0, size1, stride1, stride0)
    using FnType = void(*)(
        float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // input
        float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t   // output
    );
    auto fn = reinterpret_cast<FnType>(fnPtr);

    fn(static_cast<float*>(inputBuf.ptr),
       static_cast<float*>(inputBuf.ptr),
       0, inputBuf.shape[0], inputBuf.shape[1], inputBuf.shape[1], 1,
       static_cast<float*>(outputBuf.ptr),
       static_cast<float*>(outputBuf.ptr),
       0, outputBuf.shape[0], outputBuf.shape[1], outputBuf.shape[1], 1);

    return output;
}

// Helper for element-wise binary operations
py::array_t<float> execute_binary_1d(uintptr_t fnPtr, py::array_t<float> lhs, py::array_t<float> rhs) {
    auto lhsBuf = lhs.request();
    auto rhsBuf = rhs.request();

    if (lhsBuf.ndim != 1 || rhsBuf.ndim != 1) {
        throw std::runtime_error("Inputs must be 1D");
    }

    py::array_t<float> output(lhsBuf.shape[0]);
    auto outputBuf = output.request();

    // For binary operations: memref<Nxf32>, memref<Nxf32> -> memref<Nxf32>
    using FnType = void(*)(
        float*, float*, int64_t, int64_t, int64_t,  // lhs
        float*, float*, int64_t, int64_t, int64_t,  // rhs
        float*, float*, int64_t, int64_t, int64_t   // output
    );
    auto fn = reinterpret_cast<FnType>(fnPtr);

    fn(static_cast<float*>(lhsBuf.ptr),
       static_cast<float*>(lhsBuf.ptr),
       0, lhsBuf.shape[0], 1,
       static_cast<float*>(rhsBuf.ptr),
       static_cast<float*>(rhsBuf.ptr),
       0, rhsBuf.shape[0], 1,
       static_cast<float*>(outputBuf.ptr),
       static_cast<float*>(outputBuf.ptr),
       0, outputBuf.shape[0], 1);

    return output;
}

// Helper for matmul
py::array_t<float> execute_matmul(uintptr_t fnPtr, py::array_t<float> lhs, py::array_t<float> rhs) {
    auto lhsBuf = lhs.request();
    auto rhsBuf = rhs.request();

    if (lhsBuf.ndim != 2 || rhsBuf.ndim != 2) {
        throw std::runtime_error("Inputs must be 2D");
    }

    py::array_t<float> output({lhsBuf.shape[0], rhsBuf.shape[1]});
    auto outputBuf = output.request();

    // For matmul: memref<MxKxf32>, memref<KxNxf32> -> memref<MxNxf32>
    using FnType = void(*)(
        float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // lhs
        float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // rhs
        float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t   // output
    );
    auto fn = reinterpret_cast<FnType>(fnPtr);

    fn(static_cast<float*>(lhsBuf.ptr),
       static_cast<float*>(lhsBuf.ptr),
       0, lhsBuf.shape[0], lhsBuf.shape[1], lhsBuf.shape[1], 1,
       static_cast<float*>(rhsBuf.ptr),
       static_cast<float*>(rhsBuf.ptr),
       0, rhsBuf.shape[0], rhsBuf.shape[1], rhsBuf.shape[1], 1,
       static_cast<float*>(outputBuf.ptr),
       static_cast<float*>(outputBuf.ptr),
       0, outputBuf.shape[0], outputBuf.shape[1], outputBuf.shape[1], 1);

    return output;
}

PYBIND11_MODULE(ch7_neural_ops, m) {
    m.doc() = "MLIR Neural Operations with Computation Graph";

    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("input", &Graph::input, "Add an input placeholder")
        .def("add", &Graph::add, "Add element-wise addition")
        .def("mul", &Graph::mul, "Add element-wise multiplication")
        .def("matmul", &Graph::matmul, "Add matrix multiplication")
        .def("relu", &Graph::relu, "Add ReLU activation")
        .def("softmax", &Graph::softmax, "Add Softmax activation")
        .def("get_mlir", &Graph::getMLIR, "Get MLIR representation")
        .def("compile", &Graph::compile, "Compile the graph");

    m.def("execute_1d", &execute_1d, "Execute 1D function");
    m.def("execute_2d", &execute_2d, "Execute 2D function");
    m.def("execute_binary_1d", &execute_binary_1d, "Execute binary 1D function");
    m.def("execute_matmul", &execute_matmul, "Execute matmul function");
}