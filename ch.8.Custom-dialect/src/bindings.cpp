#include "compiler.cpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ch8 {

// Singleton compiler to avoid CommandLine option registration conflicts
static Compiler& getCompiler() {
    static Compiler compiler;
    return compiler;
}

// Helper to marshal memref arguments (from Chapter 7)
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

// Generic execute function - handles arbitrary inputs
py::array_t<float> execute(const std::string& mlir_text,
                            const std::string& func_name,
                            py::list inputs,
                            py::tuple output_shape) {
    Compiler& compiler = getCompiler();

    auto module = compiler.parseMLIR(mlir_text);
    if (!module) {
        throw std::runtime_error("Failed to parse MLIR");
    }

    void* fnPtr = compiler.compileAndGetFunctionPtr(module.get(), func_name);
    if (!fnPtr) {
        throw std::runtime_error("Failed to compile function");
    }

    // Prepare output array
    std::vector<ssize_t> out_shape;
    for (auto item : output_shape) {
        out_shape.push_back(py::cast<ssize_t>(item));
    }
    auto output = py::array_t<float>(out_shape);

    // Marshal all input arguments + output
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

    // Call the function with variable number of arguments
    // This is the tricky part - we need to match MLIR's calling convention
    size_t num_args = args.size();
    
    // Use libffi-style invocation for arbitrary arg counts
    // For simplicity, handle common cases explicitly
    if (num_args == 10) {  // 2 inputs (1D) + 1 output (1D): 5+5 params
        using FnPtr = void(*)(void*, void*, void*, void*, void*,
                               void*, void*, void*, void*, void*);
        auto fn = reinterpret_cast<FnPtr>(fnPtr);
        fn(args[0], args[1], args[2], args[3], args[4],
           args[5], args[6], args[7], args[8], args[9]);
    } else if (num_args == 14) {  // 1 input (2D) + 1 output (2D): 7+7 params
        using FnPtr = void(*)(void*, void*, void*, void*, void*, void*, void*,
                               void*, void*, void*, void*, void*, void*, void*);
        auto fn = reinterpret_cast<FnPtr>(fnPtr);
        fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
           args[7], args[8], args[9], args[10], args[11], args[12], args[13]);
    } else if (num_args == 15) {  // 2 inputs (1D) + 1 output (1D): 5+5+5 params
        using FnPtr = void(*)(void*, void*, void*, void*, void*,
                               void*, void*, void*, void*, void*,
                               void*, void*, void*, void*, void*);
        auto fn = reinterpret_cast<FnPtr>(fnPtr);
        fn(args[0], args[1], args[2], args[3], args[4],
           args[5], args[6], args[7], args[8], args[9],
           args[10], args[11], args[12], args[13], args[14]);
    } else if (num_args == 21) {  // 2 inputs (2D) + 1 output (2D): 7+7+7 params
        using FnPtr = void(*)(void*, void*, void*, void*, void*, void*, void*,
                               void*, void*, void*, void*, void*, void*, void*,
                               void*, void*, void*, void*, void*, void*, void*);
        auto fn = reinterpret_cast<FnPtr>(fnPtr);
        fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
           args[7], args[8], args[9], args[10], args[11], args[12], args[13],
           args[14], args[15], args[16], args[17], args[18], args[19], args[20]);
    } else if (num_args == 28) {  // 3 inputs (2D) + 1 output (2D): 7+7+7+7 params
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
                                  ". Extend execute() for this case.");
    }

    return output;
}

} // namespace ch8

PYBIND11_MODULE(ch8, m) {
    m.doc() = "Chapter 8: Custom Dialect with Python Lowering";

    m.def("execute", &ch8::execute,
          "Generic execute function - handles arbitrary inputs",
          py::arg("mlir_text"),
          py::arg("func_name"),
          py::arg("inputs"),
          py::arg("output_shape"));
}