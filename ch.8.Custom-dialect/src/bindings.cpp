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

// Execute function for 1D binary operations (add, mul)
py::array_t<float> execute_binary_1d(const std::string& mlir_text,
                                      const std::string& func_name,
                                      py::array_t<float> lhs,
                                      py::array_t<float> rhs) {
    Compiler& compiler = getCompiler();

    auto module = compiler.parseMLIR(mlir_text);
    if (!module) {
        throw std::runtime_error("Failed to parse MLIR");
    }

    void* fnPtr = compiler.compileAndGetFunctionPtr(module.get(), func_name);
    if (!fnPtr) {
        throw std::runtime_error("Failed to compile function");
    }

    // Prepare output
    auto output = py::array_t<float>(lhs.size());

    // Marshal arguments: lhs, rhs, output
    std::vector<void*> args;
    marshal_memref_1d(args, lhs);
    marshal_memref_1d(args, rhs);
    marshal_memref_1d(args, output);

    // Execute (3 memrefs × 5 params each = 15 args)
    using FnPtr = void(*)(void*, void*, void*, void*, void*,  // lhs (5 args)
                           void*, void*, void*, void*, void*,  // rhs (5 args)
                           void*, void*, void*, void*, void*); // output (5 args)
    auto fn = reinterpret_cast<FnPtr>(fnPtr);
    fn(args[0], args[1], args[2], args[3], args[4],
       args[5], args[6], args[7], args[8], args[9],
       args[10], args[11], args[12], args[13], args[14]);

    return output;
}

// Execute function for matmul
py::array_t<float> execute_matmul(const std::string& mlir_text,
                                   const std::string& func_name,
                                   py::array_t<float> lhs,
                                   py::array_t<float> rhs) {
    Compiler& compiler = getCompiler();

    auto module = compiler.parseMLIR(mlir_text);
    if (!module) {
        throw std::runtime_error("Failed to parse MLIR");
    }

    void* fnPtr = compiler.compileAndGetFunctionPtr(module.get(), func_name);
    if (!fnPtr) {
        throw std::runtime_error("Failed to compile function");
    }

    // Prepare output (M x N)
    auto lhs_buf = lhs.request();
    auto rhs_buf = rhs.request();
    ssize_t M = lhs_buf.shape[0];
    ssize_t N = rhs_buf.shape[1];
    auto output = py::array_t<float>({M, N});

    // Marshal arguments
    std::vector<void*> args;
    marshal_memref_2d(args, lhs);
    marshal_memref_2d(args, rhs);
    marshal_memref_2d(args, output);

    // Execute (3 memrefs × 7 params each = 21 args)
    using FnPtr = void(*)(void*, void*, void*, void*, void*, void*, void*,  // lhs (7 args)
                           void*, void*, void*, void*, void*, void*, void*,  // rhs (7 args)
                           void*, void*, void*, void*, void*, void*, void*); // output (7 args)
    auto fn = reinterpret_cast<FnPtr>(fnPtr);
    fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
       args[7], args[8], args[9], args[10], args[11], args[12], args[13],
       args[14], args[15], args[16], args[17], args[18], args[19], args[20]);

    return output;
}

// Execute function for 2D unary operations (relu)
py::array_t<float> execute_unary_2d(const std::string& mlir_text,
                                     const std::string& func_name,
                                     py::array_t<float> input) {
    Compiler& compiler = getCompiler();

    auto module = compiler.parseMLIR(mlir_text);
    if (!module) {
        throw std::runtime_error("Failed to parse MLIR");
    }

    void* fnPtr = compiler.compileAndGetFunctionPtr(module.get(), func_name);
    if (!fnPtr) {
        throw std::runtime_error("Failed to compile function");
    }

    // Prepare output (same shape as input)
    auto buf = input.request();
    auto output = py::array_t<float>({buf.shape[0], buf.shape[1]});

    // Marshal arguments
    std::vector<void*> args;
    marshal_memref_2d(args, input);
    marshal_memref_2d(args, output);

    // Execute (2 memrefs × 7 params each = 14 args)
    using FnPtr = void(*)(void*, void*, void*, void*, void*, void*, void*,  // input (7 args)
                           void*, void*, void*, void*, void*, void*, void*); // output (7 args)
    auto fn = reinterpret_cast<FnPtr>(fnPtr);
    fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
       args[7], args[8], args[9], args[10], args[11], args[12], args[13]);

    return output;
}

// Execute function for multi-input operations (3 inputs, 1 output)
py::array_t<float> execute_3inputs_2d(const std::string& mlir_text,
                                       const std::string& func_name,
                                       py::array_t<float> in1,
                                       py::array_t<float> in2,
                                       py::array_t<float> in3) {
    Compiler& compiler = getCompiler();

    auto module = compiler.parseMLIR(mlir_text);
    if (!module) {
        throw std::runtime_error("Failed to parse MLIR");
    }

    void* fnPtr = compiler.compileAndGetFunctionPtr(module.get(), func_name);
    if (!fnPtr) {
        throw std::runtime_error("Failed to compile function");
    }

    // Prepare output - infer from computation
    // For MLP: x @ W1 -> relu -> @ W2
    // Output shape: [in1.shape[0], in3.shape[1]]
    auto in1_buf = in1.request();
    auto in3_buf = in3.request();
    auto output = py::array_t<float>({in1_buf.shape[0], in3_buf.shape[1]});

    // Marshal arguments
    std::vector<void*> args;
    marshal_memref_2d(args, in1);
    marshal_memref_2d(args, in2);
    marshal_memref_2d(args, in3);
    marshal_memref_2d(args, output);

    // Execute (4 memrefs × 7 params each = 28 args)
    using FnPtr = void(*)(void*, void*, void*, void*, void*, void*, void*,  // in1 (7 args)
                           void*, void*, void*, void*, void*, void*, void*,  // in2 (7 args)
                           void*, void*, void*, void*, void*, void*, void*,  // in3 (7 args)
                           void*, void*, void*, void*, void*, void*, void*); // output (7 args)
    auto fn = reinterpret_cast<FnPtr>(fnPtr);
    fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
       args[7], args[8], args[9], args[10], args[11], args[12], args[13],
       args[14], args[15], args[16], args[17], args[18], args[19], args[20],
       args[21], args[22], args[23], args[24], args[25], args[26], args[27]);

    return output;
}

} // namespace ch8

PYBIND11_MODULE(ch8, m) {
    m.doc() = "Chapter 8: Custom Dialect with Python Lowering";

    m.def("execute_binary_1d", &ch8::execute_binary_1d,
          "Execute binary 1D operation (add, mul)",
          py::arg("mlir_text"),
          py::arg("func_name"),
          py::arg("lhs"),
          py::arg("rhs"));

    m.def("execute_matmul", &ch8::execute_matmul,
          "Execute matrix multiplication",
          py::arg("mlir_text"),
          py::arg("func_name"),
          py::arg("lhs"),
          py::arg("rhs"));

    m.def("execute_unary_2d", &ch8::execute_unary_2d,
          "Execute unary 2D operation (relu)",
          py::arg("mlir_text"),
          py::arg("func_name"),
          py::arg("input"));

    m.def("execute_3inputs_2d", &ch8::execute_3inputs_2d,
          "Execute operation with 3 2D inputs (multi-layer network)",
          py::arg("mlir_text"),
          py::arg("func_name"),
          py::arg("in1"),
          py::arg("in2"),
          py::arg("in3"));
}