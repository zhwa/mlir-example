#include "compiler.cpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <ffi.h>

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

// libffi-based execute - handles ANY signature universally
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

    // Execute the function
    ffi_call(&cif, FFI_FN(fnPtr), nullptr, arg_values.data());

    return output;
}

} // namespace ch8

PYBIND11_MODULE(ch8, m) {
    m.doc() = "Chapter 8: Custom Dialect with Python Lowering (libffi-based)";

    m.def("execute", &ch8::execute,
          "Universal execute using libffi - handles ANY signature",
          py::arg("mlir_text"),
          py::arg("func_name"),
          py::arg("inputs"),
          py::arg("output_shape"));
}