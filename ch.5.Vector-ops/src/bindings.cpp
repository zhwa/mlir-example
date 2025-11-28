//===- bindings.cpp - Python Bindings for SAXPY --------------------------===//
//
// Python bindings using pybind11. Exposes SAXPY JIT compilation to Python.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declarations
namespace mlir {
  OwningOpRef<ModuleOp> createSaxpyModule(MLIRContext &context);
  LogicalResult applyLoweringPasses(ModuleOp module);
  void executeSaxpy(float alpha, float* A, float* B, float* C, int64_t size);
}

/// Python-callable SAXPY: C = alpha * A + B
///
/// Args:
///   alpha: Scalar multiplier (float)
///   A: NumPy array (1D, float32)
///   B: NumPy array (1D, float32)
///
/// Returns:
///   C: NumPy array (1D, float32) containing result
py::array_t<float> saxpy(float alpha, py::array_t<float> A, py::array_t<float> B) {
  auto A_buf = A.request();
  auto B_buf = B.request();

  // Validate inputs
  if (A_buf.ndim != 1 || B_buf.ndim != 1) {
    throw std::runtime_error("Input arrays must be 1D");
  }

  if (A_buf.shape[0] != B_buf.shape[0]) {
    throw std::runtime_error("Arrays must have same size");
  }

  if (A_buf.itemsize != sizeof(float) || B_buf.itemsize != sizeof(float)) {
    throw std::runtime_error("Input arrays must be float32");
  }

  // Get size
  int64_t size = A_buf.shape[0];

  // Allocate output
  auto C = py::array_t<float>(size);
  auto C_buf = C.request();

  // Execute SAXPY via JIT
  mlir::executeSaxpy(
    alpha,
    static_cast<float*>(A_buf.ptr),
    static_cast<float*>(B_buf.ptr),
    static_cast<float*>(C_buf.ptr),
    size
  );

  return C;
}

/// Test function: Show generated MLIR IR
std::string test_ir_generation() {
  mlir::MLIRContext context;
  auto module = mlir::createSaxpyModule(context);

  if (!module) {
    return "ERROR: Failed to create MLIR module";
  }

  std::string result;
  llvm::raw_string_ostream stream(result);
  module->print(stream);
  return result;
}

/// Test function: Show lowered MLIR IR (after passes)
std::string test_lowered_ir() {
  mlir::MLIRContext context;
  auto module = mlir::createSaxpyModule(context);

  if (!module) {
    return "ERROR: Failed to create MLIR module";
  }

  if (mlir::failed(mlir::applyLoweringPasses(*module))) {
    return "ERROR: Failed to apply lowering passes";
  }

  std::string result;
  llvm::raw_string_ostream stream(result);
  module->print(stream);
  return result;
}

/// Python module definition
PYBIND11_MODULE(ch5_vector_ops, m) {
  m.doc() = "MLIR SAXPY implementation using SCF dialect";

  m.def("saxpy", &saxpy,
        "Compute C = alpha * A + B using MLIR JIT",
        py::arg("alpha"), py::arg("A"), py::arg("B"));

  m.def("test_ir_generation", &test_ir_generation,
        "Generate and return high-level MLIR IR");

  m.def("test_lowered_ir", &test_lowered_ir,
        "Generate and return lowered MLIR IR");
}