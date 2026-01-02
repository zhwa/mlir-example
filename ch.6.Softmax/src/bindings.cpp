//===- bindings.cpp - Python Bindings for Softmax -----------------------===//
//
// This file exposes the softmax JIT compiler to Python via pybind11.
//
//===----------------------------------------------------------------------===//

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/raw_ostream.h"

namespace py = pybind11;

namespace mlir {

// Forward declarations
void executeSoftmax(float* input, float* output, int64_t size);
std::string printSoftmaxModule(MLIRContext& context);
OwningOpRef<ModuleOp> createSoftmaxModule(MLIRContext& context);
LogicalResult applyLoweringPasses(ModuleOp module);

} // namespace mlir

//===----------------------------------------------------------------------===//
// Python Wrapper Functions
//===----------------------------------------------------------------------===//

/// Python-exposed softmax function
///
/// Takes a NumPy array and returns the softmax-normalized result.
/// The output array is automatically allocated.
///
/// Args:
///   input: 1D NumPy array of float32
///
/// Returns:
///   1D NumPy array of float32 (probabilities summing to 1.0)
py::array_t<float> softmax(const py::array_t<float>& input) {
  // Get buffer info
  auto buf = input.request();

  // Validate input
  if (buf.ndim != 1) {
    throw std::runtime_error("Input must be 1-dimensional");
  }

  int64_t size = buf.shape[0];

  // Allocate output array
  auto output = py::array_t<float>(size);
  auto output_buf = output.request();

  // Call JIT execution
  mlir::executeSoftmax(
      static_cast<float*>(buf.ptr),
      static_cast<float*>(output_buf.ptr),
      size
  );

  return output;
}

/// Test function to show generated high-level IR
std::string test_ir_generation() {
  mlir::MLIRContext context;
  return mlir::printSoftmaxModule(context);
}

/// Test function to show lowered IR (after passes)
std::string test_lowered_ir() {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect,
                      mlir::LLVM::LLVMDialect,
                      mlir::arith::ArithDialect,
                      mlir::math::MathDialect,
                      mlir::memref::MemRefDialect,
                      mlir::scf::SCFDialect>();

  auto module = mlir::createSoftmaxModule(context);
  if (!module) {
    return "Failed to create module";
  }

  if (mlir::failed(mlir::applyLoweringPasses(*module))) {
    return "Failed to apply lowering passes";
  }

  std::string result;
  llvm::raw_string_ostream os(result);
  module->print(os);
  return result;
}

//===----------------------------------------------------------------------===//
// Python Module Definition
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(ch6_softmax, m) {
  m.doc() = "MLIR JIT-compiled softmax using Math dialect";

  m.def("softmax", &softmax,
        "Compute softmax normalization of input array",
        py::arg("input"));

  m.def("test_ir_generation", &test_ir_generation,
        "Generate and print high-level MLIR IR for softmax");

  m.def("test_lowered_ir", &test_lowered_ir,
        "Generate and print lowered MLIR IR (LLVM dialect)");
}