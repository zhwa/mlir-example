//===- bindings.cpp - Python Bindings for MLIR GEMM ----------------------===//
//
// This file bridges Python and C++ using pybind11. It exposes our JIT-compiled
// GEMM function to Python, accepting NumPy arrays as inputs.
//
// Functions exposed to Python:
//   - gemm(A, B) → C              # Main matrix multiply
//   - test_ir_generation() → str  # Show high-level MLIR
//   - test_optimized_ir() → str   # Show lowered LLVM dialect
//
//===----------------------------------------------------------------------===//

// Header include order matters here!
// MLIR headers must come before pybind11 to avoid cxxabi.h conflicts
// (this is a quirk of LLVM 18 + pybind11 + libstdc++)
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

// Now include pybind11 (after MLIR to avoid cxxabi.h conflicts)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declarations
namespace mlir {
  mlir::OwningOpRef<mlir::ModuleOp> createGemmModule(mlir::MLIRContext &context);
  mlir::LogicalResult applyOptimizationPasses(mlir::ModuleOp module);
  void executeGemm(float* A, float* B, float* C);
}

/// Python-callable matrix multiplication: C = A @ B
///
/// This is the main entry point from Python. It:
///   1. Validates NumPy array shapes and types
///   2. Extracts raw float* pointers
///   3. Calls the JIT-compiled GEMM function
///   4. Returns result as a new NumPy array
///
/// Expected shapes: A is 8×32, B is 32×16, returns C as 8×16
py::array_t<float> gemm(py::array_t<float> A, py::array_t<float> B) {
  // Get buffer info
  auto A_buf = A.request();
  auto B_buf = B.request();

  // Validate input shapes
  if (A_buf.ndim != 2 || B_buf.ndim != 2) {
    throw std::runtime_error("Input arrays must be 2D");
  }

  if (A_buf.shape[0] != 8 || A_buf.shape[1] != 32) {
    throw std::runtime_error("Matrix A must be 8x32, got " + 
                            std::to_string(A_buf.shape[0]) + "x" + 
                            std::to_string(A_buf.shape[1]));
  }

  if (B_buf.shape[0] != 32 || B_buf.shape[1] != 16) {
    throw std::runtime_error("Matrix B must be 32x16, got " + 
                            std::to_string(B_buf.shape[0]) + "x" + 
                            std::to_string(B_buf.shape[1]));
  }

  // Validate data type
  if (A_buf.itemsize != sizeof(float) || B_buf.itemsize != sizeof(float)) {
    throw std::runtime_error("Input arrays must be float32");
  }

  // Allocate output array (8x16)
  auto C = py::array_t<float>({8, 16});
  auto C_buf = C.request();

  // Execute GEMM (currently uses placeholder CPU implementation)
  mlir::executeGemm(
    static_cast<float*>(A_buf.ptr),
    static_cast<float*>(B_buf.ptr),
    static_cast<float*>(C_buf.ptr)
  );

  return C;
}

/// Test function to verify MLIR IR generation and optimization
std::string test_ir_generation() {
  mlir::MLIRContext context;

  // Generate the MLIR module
  auto module = mlir::createGemmModule(context);

  if (!module) {
    return "ERROR: Failed to create MLIR module";
  }

  // Convert to string
  std::string result;
  llvm::raw_string_ostream os(result);
  module->print(os);
  os.flush();

  return result;
}

/// Test function to verify the optimization pipeline
std::string test_optimized_ir() {
  mlir::MLIRContext context;

  // Load all necessary dialects for the optimization pipeline
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  // Generate the MLIR module
  auto module = mlir::createGemmModule(context);

  if (!module) {
    return "ERROR: Failed to create MLIR module";
  }

  // Apply optimization passes
  if (mlir::failed(mlir::applyOptimizationPasses(*module))) {
    return "ERROR: Failed to apply optimization passes";
  }

  // Convert to string
  std::string result;
  llvm::raw_string_ostream os(result);
  module->print(os);
  os.flush();

  return result;
}

PYBIND11_MODULE(ch1_fixed_size, m) {
  m.doc() = "MLIR-powered matrix multiplication accelerator (Chapter 1: Fixed-size)";

  m.def("gemm", &gemm, 
        "Compute C = A @ B where A is 8x32 and B is 32x16",
        py::arg("A"), py::arg("B"));

  m.def("test_ir_generation", &test_ir_generation,
        "Test MLIR IR generation and return the generated MLIR code");

  m.def("test_optimized_ir", &test_optimized_ir,
        "Test the optimization pipeline and return the optimized MLIR code");
}