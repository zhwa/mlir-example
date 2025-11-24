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
  void executeGemm(float* A, float* B, float* C, int64_t M, int64_t N, int64_t K);
}

/// Python-callable matrix multiplication: C = A @ B (any compatible size!)
///
/// This function accepts matrices of arbitrary sizes and:
///   1. Extracts NumPy array shapes dynamically
///   2. Validates dimensions are compatible (A.cols == B.rows)
///   3. Calls executeGemm with runtime dimensions
///   4. Returns result as a new NumPy array
///
/// Expected shapes: A is M×K, B is K×N, returns C as M×N
py::array_t<float> gemm(py::array_t<float> A, py::array_t<float> B) {
  // Get buffer info
  auto A_buf = A.request();
  auto B_buf = B.request();

  // Validate input arrays are 2D
  if (A_buf.ndim != 2 || B_buf.ndim != 2) {
    throw std::runtime_error("Input arrays must be 2D");
  }

  // Extract dimensions
  int64_t M = A_buf.shape[0];  // Rows in A
  int64_t K = A_buf.shape[1];  // Cols in A (must match rows in B)
  int64_t K2 = B_buf.shape[0]; // Rows in B
  int64_t N = B_buf.shape[1];  // Cols in B

  // Validate matrix multiplication compatibility
  if (K != K2) {
    throw std::runtime_error(
        "Matrix dimensions incompatible for multiplication: A is " +
        std::to_string(M) + "x" + std::to_string(K) + 
        ", B is " + std::to_string(K2) + "x" + std::to_string(N) +
        " (A.cols must equal B.rows)"
    );
  }

  // Validate data type
  if (A_buf.itemsize != sizeof(float) || B_buf.itemsize != sizeof(float)) {
    throw std::runtime_error("Input arrays must be float32");
  }

  // Allocate output array (M×N)
  auto C = py::array_t<float>({M, N});
  auto C_buf = C.request();

  // Execute GEMM with runtime dimensions
  mlir::executeGemm(
    static_cast<float*>(A_buf.ptr),
    static_cast<float*>(B_buf.ptr),
    static_cast<float*>(C_buf.ptr),
    M, N, K  // Pass runtime dimensions!
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

PYBIND11_MODULE(ch3_jit_caching, m) {
  m.doc() = "MLIR-powered matrix multiplication accelerator (Chapter 3: JIT-caching)";

  m.def("gemm", &gemm, 
        "Compute C = A @ B for matrices of any compatible size",
        py::arg("A"), py::arg("B"));

  m.def("test_ir_generation", &test_ir_generation,
        "Test MLIR IR generation and return the generated MLIR code");

  m.def("test_optimized_ir", &test_optimized_ir,
        "Test the optimization pipeline and return the optimized MLIR code");
}