//===- jit.cpp - JIT Execution using LLJIT -------------------------------===//
//
// This file demonstrates Just-In-Time (JIT) compilation:
//   1. Generate MLIR IR with dynamic shapes (high-level)
//   2. Lower to LLVM IR (low-level)
//   3. Compile to native machine code (x86_64 instructions)
//   4. Execute directly in memory (no files written to disk!)
//
// LLJIT is LLVM's modern JIT API (ORC v2). It's more powerful than the
// older MCJIT API and is the recommended approach for new projects.
//
// The trickiest part: memref descriptors expand to 7 arguments each!
//   memref<?x?xf32> → (ptr, ptr, offset, size0, size1, stride0, stride1)
//   Runtime dimensions are passed through size0 and size1!
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <memory>

namespace mlir {

// Forward declarations from ir.cpp and lowering.cpp
OwningOpRef<ModuleOp> createGemmModule(MLIRContext& context);
OwningOpRef<ModuleOp> createGemmModuleTensor(MLIRContext& context);
LogicalResult applyOptimizationPasses(ModuleOp module);

//===----------------------------------------------------------------------===//
// Cache Infrastructure for JIT-compiled Functions
//===----------------------------------------------------------------------===//

/// MemRef descriptor structure (matches MLIR's memref lowering)
struct MemRefDescriptor {
  float* allocated;   // Pointer to allocated memory
  float* aligned;     // Aligned pointer for data access
  int64_t offset;     // Offset from base
  int64_t sizes[2];   // [rows, cols]
  int64_t strides[2]; // [row_stride, col_stride]
};

/// Function pointer type for compiled GEMM.
/// Takes 3 memref descriptors (A, B, C) as out-parameter style (void return)
using GemmFnPtr = void(*)(
    float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // A
    float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // B
    float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t   // C
);

/// Global cache for the JIT-compiled GEMM function.
/// Since we use dynamic shapes (memref<?x?xf32>), the compiled code is
/// shape-agnostic and works for ANY matrix dimensions. We only need to
/// compile ONCE and reuse for all shapes!
struct GlobalJITCache {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  GemmFnPtr funcPtr = nullptr;
  bool isCompiled = false;
} gGemmJIT;

//===----------------------------------------------------------------------===//
// JIT Compilation Function
//===----------------------------------------------------------------------===//

/// Compiles GEMM function and returns the callable function.
/// This function performs the expensive JIT compilation.
/// Since we use dynamic shapes, this only needs to be called ONCE!
///
/// Returns: pair<LLJIT instance, function pointer> or {nullptr, nullptr} on error.
std::pair<std::unique_ptr<llvm::orc::LLJIT>, GemmFnPtr> 
compileGemmFunction() {
  llvm::errs() << "[JIT] Compiling shape-agnostic GEMM function...\n";

  // Initialize LLVM (same as static version)
  static bool initialized = false;
  if (!initialized) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    initialized = true;
  }

  // Create MLIR context and load all necessary dialects
  MLIRContext context;
  context.loadDialect<func::FuncDialect, 
                      LLVM::LLVMDialect,
                      arith::ArithDialect,
                      tensor::TensorDialect,
                      linalg::LinalgDialect,
                      memref::MemRefDialect,
                      bufferization::BufferizationDialect,
                      scf::SCFDialect,
                      cf::ControlFlowDialect>();

  // Use tensor-based IR generation (will be bufferized during lowering!)
  auto mlirModule = createGemmModuleTensor(context);
  if (!mlirModule) {
    llvm::errs() << "[JIT] Failed to create GEMM module\n";
    return {nullptr, nullptr};
  }

  // Apply optimization passes (includes bufferization: tensor→memref)
  if (failed(applyOptimizationPasses(*mlirModule))) {
    llvm::errs() << "[JIT] Failed to apply optimization passes\n";
    return {nullptr, nullptr};
  }

  // Register dialect translations
  registerBuiltinDialectTranslation(*mlirModule->getContext());
  registerLLVMDialectTranslation(*mlirModule->getContext());

  // Translate MLIR to LLVM IR
  llvm::errs() << "[JIT] Translating MLIR to LLVM IR...\n";
  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = translateModuleToLLVMIR(*mlirModule, *llvmContext);
  if (!llvmModule) {
    llvm::errs() << "[JIT] Failed to translate MLIR to LLVM IR\n";
    return {nullptr, nullptr};
  }

  // Apply LLVM optimizations
  auto optPipeline = makeOptimizingTransformer(
      /*optLevel=*/3,
      /*sizeLevel=*/0,
      /*targetMachine=*/nullptr
  );
  if (auto Err = optPipeline(llvmModule.get())) {
    llvm::errs() << "[JIT] Failed to apply LLVM optimization: " << Err << "\n";
    return {nullptr, nullptr};
  }

  llvm::errs() << "[JIT] Creating LLJIT instance...\n";

  // Create LLJIT
  auto JIT = llvm::orc::LLJITBuilder().create();
  if (!JIT) {
    llvm::errs() << "[JIT] Failed to create LLJIT: " << JIT.takeError() << "\n";
    return {nullptr, nullptr};
  }

  // Add the LLVM IR module to LLJIT
  auto TSM = llvm::orc::ThreadSafeModule(
      std::move(llvmModule), 
      std::move(llvmContext)
  );
  if (auto Err = (*JIT)->addIRModule(std::move(TSM))) {
    llvm::errs() << "[JIT] Failed to add IR module: " << Err << "\n";
    return {nullptr, nullptr};
  }

  llvm::errs() << "[JIT] Looking up function 'gemm'...\n";

  // Lookup the function
  auto Sym = (*JIT)->lookup("gemm");
  if (!Sym) {
    llvm::errs() << "[JIT] Failed to lookup function: " << Sym.takeError() << "\n";
    return {nullptr, nullptr};
  }

  llvm::errs() << "[JIT] Function found! Compilation successful.\n";

  // Extract function pointer and return along with JIT instance
  auto* gemm_func = Sym->toPtr<GemmFnPtr>();
  return {std::move(*JIT), gemm_func};
}

//===----------------------------------------------------------------------===//
// Execution Function
//===----------------------------------------------------------------------===//

/// JIT-compiles and executes GEMM with DYNAMIC shapes (supports any matrix size).
///
/// This function works with arbitrary matrix dimensions. The workflow is:
///   1. Check if we've compiled the function yet
///   2. If not compiled: Compile once and cache (expensive, ~10-100ms)
///   3. If compiled: Use cached function (fast, <1ms)
///   4. Call the compiled function with actual data pointers and runtime dimensions
///
/// Key insight: Since we use dynamic shapes (memref<?x?xf32>), the compiled
/// code works for ANY dimensions. We only compile ONCE, not once per shape!
///
/// Args:
///   A: Float array for MxK matrix A (row-major layout)
///   B: Float array for KxN matrix B (row-major layout)
///   C: Float array for MxN matrix C (output, will be overwritten)
///   M: Number of rows in A and C
///   N: Number of columns in B and C
///   K: Number of columns in A and rows in B (shared dimension)
void executeGemm(float* A, float* B, float* C, int64_t M, int64_t N, int64_t K) {
  llvm::errs() << "[JIT] Starting executeGemm\n";
  llvm::errs() << "[JIT] Matrix dimensions: A(" << M << "x" << K << ") × B(" 
               << K << "x" << N << ") → C(" << M << "x" << N << ")\n";

  // Check if we've already compiled the function
  if (!gGemmJIT.isCompiled) {
    llvm::errs() << "[JIT] ✗ First call - compiling function...\n";
    
    auto [jit, func] = compileGemmFunction();
    
    if (!func) {
      llvm::errs() << "[JIT] Compilation failed!\n";
      return;
    }
    
    // Store in global cache
    gGemmJIT.jit = std::move(jit);
    gGemmJIT.funcPtr = func;
    gGemmJIT.isCompiled = true;
    
    llvm::errs() << "[JIT] ✓ Function compiled and cached!\n";
  } else {
    llvm::errs() << "[JIT] ✓ Using cached function (works for any shape!)\n";
  }

  llvm::errs() << "[JIT] Calling JIT-compiled function with runtime dimensions...\n";
  
  // Call the cached function
  GemmFnPtr gemm_func = gGemmJIT.funcPtr;
  
  // Memref descriptor layout for row-major (C-style) arrays:
  gemm_func(
      A, A, 0, M, K, K, 1,      // A: memref<?x?xf32>
      B, B, 0, K, N, N, 1,      // B: memref<?x?xf32>
      C, C, 0, M, N, N, 1       // C: memref<?x?xf32>
  );

  llvm::errs() << "[JIT] JIT execution completed successfully!\n";
}

} // namespace mlir