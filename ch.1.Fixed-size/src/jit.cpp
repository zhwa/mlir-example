//===- jit.cpp - JIT Execution using ExecutionEngine ---------------------===//
//
// This file demonstrates Just-In-Time (JIT) compilation using MLIR's official
// ExecutionEngine wrapper:
//   1. Generate MLIR IR (high-level)
//   2. Lower to LLVM IR (automatic via ExecutionEngine)
//   3. Compile to native machine code (x86_64 instructions)
//   4. Execute directly in memory (no files written to disk!)
//
// ExecutionEngine is MLIR's official wrapper around LLVM's LLJIT (ORC v2).
// It simplifies the JIT workflow by handling LLVM IR translation and 
// optimization internally.
//
// The trickiest part: memref descriptors expand to 7 arguments each!
//   memref<8x32xf32> → (ptr, ptr, offset, size0, size1, stride0, stride1)
//
//===----------------------------------------------------------------------====//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"
#include <cstdint>

namespace mlir {

// Forward declarations from ir.cpp and lowering.cpp
OwningOpRef<ModuleOp> createGemmModule(MLIRContext& context);
LogicalResult applyOptimizationPasses(ModuleOp module);

/// JIT-compiles and executes the GEMM function using ExecutionEngine.
///
/// This function generates machine code at runtime using MLIR's official
/// ExecutionEngine API. The workflow is:
///   1. Generate MLIR (high-level matrix multiply)
///   2. Apply optimization passes (linalg → loops → LLVM)
///   3. ExecutionEngine handles LLVM IR translation + optimization internally
///   4. Look up the function pointer
///   5. Call it directly!
///
/// Args:
///   A: Float array for 8×32 matrix A (row-major layout)
///   B: Float array for 32×16 matrix B (row-major layout)
///   C: Float array for 8×16 matrix C (output, will be overwritten)
void executeGemm(float* A, float* B, float* C) {
  llvm::errs() << "[JIT] Starting executeGemm with ExecutionEngine\n";

  // Step 1: Initialize LLVM's code generation for our CPU
  // This tells LLVM what instruction set we support (x86_64, ARM, etc.)
  // Only needs to happen once per process (hence the static guard)
  static bool initialized = false;
  if (!initialized) {
    llvm::InitializeNativeTarget();           // Enable codegen for our CPU
    llvm::InitializeNativeTargetAsmPrinter(); // Enable assembly output
    initialized = true;
  }

  // Create MLIR context and generate module
  MLIRContext context;
  context.loadDialect<func::FuncDialect, LLVM::LLVMDialect>();

  auto mlirModule = createGemmModule(context);
  if (!mlirModule) {
    llvm::errs() << "[JIT] Failed to create GEMM module\n";
    return;
  }

  // Apply optimization passes (lowers to LLVM dialect)
  if (failed(applyOptimizationPasses(*mlirModule))) {
    llvm::errs() << "[JIT] Failed to apply optimization passes\n";
    return;
  }

  // Register LLVM dialect translations (required for ExecutionEngine)
  registerBuiltinDialectTranslation(*mlirModule->getContext());
  registerLLVMDialectTranslation(*mlirModule->getContext());

  llvm::errs() << "[JIT] Creating ExecutionEngine...\n";

  // Create ExecutionEngine with optimization pipeline
  // ExecutionEngine handles LLVM IR translation internally
  mlir::ExecutionEngineOptions options;
  auto transformer = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
  options.transformer = std::move(transformer);

  auto maybeEngine = mlir::ExecutionEngine::create(*mlirModule, options);
  if (!maybeEngine) {
    llvm::errs() << "[JIT] Failed to create ExecutionEngine: " 
                 << maybeEngine.takeError() << "\n";
    return;
  }

  auto engine = std::move(*maybeEngine);

  llvm::errs() << "[JIT] Looking up function 'gemm_8x16x32'...\n";

  // Lookup the function
  auto expectedFPtr = engine->lookup("gemm_8x16x32");
  if (!expectedFPtr) {
    llvm::errs() << "[JIT] Failed to lookup function: " 
                 << expectedFPtr.takeError() << "\n";
    return;
  }

  llvm::errs() << "[JIT] Function found, preparing to call...\n";

  // Function signature (21 arguments: 7 for A, 7 for B, 7 for C)
  using FnPtr = void(*)(
      float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // A
      float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // B
      float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t   // C
  );

  auto* gemm_func = reinterpret_cast<FnPtr>(*expectedFPtr);

  llvm::errs() << "[JIT] Calling JIT-compiled function...\n";
  gemm_func(
      A, A, 0, 8, 32, 32, 1,      // A: memref<8x32xf32>
      B, B, 0, 32, 16, 16, 1,     // B: memref<32x16xf32>
      C, C, 0, 8, 16, 16, 1       // C: memref<8x16xf32>
  );

  llvm::errs() << "[JIT] JIT execution completed successfully!\n";
}

} // namespace mlir