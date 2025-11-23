//===- jit.cpp - JIT Execution using LLJIT -------------------------------===//
//
// This file demonstrates Just-In-Time (JIT) compilation:
//   1. Generate MLIR IR (high-level)
//   2. Lower to LLVM IR (low-level)
//   3. Compile to native machine code (x86_64 instructions)
//   4. Execute directly in memory (no files written to disk!)
//
// LLJIT is LLVM's modern JIT API (ORC v2). It's more powerful than the
// older MCJIT API and is the recommended approach for new projects.
//
// The trickiest part: memref descriptors expand to 7 arguments each!
//   memref<8x32xf32> → (ptr, ptr, offset, size0, size1, stride0, stride1)
//
//===----------------------------------------------------------------------====//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

namespace mlir {

// Forward declarations from ir.cpp and lowering.cpp
OwningOpRef<ModuleOp> createGemmModule(MLIRContext& context);
LogicalResult applyOptimizationPasses(ModuleOp module);

/// JIT-compiles and executes the GEMM function.
///
/// This function does something remarkable: it generates machine code at runtime
/// and executes it immediately. The workflow is:
///   1. Generate MLIR (high-level matrix multiply)
///   2. Apply optimization passes (linalg → loops → LLVM)
///   3. Translate to LLVM IR
///   4. JIT compile to native x86_64 code
///   5. Look up the function pointer
///   6. Call it directly!
///
/// Args:
///   A: Float array for 8×32 matrix A (row-major layout)
///   B: Float array for 32×16 matrix B (row-major layout)
///   C: Float array for 8×16 matrix C (output, will be overwritten)
void executeGemm(float* A, float* B, float* C) {
  llvm::errs() << "[JIT] Starting executeGemm with LLJIT\n";

  // Step 1: Initialize LLVM's code generation for our CPU
  // This tells LLVM what instruction set we support (x86_64, ARM, etc.)
  // Only needs to happen once per process (hence the static guard)
  static bool initialized = false;
  if (!initialized) {
    llvm::InitializeNativeTarget();           // Enable codegen for our CPU
    llvm::InitializeNativeTargetAsmPrinter(); // Enable assembly output
    llvm::InitializeNativeTargetAsmParser();  // Enable assembly input
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

  // Apply optimization passes
  if (failed(applyOptimizationPasses(*mlirModule))) {
    llvm::errs() << "[JIT] Failed to apply optimization passes\n";
    return;
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
    return;
  }

  // Apply LLVM optimizations
  auto optPipeline = makeOptimizingTransformer(
      /*optLevel=*/3,
      /*sizeLevel=*/0,
      /*targetMachine=*/nullptr
  );
  if (auto Err = optPipeline(llvmModule.get())) {
    llvm::errs() << "[JIT] Failed to apply LLVM optimization: " << Err << "\n";
    return;
  }

  llvm::errs() << "[JIT] Creating LLJIT instance...\n";

  // Create LLJIT
  auto JIT = llvm::orc::LLJITBuilder().create();
  if (!JIT) {
    llvm::errs() << "[JIT] Failed to create LLJIT: " << JIT.takeError() << "\n";
    return;
  }

  // Add the LLVM IR module to LLJIT (use same context)
  auto TSM = llvm::orc::ThreadSafeModule(
      std::move(llvmModule), 
      std::move(llvmContext)
  );
  if (auto Err = (*JIT)->addIRModule(std::move(TSM))) {
    llvm::errs() << "[JIT] Failed to add IR module: " << Err << "\n";
    return;
  }

  llvm::errs() << "[JIT] Looking up function 'gemm_8x16x32'...\n";

  // Lookup the function
  auto Sym = (*JIT)->lookup("gemm_8x16x32");
  if (!Sym) {
    llvm::errs() << "[JIT] Failed to lookup function: " << Sym.takeError() << "\n";
    return;
  }

  llvm::errs() << "[JIT] Function found, preparing to call...\n";

  // Function signature (21 arguments: 7 for A, 7 for B, 7 for C)
  using FnPtr = void(*)(
      float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // A
      float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // B
      float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t   // C
  );

  auto* gemm_func = Sym->toPtr<FnPtr>();

  llvm::errs() << "[JIT] Calling JIT-compiled function...\n";
  gemm_func(
      A, A, 0, 8, 32, 32, 1,      // A: memref<8x32xf32>
      B, B, 0, 32, 16, 16, 1,     // B: memref<32x16xf32>
      C, C, 0, 8, 16, 16, 1       // C: memref<8x16xf32>
  );

  llvm::errs() << "[JIT] JIT execution completed successfully!\n";
}

} // namespace mlir