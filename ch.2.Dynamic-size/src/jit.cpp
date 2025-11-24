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

/// JIT-compiles and executes GEMM with DYNAMIC shapes (supports any matrix size).
///
/// This function works with arbitrary matrix dimensions. The workflow is:
///   1. Generate MLIR with dynamic memrefs (memref<?x?xf32>)
///   2. Apply optimization passes (linalg → loops → LLVM)
///   3. Translate to LLVM IR
///   4. JIT compile to native x86_64 code
///   5. Call with actual runtime dimensions
///
/// Args:
///   A: Float array for MxK matrix A (row-major layout)
///   B: Float array for KxN matrix B (row-major layout)
///   C: Float array for MxN matrix C (output, will be overwritten)
///   M: Number of rows in A and C
///   N: Number of columns in B and C
///   K: Number of columns in A and rows in B (shared dimension)
void executeGemm(float* A, float* B, float* C, 
                 int64_t M, int64_t N, int64_t K) {
  llvm::errs() << "[JIT] Starting executeGemm with LLJIT\n";
  llvm::errs() << "[JIT] Matrix dimensions: A(" << M << "x" << K << ") × B(" 
               << K << "x" << N << ") → C(" << M << "x" << N << ")\n";

  // Initialize LLVM (same as static version)
  static bool initialized = false;
  if (!initialized) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    initialized = true;
  }

  // Create MLIR context and generate module
  MLIRContext context;
  context.loadDialect<func::FuncDialect, LLVM::LLVMDialect>();

  auto mlirModule = createGemmModule(context);  // Now creates dynamic version!
  if (!mlirModule) {
    llvm::errs() << "[JIT] Failed to create GEMM module\n";
    return;
  }

  // Apply optimization passes (work perfectly with dynamic shapes!)
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

  // Add the LLVM IR module to LLJIT
  auto TSM = llvm::orc::ThreadSafeModule(
      std::move(llvmModule), 
      std::move(llvmContext)
  );
  if (auto Err = (*JIT)->addIRModule(std::move(TSM))) {
    llvm::errs() << "[JIT] Failed to add IR module: " << Err << "\n";
    return;
  }

  llvm::errs() << "[JIT] Looking up function 'gemm'...\n";

  // Lookup the function
  auto Sym = (*JIT)->lookup("gemm");
  if (!Sym) {
    llvm::errs() << "[JIT] Failed to lookup function: " << Sym.takeError() << "\n";
    return;
  }

  llvm::errs() << "[JIT] Function found, preparing to call...\n";

  // Function signature (21 arguments: 7 for A, 7 for B, 7 for C)
  // Same signature structure, but now we pass RUNTIME dimensions!
  using FnPtr = void(*)(
      float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // A
      float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // B
      float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t   // C
  );

  auto* gemm_func = Sym->toPtr<FnPtr>();

  llvm::errs() << "[JIT] Calling JIT-compiled function with runtime dimensions...\n";
  
  // KEY DIFFERENCE: Pass actual M, N, K instead of hardcoded 8, 32, 16
  // Memref descriptor layout for row-major (C-style) arrays:
  //   - size0 = rows, size1 = cols
  //   - stride0 = cols (skip entire row), stride1 = 1 (contiguous elements)
  gemm_func(
      A, A, 0, M, K, K, 1,      // A: memref<?x?xf32> with actual dimensions M×K
      //       │  │  │  │  └─ stride[1] = 1 (contiguous columns)
      //       │  │  │  └───── stride[0] = K (skip K elements to next row)
      //       │  │  └────────── size[1] = K columns
      //       │  └───────────── size[0] = M rows
      //       └──────────────── offset = 0
      
      B, B, 0, K, N, N, 1,      // B: memref<?x?xf32> with actual dimensions K×N
      C, C, 0, M, N, N, 1       // C: memref<?x?xf32> with actual dimensions M×N
  );

  llvm::errs() << "[JIT] JIT execution completed successfully!\n";
}

} // namespace mlir