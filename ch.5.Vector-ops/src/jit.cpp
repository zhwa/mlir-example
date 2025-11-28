//===- jit.cpp - JIT Execution for SAXPY ---------------------------------===//
//
// JIT compilation workflow:
//   1. Generate MLIR IR with SCF
//   2. Lower to LLVM IR
//   3. JIT compile to native code
//   4. Execute with dynamic sizes
//
// Key difference from Chapter 1: Dynamic memrefs (memref<?xf32>)
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
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include <cstdint>

namespace mlir {

// Forward declarations
OwningOpRef<ModuleOp> createSaxpyModule(MLIRContext& context);
LogicalResult applyLoweringPasses(ModuleOp module);

/// JIT-compiles and executes SAXPY with dynamic sizes.
///
/// MemRef descriptor for memref<?xf32> expands to 5 arguments:
///   (ptr, ptr, offset, size, stride)
///
/// Full signature:
///   void saxpy(float alpha,
///              float* A_allocated, float* A_aligned, int64_t A_offset, int64_t A_size, int64_t A_stride,
///              float* B_allocated, float* B_aligned, int64_t B_offset, int64_t B_size, int64_t B_stride,
///              float* C_allocated, float* C_aligned, int64_t C_offset, int64_t C_size, int64_t C_stride)
void executeSaxpy(float alpha, float* A, float* B, float* C, int64_t size) {
  // Initialize LLVM once
  static bool initialized = false;
  if (!initialized) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    initialized = true;
  }

  // Create MLIR context
  MLIRContext context;
  context.loadDialect<func::FuncDialect, LLVM::LLVMDialect>();

  // Generate MLIR module
  auto mlirModule = createSaxpyModule(context);
  if (!mlirModule) {
    llvm::errs() << "[JIT] Failed to create SAXPY module\n";
    return;
  }

  // Apply lowering passes
  if (failed(applyLoweringPasses(*mlirModule))) {
    llvm::errs() << "[JIT] Failed to apply lowering passes\n";
    return;
  }

  // Register LLVM dialect translation
  registerBuiltinDialectTranslation(context);
  registerLLVMDialectTranslation(context);

  // Translate MLIR → LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(*mlirModule, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "[JIT] Failed to translate to LLVM IR\n";
    return;
  }

  // Create LLJIT instance
  auto jitOrErr = llvm::orc::LLJITBuilder().create();
  if (!jitOrErr) {
    llvm::errs() << "[JIT] Failed to create LLJIT: "
                 << llvm::toString(jitOrErr.takeError()) << "\n";
    return;
  }
  auto jit = std::move(*jitOrErr);

  // Add LLVM module to JIT
  auto tsm = llvm::orc::ThreadSafeModule(std::move(llvmModule),
                                         std::make_unique<llvm::LLVMContext>());
  if (auto err = jit->addIRModule(std::move(tsm))) {
    llvm::errs() << "[JIT] Failed to add module: "
                 << llvm::toString(std::move(err)) << "\n";
    return;
  }

  // Look up the function
  auto symOrErr = jit->lookup("saxpy");
  if (!symOrErr) {
    llvm::errs() << "[JIT] Failed to look up function: "
                 << llvm::toString(symOrErr.takeError()) << "\n";
    return;
  }

  // Cast to function pointer
  // Signature: void(float, ptr, ptr, i64, i64, i64, ptr, ptr, i64, i64, i64, ptr, ptr, i64, i64, i64)
  using SaxpyFnPtr = void(*)(
    float,                                         // alpha
    float*, float*, int64_t, int64_t, int64_t,   // A descriptor
    float*, float*, int64_t, int64_t, int64_t,   // B descriptor
    float*, float*, int64_t, int64_t, int64_t    // C descriptor
  );
  auto saxpyFn = symOrErr->toPtr<SaxpyFnPtr>();

  // Prepare memref descriptors
  // For contiguous arrays: allocated = aligned, offset = 0, stride = 1
  int64_t offset = 0;
  int64_t stride = 1;

  // Execute JIT-compiled function
  saxpyFn(alpha,
          A, A, offset, size, stride,  // A descriptor
          B, B, offset, size, stride,  // B descriptor
          C, C, offset, size, stride); // C descriptor
}

} // namespace mlir