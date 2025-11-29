//===- jit.cpp - JIT Compilation for Softmax ----------------------------===//
//
// This file demonstrates JIT compilation and execution of softmax using LLJIT.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"

#include <memory>

namespace mlir {

// Forward declarations
OwningOpRef<ModuleOp> createSoftmaxModule(MLIRContext& context);
LogicalResult applyLoweringPasses(ModuleOp module);

//===----------------------------------------------------------------------===//
// JIT Compilation Function
//===----------------------------------------------------------------------===//

/// JIT-compiles and executes softmax with dynamic shapes.
///
/// This function:
///   1. Creates high-level MLIR IR (scf.for, math.exp)
///   2. Applies lowering passes (math→libm, scf→cf, *→llvm)
///   3. Translates to LLVM IR
///   4. JIT compiles using LLJIT
///   5. Executes the compiled function
///
/// Args:
///   input: Input array
///   output: Output array (pre-allocated)
///   size: Number of elements
void executeSoftmax(float* input, float* output, int64_t size) {
  // Initialize LLVM targets (one-time initialization)
  static bool initialized = false;
  if (!initialized) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    initialized = true;
  }

  // Create MLIR context and load all necessary dialects
  MLIRContext context;
  context.loadDialect<func::FuncDialect, 
                      LLVM::LLVMDialect,
                      arith::ArithDialect,
                      math::MathDialect,
                      memref::MemRefDialect,
                      scf::SCFDialect>();

  // Generate high-level MLIR IR
  auto mlirModule = createSoftmaxModule(context);
  if (!mlirModule) {
    llvm::errs() << "[JIT] Failed to create softmax module\n";
    return;
  }

  // Apply lowering passes (math→libm, scf→cf, *→llvm)
  if (failed(applyLoweringPasses(*mlirModule))) {
    llvm::errs() << "[JIT] Failed to apply lowering passes\n";
    return;
  }

  // Register dialect translations
  registerBuiltinDialectTranslation(*mlirModule->getContext());
  registerLLVMDialectTranslation(*mlirModule->getContext());

  // Translate MLIR to LLVM IR
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
  auto symOrErr = jit->lookup("softmax");
  if (!symOrErr) {
    llvm::errs() << "[JIT] Failed to look up function: "
                 << llvm::toString(symOrErr.takeError()) << "\n";
    return;
  }

  // Cast to function pointer
  // Signature: void(ptr, ptr, i64, i64, i64, ptr, ptr, i64, i64, i64)
  // Two memref<?xf32> descriptors (input and output)
  using SoftmaxFnPtr = void(*)(
    float*, float*, int64_t, int64_t, int64_t,   // input descriptor
    float*, float*, int64_t, int64_t, int64_t    // output descriptor
  );
  auto softmaxFn = symOrErr->toPtr<SoftmaxFnPtr>();

  // Prepare memref descriptors
  // For contiguous 1D arrays: allocated = aligned, offset = 0, stride = 1
  int64_t offset = 0;
  int64_t stride = 1;

  // Execute JIT-compiled function
  softmaxFn(input, input, offset, size, stride,   // input descriptor
            output, output, offset, size, stride); // output descriptor
}

} // namespace mlir