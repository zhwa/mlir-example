//===- TransformDialectOptimization.cpp - Transform Dialect Optimizations -*-===//
//
// This file implements REAL Transform Dialect-based optimizations using
// external transform scripts executed via the Transform Dialect interpreter.
//
// This COMPLETELY REPLACES traditional passes with Transform Dialect operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

#include <dlfcn.h>
#include <filesystem>

using namespace mlir;

/// Get the directory where this shared library is loaded from
static std::filesystem::path getModuleDirectory() {
  Dl_info info;
  if (dladdr((void*)getModuleDirectory, &info)) {
    return std::filesystem::path(info.dli_fname).parent_path();
  }
  return {};
}

/// Apply Transform Dialect optimizations using external script
///
/// This uses REAL Transform Dialect operations from an external .mlir script.
/// Uses the lower-level Transform Dialect API that doesn't require specific
/// module structure.
///
static LogicalResult applyTransformDialectOptimizationsImpl(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  
  // ==========================================================================
  // Load transform script
  // ==========================================================================
  auto moduleDir = getModuleDirectory();
  if (moduleDir.empty()) {
    llvm::errs() << "ERROR: Could not determine module directory\n";
    return failure();
  }
  
  auto scriptPath = moduleDir / "optimize.mlir";
  std::string scriptPathStr = scriptPath.string();
  
  llvm::outs() << "Loading Transform Dialect script: " << scriptPathStr << "\n";
  
  auto transformModule = parseSourceFile<ModuleOp>(scriptPathStr, ctx);
  if (!transformModule) {
    llvm::errs() << "ERROR: Failed to load transform script\n";
    return failure();
  }
  
  llvm::outs() << "✓ Loaded transform script\n";
  
  // ==========================================================================
  // Execute transform via lower-level API
  // ==========================================================================
  // Use applyTransforms with RaggedArray API for more control
  
  llvm::outs() << "Executing Transform Dialect sequence...\n";
  
  // Find the top-level transform op (either named_sequence or sequence)
  Operation *topLevelTransformOp = nullptr;
  for (Operation &op : transformModule->getBodyRegion().getOps()) {
    if (auto transformOp = dyn_cast<transform::TransformOpInterface>(&op)) {
      topLevelTransformOp = transformOp;
      break;
    }
  }
  
  if (!topLevelTransformOp) {
    llvm::errs() << "ERROR: No top-level transform op found in script\n";
    return failure();
  }
  
  // Apply the transformations using Transform Dialect interpreter
  transform::TransformOptions options;
  
  if (failed(transform::applyTransforms(
          module.getOperation(),                        // Payload root - the module we want to optimize
          cast<transform::TransformOpInterface>(topLevelTransformOp),  // Transform operation
          {},                                           // No extra mappings
          options))) {
    llvm::errs() << "ERROR: Transform sequence execution failed\n";
    return failure();
  }
  
  llvm::outs() << "✓ Transform Dialect optimizations applied (100% Transform ops, 0% passes)\n";
  return success();
}

// C++ interface for direct calls
namespace mlir {
LogicalResult applyTransformDialectOptimizations(ModuleOp module) {
  return applyTransformDialectOptimizationsImpl(module);
}
} // namespace mlir
