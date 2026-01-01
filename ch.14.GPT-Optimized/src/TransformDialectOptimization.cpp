//===- TransformDialectOptimization.cpp - Transform Dialect Optimizations -*-===//
//
// This file implements REAL Transform Dialect-based optimizations using
// EMBEDDED transform scripts (production pattern used by major MLIR projects).
//
// The transform script is embedded as a string literal and parsed once at
// initialization. This production approach provides:
// - No external file dependencies
// - Parse once, cache the result
// - Zero runtime I/O overhead
// - Transform script is part of the binary
//
//===----------------------------------------------------------------------===//

#include "TransformDialectOptimization.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Embedded Transform Script (Production Pattern)
//===----------------------------------------------------------------------===//

// Transform Dialect optimization sequence
// This is embedded directly in the binary, no external file needed
static constexpr const char *kOptimizeTransformIR = R"mlir(
// Transform Dialect optimization sequence for GPT model
//
// This uses REAL Transform Dialect operations to optimize the IR.
// Replaces traditional passes with declarative transform specifications.

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // Apply canonicalization patterns to simplify IR
  // Includes CSE, folding, dead code elimination
  transform.apply_patterns to %arg0 {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
}
)mlir";

//===----------------------------------------------------------------------===//
// Cached Transform Module
//===----------------------------------------------------------------------===//

// Parse transform script once and cache it
// This avoids reparsing on every compilation (production optimization)
static OwningOpRef<ModuleOp> getCachedTransformModule(MLIRContext *ctx) {
  static OwningOpRef<ModuleOp> cachedModule;
  static std::once_flag initFlag;

  std::call_once(initFlag, [ctx]() {
    ParserConfig config(ctx);
    cachedModule = parseSourceString<ModuleOp>(kOptimizeTransformIR, config);
    if (!cachedModule) {
      llvm::errs() << "FATAL: Failed to parse embedded transform script\n";
    }
  });

  return cachedModule.get() ? cachedModule.get().clone() : OwningOpRef<ModuleOp>();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

namespace mlir {

/// Apply Transform Dialect optimizations using embedded transform script
///
/// Production approach (used by major MLIR-based compilers):
/// 1. Transform script embedded as string literal in the binary
/// 2. Parse once, cache the parsed module
/// 3. Clone and apply to each compilation target
/// 
/// Benefits:
/// - No external file dependencies
/// - No runtime file I/O
/// - Parse overhead paid only once
/// - Proven production pattern
///
LogicalResult applyTransformDialectOptimizations(ModuleOp module) {
  MLIRContext *ctx = module.getContext();

  llvm::outs() << "Applying embedded Transform Dialect optimizations...\n";

  // Get the cached (and cloned) transform module
  auto transformModule = getCachedTransformModule(ctx);
  if (!transformModule) {
    llvm::errs() << "ERROR: Failed to get transform module\n";
    return failure();
  }

  // Find the top-level transform operation
  Operation *topLevelTransformOp = nullptr;
  for (Operation &op : transformModule->getBodyRegion().getOps()) {
    if (auto transformOp = dyn_cast<transform::TransformOpInterface>(&op)) {
      topLevelTransformOp = transformOp;
      break;
    }
  }

  if (!topLevelTransformOp) {
    llvm::errs() << "ERROR: No transform operation found in embedded script\n";
    return failure();
  }

  // Apply the transformations
  transform::TransformOptions options;

  if (failed(transform::applyTransforms(
          module.getOperation(),
          cast<transform::TransformOpInterface>(topLevelTransformOp),
          {},
          options))) {
    llvm::errs() << "ERROR: Transform sequence execution failed\n";
    return failure();
  }

  llvm::outs() << "✓ Transform Dialect optimizations applied (cached transform script)\n";
  return success();
}

} // namespace mlir