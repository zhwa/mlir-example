//===- TransformDialectOptimization.h - Transform Dialect Optimizations -*-===//
//
// Header for Transform Dialect-based optimization functions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace mlir {

/// Apply Transform Dialect optimizations to the given module
/// Returns success if all optimizations applied successfully
LogicalResult applyTransformDialectOptimizations(ModuleOp module);

} // namespace mlir
