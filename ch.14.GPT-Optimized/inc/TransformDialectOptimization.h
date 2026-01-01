//===- TransformDialectOptimization.h - Transform Dialect API ---*- C++ -*-===//
#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

/// Apply Transform Dialect-based optimizations to a module.
/// This uses real transform operations (not traditional passes).
///
/// The function constructs a transform sequence programmatically and
/// applies it to the payload IR using the transform interpreter.
///
/// Returns success if optimizations were applied successfully.
LogicalResult applyTransformDialectOptimizations(ModuleOp module);

} // namespace mlir
