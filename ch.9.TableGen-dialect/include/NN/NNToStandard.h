//===- NNToStandard.h - NN to Standard dialect conversion -----*- C++ -*-===//
#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

// Create a pass to convert NN dialect to standard MLIR dialects
std::unique_ptr<OperationPass<ModuleOp>> createConvertNNToStandardPass();

} // namespace mlir