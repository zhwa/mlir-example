//===- TransformerPasses.h - Transformer passes ------------------*- C++ -*-===//
#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>
namespace mlir {

/// Create pass to lower Transformer dialect ops to standard dialects
std::unique_ptr<Pass> createLowerTransformerToStandardPass();

} // namespace mlir