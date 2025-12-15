//===- TransformerToStandard.h - Lower to standard dialects -------*- C++ -*-===//
#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace transformer {

// Lower Transformer dialect operations to standard dialects (scf, arith, math, memref)
std::unique_ptr<Pass> createLowerTransformerToStandardPass();

} // namespace transformer
} // namespace mlir