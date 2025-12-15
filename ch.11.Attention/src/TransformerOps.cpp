//===- TransformerOps.cpp - Transformer operations -----------------------===//
#include "TransformerOps.h"
#include "TransformerDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::transformer;

#define GET_OP_CLASSES
#include "TransformerOps.cpp.inc"