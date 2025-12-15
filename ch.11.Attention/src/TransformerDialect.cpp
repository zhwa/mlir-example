//===- TransformerDialect.cpp - Transformer dialect ----------------------===//
#include "TransformerDialect.h"
#include "TransformerOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::transformer;

#include "TransformerDialect.cpp.inc"

void TransformerDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TransformerOps.cpp.inc"
  >();
}