//===- TransformerDialect.cpp - Transformer dialect --------------*- C++ -*-===//
#include "TransformerDialect.h"
#include "TransformerOps.h"

using namespace mlir;
using namespace mlir::transformer;

#include "TransformerDialect.cpp.inc"

void mlir::transformer::TransformerDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TransformerOps.cpp.inc"
      >();
}