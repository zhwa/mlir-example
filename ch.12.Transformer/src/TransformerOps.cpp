//===- TransformerOps.cpp - Transformer operations ---------------*- C++ -*-===//
#include "TransformerOps.h"
#include "TransformerDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::transformer;

#define GET_OP_CLASSES
#include "TransformerOps.cpp.inc"