//===- NNDialect.cpp - NN dialect implementation --------------------------===//
//
// Chapter 9: Custom Dialect with TableGen
//
//===----------------------------------------------------------------------===//

#include "NNDialect.h"
#include "NNOps.h"

using namespace mlir;
using namespace mlir::nn;

#include "NNOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// NN Dialect Initialization
//===----------------------------------------------------------------------===//

void NNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NNOps.cpp.inc"
  >();
}
