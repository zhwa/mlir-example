//===- NNDialect.cpp - NN dialect implementation --------------------------===//
//
// Chapter 9: Custom Dialect with TableGen
//
//===----------------------------------------------------------------------===//

#include "NN/NNDialect.h"
#include "NN/NNOps.h"

using namespace mlir;
using namespace mlir::nn;

#include "NN/NNOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// NN Dialect Initialization
//===----------------------------------------------------------------------===//

void NNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NN/NNOps.cpp.inc"
  >();
}
