//===- NNOps.cpp - NN dialect operations ----------------------------------===//
//
// Chapter 9: Custom Dialect with TableGen
//
//===----------------------------------------------------------------------===//

#include "NNOps.h"
#include "NNDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::nn;

#define GET_OP_CLASSES
#include "NNOps.cpp.inc"
