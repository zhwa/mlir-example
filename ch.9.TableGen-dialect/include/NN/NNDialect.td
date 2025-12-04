//===- NNDialect.td - NN dialect definition ----------------*- tablegen -*-===//
//
// Chapter 9: Custom Dialect with TableGen
// Neural Network Dialect Definition
//
//===----------------------------------------------------------------------===//

#ifndef NN_DIALECT
#define NN_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// NN Dialect Definition
//===----------------------------------------------------------------------===//

def NN_Dialect : Dialect {
  let name = "nn";
  let summary = "Neural Network Operations Dialect";
  let description = [{
    This dialect provides high-level operations for neural networks,
    including linear layers, activations, and normalization operations.
    
    Operations are designed to lower to standard MLIR dialects (linalg, arith, math).
  }];
  
  let cppNamespace = "::mlir::nn";
  
  // Disable automatic attribute/type parsers (use default empty implementations)
  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
}

//===----------------------------------------------------------------------===//
// Base NN Operation Definition
//===----------------------------------------------------------------------===//

class NN_Op<string mnemonic, list<Trait> traits = []> :
    Op<NN_Dialect, mnemonic, traits>;

#endif // NN_DIALECT
