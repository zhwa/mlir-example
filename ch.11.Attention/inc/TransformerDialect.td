//===- TransformerDialect.td - Transformer dialect --------*- tablegen -*-===//
//
// Defines the Transformer dialect for attention mechanisms and related ops.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMER_DIALECT
#define TRANSFORMER_DIALECT

include "mlir/IR/OpBase.td"

def Transformer_Dialect : Dialect {
  let name = "transformer";
  let summary = "Transformer attention operations";
  let description = [{
    The Transformer dialect provides operations for implementing
    attention mechanisms and transformer architectures.

    Operations work on memref types for efficient computation.
  }];
  let cppNamespace = "::mlir::transformer";
}

// Base class for transformer dialect operations
class Transformer_Op<string mnemonic, list<Trait> traits = []> :
    Op<Transformer_Dialect, mnemonic, traits>;

#endif // TRANSFORMER_DIALECT