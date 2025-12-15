//===- TransformerDialect.td - Transformer dialect --------*- tablegen -*-===//
//
// Chapter 12: Transformer Block Dialect
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMER_DIALECT
#define TRANSFORMER_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Transformer Dialect Definition
//===----------------------------------------------------------------------===//

def Transformer_Dialect : Dialect {
  let name = "transformer";
  let summary = "Transformer operations for neural networks";
  let description = [{
    The Transformer dialect provides operations for building transformer models:
    - Layer normalization
    - Multi-head attention
    - Feed-forward networks
    - Activation functions (GELU)
    - Linear projections with bias
    
    This dialect emphasizes clean Python API with Tensor abstraction and
    deferred execution via computation graphs.
  }];
  
  let cppNamespace = "::mlir::transformer";
}

// Base class for Transformer dialect operations
class Transformer_Op<string mnemonic, list<Trait> traits = []> :
    Op<Transformer_Dialect, mnemonic, traits>;

#endif // TRANSFORMER_DIALECT