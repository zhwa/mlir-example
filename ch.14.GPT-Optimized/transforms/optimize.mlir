// Transform Dialect optimization sequence for GPT model
//
// This file contains the transformation pipeline using REAL Transform Dialect
// operations. It replaces traditional passes with declarative transform IR.

// Top-level transform sequence that applies optimizations
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // Apply canonicalization patterns to simplify IR
  // Includes CSE, folding, dead code elimination
  transform.apply_patterns to %arg0 {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
}
