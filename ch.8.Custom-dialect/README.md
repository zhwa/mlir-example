# Chapter 8: Custom Dialect - Python String-Based Implementation

## Overview

This chapter demonstrates custom dialect workflows using a **Python string-based approach**. This pedagogical implementation clearly shows MLIR concepts without the complexity of C++ TableGen or Python IR builders.

**Key Design**: Simple, readable, educational - perfect for understanding MLIR before moving to Chapter 9's TableGen approach.

## Architecture

```
Python Graph Builder (dataclasses)
           ↓
  Track operations as tuples
           ↓
Python Lowering → MLIR text strings
           ↓
C++ MLIR Parser & Compiler
           ↓
     LLJIT Execution
```

**Philosophy**: Use Python's simplicity for graph building and MLIR text generation. Let C++ handle parsing, optimization, and JIT compilation.

## Key Components

### 1. **Graph Builder** (`python/graph_builder.py`)
PyTorch-style deferred execution API:
```python
g = Graph()
x = g.variable([4])
y = g.variable([4])
z = g.add(x, y)
```

### 2. **Python Lowering** (`python/lowering.py`)
Converts high-level `nn` operations to standard MLIR dialects:
- `nn.add` → `linalg.generic` + `arith.addf`
- `nn.mul` → `linalg.generic` + `arith.mulf`
- `nn.matmul` → `linalg.matmul`
- `nn.relu` → `linalg.generic` + `arith.maximumf`

### 3. **C++ Compiler** (`src/compiler.cpp`)
Standard MLIR pipeline using LLJIT (same as Chapter 7):
- Parse MLIR text
- Apply lowering passes: linalg → loops → SCF → CF → LLVM
- JIT compile with LLJIT

### 4. **Python Bindings** (`src/bindings.cpp`)
Generic execution handler with automatic memref marshaling:

**API**: `execute(mlir_text, func_name, inputs, output_shape)`
- Handles arbitrary number of inputs (1D or 2D)
- Runtime shape introspection
- Automatic memref descriptor construction
- Single function replaces all shape-specific helpers

**Usage**:
```python
import ch8

# All operations use the same generic API:
result = ch8.execute(mlir_text, "add", [a, b], (4,))           # 1D binary
result = ch8.execute(mlir_text, "matmul", [a, b], (2, 4))      # 2D matmul
result = ch8.execute(mlir_text, "relu", [input], (2, 4))       # 2D unary
result = ch8.execute(mlir_text, "mlp", [x, W1, W2], (2, 2))    # Multi-input
```

**Key Design**: Shape-generic binding eliminates the need for operation-specific or shape-specific helper functions.

## Building

```bash
cmake --build build/x64-release --target ch8
```

## Running Tests

```bash
cd ch.8.Custom-dialect
PYTHONPATH=build/x64-release/ch.8.Custom-dialect:python python3 test_jit.py
```

## Test Results

All 5 tests pass:
1. ✓ Element-wise Addition
2. ✓ Element-wise Multiplication  
3. ✓ Matrix Multiplication
4. ✓ ReLU Activation
5. ✓ Multi-layer Network (2 layers with ReLU)

## Chapter 7 vs Chapter 8

| Aspect | Chapter 7 | Chapter 8 |
|--------|-----------|-----------|
| **Dialect** | Custom C++ dialect | Text-based "nn" dialect |
| **Lowering** | C++ passes | Python text generation |
| **Bindings** | ✅ `execute_generic(fnPtr, inputs, shape)` | ✅ `execute(mlir, func, inputs, shape)` |
| **Complexity** | C++ dialect + C++ lowering + generic bindings | Python graph + Python lowering + generic bindings |
| **Shape Handling** | Runtime introspection | Runtime introspection |
| **Code Lines (bindings)** | ~110 lines (generic only) | ~100 lines (generic only) |

**Key Achievement**: Generic binding layer uses runtime shape introspection to eliminate all shape-specific helper functions. Both chapters benefit from the same clean, maintainable API design.

## Key Insight

**Chapter 8's advantage**: Custom dialects move ABI complexity from bindings (C++) to lowering passes (Python). The bindings become shape-generic, while the lowering handles operation-specific semantics. This demonstrates how MLIR's multi-level IR design separates concerns effectively.

## Architecture

```
Python Graph Builder
         ↓
  nn dialect (text)
         ↓
Python Lowering (MLIRLowering class)
         ↓
Standard MLIR (linalg, arith, memref)
         ↓
C++ Compiler (linalg → loops → LLVM)
         ↓
  LLJIT Execution
         ↓
     NumPy Arrays
```

## Comparison with Chapter 9 (TableGen)

This chapter sets up a direct comparison for the next chapter:

| Aspect | Chapter 8 (Python) | Chapter 9 (TableGen) |
|--------|-------------------|---------------------|
| **Dialect Definition** | Implicit in Python strings | Explicit C++ classes via TableGen |
| **Lowering** | Python string generation | C++ RewritePatterns |
| **Type Safety** | Runtime (Python) | Compile-time (C++) |
| **Debugging** | Print MLIR strings | MLIR verifier + C++ tools |
| **Performance** | Parse strings at runtime | Direct IR construction |
| **Learning Curve** | Gentle (Python + MLIR text) | Steep (C++ + TableGen + ODS) |

**When to use Python approach**: Prototyping, learning, DSLs with simple lowering  
**When to use TableGen**: Production compilers, complex transformations, performance-critical

## Fixed Issues

1. **LLVM CommandLine Error**: Fixed with `llvm::cl::ResetCommandLineParser()` before initialization
2. **linalg.generic Returns**: Changed from returning memrefs to output parameters
3. **Calling Convention**: Fixed memref parameter passing (5 params for 1D, 7 for 2D)
4. **Import Errors**: Resolved Python module import paths

## Summary

**Chapter 8 Achievements**:
- ✓ Python-based custom dialect workflow (string generation)
- ✓ Clean separation: Python (high-level) vs C++ (compilation)
- ✓ Educational foundation for understanding MLIR concepts
- ✓ **Generic binding layer** - single `execute()` function for all operations
- ✓ Ready for Chapter 9's TableGen comparison

**Key Insight**: The generic binding layer (added to both Chapter 7 and 8) solves the shape-specific binding problem through runtime introspection. This is orthogonal to the custom dialect concept - both chapters benefit equally.

**Philosophy**: Start simple (Python strings) to understand concepts, then move to industrial tools (TableGen) for production use.