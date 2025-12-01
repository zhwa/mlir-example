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
Universal execution using libffi for production-grade flexibility:

**API**: `execute(mlir_text, func_name, inputs, output_shape)`
- Uses libffi for truly variadic calling
- Handles ANY parameter count without code changes
- Runtime shape introspection
- Automatic memref descriptor construction
- ~80 lines total (vs ~250 for shape-specific helpers)

**Usage**:
```python
import ch8

# Same universal API for all operations:
result = ch8.execute(mlir_text, "add", [a, b], (4,))           # 1D binary
result = ch8.execute(mlir_text, "matmul", [a, b], (2, 4))      # 2D matmul
result = ch8.execute(mlir_text, "relu", [input], (2, 4))       # 2D unary
result = ch8.execute(mlir_text, "mlp", [x, W1, W2], (2, 2))    # 3 inputs
result = ch8.execute(mlir_text, "custom", [many, inputs], out) # Any shape!
```

**Key Design**: libffi eliminates all explicit parameter count cases - one code path handles 10, 21, 28, or 1000 parameters identically.

**Comparison with Chapter 7**: Chapter 7 uses explicit cases for educational transparency (zero overhead, clear code). Chapter 8 uses libffi for production flexibility (minimal overhead ~5-10%, universal).

See `GENERIC_BINDINGS.md` for detailed implementation and IREE comparison.

## Building

```bash
cmake --build build/x64-release --target ch8
```

## Running Tests

```bash
cd ch.8.Custom-dialect

# Main test suite (6 tests including raw MLIR)
PYTHONPATH=build/x64-release/ch.8.Custom-dialect:python python3 test_jit.py
```

## Test Results

All 6 tests pass:
1. ✓ Element-wise Addition (1D binary)
2. ✓ Element-wise Multiplication (1D binary)
3. ✓ Matrix Multiplication (2D)
4. ✓ ReLU Activation (2D unary)
5. ✓ Multi-layer Network (3 inputs, 28 parameters)
6. ✓ Raw MLIR Text Execution (libffi flexibility demo)

## Chapter 7 vs Chapter 8

| Aspect | Chapter 7 | Chapter 8 |
|--------|-----------|-----------|
| **Dialect** | Custom C++ dialect | Text-based "nn" dialect |
| **Lowering** | C++ passes | Python text generation |
| **Bindings** | ✅ Explicit cases (~110 lines) | ✅ libffi universal (~80 lines) |
| **Complexity** | C++ dialect + C++ lowering + explicit bindings | Python graph + Python lowering + libffi bindings |
| **Shape Handling** | Switch on parameter count | Dynamic FFI dispatch |
| **Flexibility** | Common patterns (10, 14, 21, 28 params) | **ANY signature** |
| **Overhead** | Zero (direct calls) | Minimal (~5-10% from libffi) |
| **Maintenance** | Add cases for new patterns | No code changes needed |

**Key Differentiation**:
- **Chapter 7**: Educational transparency - see exactly how explicit cases work
- **Chapter 8**: Production technique - libffi enables universal flexibility

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
- ✓ **libffi-based universal binding** - handles ANY signature
- ✓ Comprehensive comparison with industrial compilers (see `GENERIC_BINDINGS.md`)
- ✓ Production-grade flexibility without sacrificing simplicity

**Key Insights**:
1. libffi eliminates explicit parameter count enumeration
2. Custom dialects are orthogonal to binding complexity (FFI boundary issue)
3. Industrial compilers (IREE) use similar libffi approach
4. Minimal overhead (~5-10%) for huge flexibility gain
5. Chapter 7 (explicit) vs Chapter 8 (libffi) shows educational vs production trade-offs

**Philosophy**: Start simple (Python strings + explicit cases in Ch7), then show production techniques (libffi in Ch8), preparing for industrial tools (TableGen in Ch9).