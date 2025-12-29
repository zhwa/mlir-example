# Learning Linalg Dialect - Chapter 1: Fixed-Size Matrices

**Goal**: Understand MLIR basics with hardcoded 8×32 × 32×16 matrix multiplication.

**Key Concepts**:
- MLIR IR generation using Linalg dialect (`linalg.matmul`)
- Progressive lowering through optimization passes
- JIT execution with `mlir::ExecutionEngine`
- Python bindings with pybind11

## Quick Start

```bash
# Build
cmake --build --preset x64-release --target ch1-fixed-size

# Test
cd ch.1.Fixed-size
python3 test_jit.py
```

### View Generated MLIR IR

```python
import ch1_fixed_size

# See the high-level MLIR with linalg.matmul
print(ch1_fixed_size.test_ir_generation())

# See the optimized/lowered LLVM dialect
print(ch1_fixed_size.test_optimized_ir())
```

## Architecture Overview

### 1. IR Generation (`src/ir.cpp`)
Generates high-level MLIR with `linalg.matmul`:
- Input: `memref<8x32xf32>` (A matrix)
- Input: `memref<32x16xf32>` (B matrix)  
- Output: `memref<8x16xf32>` (C matrix)
- Declarative: "multiply these matrices" without loop implementation details

### 2. Optimization Pipeline (`src/lowering.cpp`)
Progressive lowering in 6 phases:

```
Phase 1: Canonicalization → Simplify patterns
Phase 2: Linalg → Loops → Convert to explicit scf.for loops
Phase 3: SCF → Control Flow → Convert to basic blocks (cf.br)
Phase 4: MemRef → LLVM → Lower memory operations
Phase 5: Arith/Func/CF → LLVM → Complete lowering to LLVM dialect
Phase 6: Cleanup → Remove unrealized cast operations
```

**Key transformation**: `linalg.matmul` becomes three nested loops:
```mlir
scf.for %i = 0 to 8
  scf.for %j = 0 to 16
    scf.for %k = 0 to 32
      C[i,j] += A[i,k] * B[k,j]
```

### 3. JIT Execution (`src/jit.cpp`)
- Uses `mlir::ExecutionEngine` to compile LLVM dialect to native code
- Explicit symbol resolution for `malloc`/`free`
- Direct function pointer invocation with optimizations enabled (O3)

### 4. Python Bindings (`src/bindings.cpp`)
```python
import ch1_fixed_size
C = ch1_fixed_size.gemm(A, B)  # A: 8×32, B: 32×16 → C: 8×16
```

## Experiments

Try modifying `src/ir.cpp` to:
- Change matrix dimensions (requires recompilation)
- Add `linalg.fill` to zero-initialize output
- Use `linalg.transpose` or `linalg.add`
- Explore different memref layouts

Try modifying `src/lowering.cpp` to:
- Reorder passes and observe effects
- Add debug prints between phases
- Compare different optimization levels

## Key Design Decisions

- **MemRef-based IR** instead of tensor-based (simpler, avoids bufferization complexity)
- **Fixed dimensions** (8×32×16) hardcoded for learning purposes
- **LLVM 21** via official APT repository for latest optimizations
- **Clang-21** compiler for consistency with LLVM toolchain

## Troubleshooting

### Module not found
```bash
# Make sure the module is built:
cmake --build --preset x64-release --target ch1-fixed-size
# Check the build directory contains ch1_fixed_size.*.so
ls build/x64-release/ch.1.Fixed-size/
```

### Runtime assertion failures
Common issues:
1. **Missing insertion point** in `ir.cpp`:
   ```cpp
   builder.setInsertionPointToStart(module.getBody());  // Required!
   ```

2. **Function name mismatch** between `lowering.cpp` export and `bindings.cpp` import

### Build errors
If you see MLIR API errors, verify you're using LLVM 21:
```bash
clang-21 --version  # Should show version 21.x
```