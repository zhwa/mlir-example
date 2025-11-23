# MLIR GEMM Example - WSL2/Ubuntu

A learning project demonstrating the MLIR compilation pipeline for matrix multiplication with **dynamic shapes** (supports any matrix size).

## Quick Setup

### 1. Install Dependencies (One-Time)

```bash
# Install MLIR and development tools
sudo apt install -y libmlir-18-dev libmlir-18 mlir-18-tools
sudo apt install -y python3-dev python3-numpy ninja-build libzstd-dev

# Install compiler (pick one - both work fine)
sudo apt install -y g++        # GCC C++ compiler (default)
# OR
sudo apt install -y clang-18   # Clang C++ compiler (also works)
```

### 2. Fix LLVM Header Conflict (Required)

**Why needed:** LLVM 18 ships with its own `cxxabi.h` that conflicts with the system's version. When building Python extensions with pybind11, both headers get included, causing compilation errors about duplicate function declarations.

**This affects BOTH g++ and clang++** - the fix is the same for either compiler.

**Why it doesn't affect most users:**
- Most LLVM users don't build Python extensions  
- Pre-built Python wheels don't compile from source
- Pure C++ LLVM projects don't use pybind11

**The simple fix:**
```bash
# Temporarily rename LLVM's cxxabi.h to avoid conflict
sudo mv /usr/lib/llvm-18/include/cxxabi.h /usr/lib/llvm-18/include/cxxabi.h.backup
```

**Compiler choice:** Both g++ and clang++-18 work perfectly after this fix. The preset uses clang++-18 (keeping it in the LLVM family!), but you can switch to g++ by editing `CMakePresets.json` if preferred.

### 3. Build

```bash
cd ~/llvm-example
cmake --preset x64-release
cmake --build --preset x64-release
```

Build output: `build/x64-release/llvm_example.cpython-312-x86_64-linux-gnu.so`

### 4. Test

```bash
python3 test_jit.py
```

**Expected output:**
```
=== Test 1: Ones matrix ===
[JIT] Starting executeGemm with LLJIT
[JIT] Matrix dimensions: A(8x32) × B(32x16) → C(8x16)
[JIT] JIT execution completed successfully!
C[0,0] = 32.0 (expected: 32.0)
All values correct: True

=== Test 2: Random matrices ===
[JIT] JIT execution completed successfully!
Max error vs NumPy: 0.0
Results match NumPy: True
```

## Current Status

- ✅ **Build:** SUCCESS  
- ✅ **Runtime:** SUCCESS - JIT execution works!
- ✅ **Tests:** All passing with correct results
- ✅ **Dynamic Shapes:** Works with any matrix size!

**Key Features:**
- Dynamic matrix dimensions (no fixed size limitations!)
- Using shared LLVM library (`libLLVM-18.so`) - no symbol conflicts
- Shape validation and error handling

## What This Project Does

Demonstrates the MLIR compilation stack with **dynamic shapes**:

1. **IR Generation** (`src/ir.cpp`) - Creates MLIR with `linalg.matmul` on dynamic memrefs (`memref<?x?xf32>`)
2. **Optimization** (`src/lowering.cpp`) - 7-pass lowering: Linalg → SCF → CF → LLVM dialect
3. **JIT Execution** (`src/jit.cpp`) - LLJIT-based compilation and execution with runtime dimensions
4. **Python Bindings** (`src/bindings.cpp`) - NumPy-compatible interface via pybind11

**Operation:** C = A × B where A, B, C can be any compatible sizes (all float32)

**Example:**
```python
import llvm_example
import numpy as np

# Works with any size!
A = np.ones((10, 20), dtype=np.float32)
B = np.ones((20, 15), dtype=np.float32)
C = llvm_example.gemm(A, B)  # Returns 10×15 matrix
```

## Project Structure

```
llvm-example/
├── CMakeLists.txt          # Build configuration
├── CMakePresets.json       # Linux/Ninja presets
├── test_jit.py             # Python test script
├── src/
│   ├── ir.cpp              # MLIR IR generation
│   ├── lowering.cpp        # Optimization pipeline
│   ├── jit.cpp             # JIT execution
│   └── bindings.cpp        # Python bindings
└── build/x64-release/      # Build output
```

## Learning Linalg Dialect

Now that the project is working with dynamic shapes, you can explore the MLIR linalg dialect:

### View Generated MLIR IR

```python
import llvm_example

# See the high-level MLIR with linalg.matmul (dynamic shapes!)
print(llvm_example.test_ir_generation())
# Output: func.func @gemm(%arg0: memref<?x?xf32>, ...)

# See the optimized/lowered LLVM dialect
print(llvm_example.test_optimized_ir())
```

### Explore the Pipeline

1. **High-level IR** (`src/ir.cpp`): 
   - `linalg.matmul` operates on `memref<?x?xf32>` types (dynamic dimensions!)
   - Declarative: "multiply these matrices" without loop details
   - Shape-polymorphic: works with any compatible sizes

2. **Optimization Pipeline** (`src/lowering.cpp`):
   - Canonicalization → Simplify patterns
   - Linalg → Loops → Convert to explicit `scf.for` loops (with runtime bounds!)
   - SCF → CF → Convert to basic blocks
   - MemRef → LLVM → Lower memory operations
   - Arith/Func/CF → LLVM → Complete lowering

3. **JIT Execution** (`src/jit.cpp`):
   - LLJIT compiles LLVM dialect to native code
   - Runtime dimensions passed through memref descriptor
   - Direct function pointer invocation

### Experiment

Try modifying `src/ir.cpp` to:
- Add different `linalg` operations
- Experiment with memref layouts
- Add constraints or assertions

## Improvement Roadmap

See `IMPROVEMENT_ROADMAP.md` and `CHAPTER_2_DYNAMIC_SHAPES.md` for:
- ✅ **Chapter 2:** Dynamic shapes (COMPLETE!)
- 📋 **Chapter 3:** Function caching for better performance
- 📋 **Chapter 4:** Tensor-based approach with bufferization
- 📋 **Chapter 5:** Advanced optimizations (tiling, vectorization)

## Troubleshooting

### Build fails with cxxabi.h conflict

Make sure LLVM's cxxabi.h is renamed:
```bash
sudo mv /usr/lib/llvm-18/include/cxxabi.h /usr/lib/llvm-18/include/cxxabi.h.backup
```

### Import fails with "CommandLine Error"

This is a known LLVM 18 static linking issue. To fix properly, the CMakeLists.txt needs to be updated to use shared LLVM libraries instead of static ones.

Potential fix (not yet implemented):
```cmake
# In CMakeLists.txt after find_package(LLVM ...)
set(LLVM_LINK_LLVM_DYLIB ON CACHE BOOL "" FORCE)
```

### Clean build

```bash
rm -rf build/
cmake --preset x64-release
cmake --build --preset x64-release
```

## Build from Windows PowerShell

```powershell
wsl bash -c "cd ~/llvm-example && cmake --preset x64-release"
wsl bash -c "cd ~/llvm-example && cmake --build --preset x64-release"
wsl bash -c "cd ~/llvm-example && python3 test_jit.py"
```

## Key Design Decisions

- **MemRef-based IR** instead of tensor-based (simpler, avoids bufferization)
- **Fixed dimensions** (8×16×32) for learning purposes
- **System LLVM/MLIR** via apt packages (not vcpkg)
- **g++ compiler** to avoid clang/LLVM header conflicts

## Restoring System Files

If you need to restore the renamed header:
```bash
sudo mv /usr/lib/llvm-18/include/cxxabi.h.backup /usr/lib/llvm-18/include/cxxabi.h
```

## References

- [MLIR Documentation](https://mlir.llvm.org/)
- [Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)
