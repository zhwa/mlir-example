# MLIR GEMM Example - WSL2/Ubuntu

A learning project demonstrating the MLIR compilation pipeline for matrix multiplication (8×32 × 32×16 GEMM).

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

**Key Fix:** Using shared LLVM library (`libLLVM-18.so`) instead of static libraries eliminates the CommandLine option duplicate registration issue.

## What This Project Does

Demonstrates the MLIR compilation stack:

1. **IR Generation** (`src/ir.cpp`) - Creates MLIR with `linalg.matmul` on memrefs
2. **Optimization** (`src/lowering.cpp`) - 7-pass lowering: Linalg → SCF → CF → LLVM dialect
3. **JIT Execution** (`src/jit.cpp`) - LLJIT-based compilation and execution
4. **Python Bindings** (`src/bindings.cpp`) - NumPy-compatible interface via pybind11

**Operation:** C = A × B where A is 8×32, B is 32×16, C is 8×16

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

Now that the project is working, you can explore the MLIR linalg dialect:

### View Generated MLIR IR

```python
import llvm_example

# See the high-level MLIR with linalg.matmul
print(llvm_example.test_ir_generation())

# See the optimized/lowered LLVM dialect
print(llvm_example.test_optimized_ir())
```

### Explore the Pipeline

1. **High-level IR** (`src/ir.cpp`): 
   - `linalg.matmul` operates on `memref<8x32xf32>` types
   - Declarative: "multiply these matrices" without loop details

2. **Optimization Pipeline** (`src/lowering.cpp`):
   - Canonicalization → Simplify patterns
   - Linalg → Loops → Convert to explicit `scf.for` loops
   - SCF → CF → Convert to basic blocks
   - MemRef → LLVM → Lower memory operations
   - Arith/Func/CF → LLVM → Complete lowering

3. **JIT Execution** (`src/jit.cpp`):
   - LLJIT compiles LLVM dialect to native code
   - Explicit symbol resolution for `malloc`/`free`
   - Direct function pointer invocation

### Experiment

Try modifying `src/ir.cpp` to:
- Change matrix dimensions
- Add `linalg.fill` to zero-initialize output
- Use `linalg.transpose` or `linalg.add`
- Explore different memref layouts

Try modifying `src/lowering.cpp` to:
- Add loop tiling (uncomment tiling pass)
- Experiment with vectorization
- Change optimization order

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
