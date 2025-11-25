# Learning Linalg Dialect

### View Generated MLIR IR

```python
import ch1_fixed_size

# See the high-level MLIR with linalg.matmul
print(ch1_fixed_size.test_ir_generation())

# See the optimized/lowered LLVM dialect
print(ch1_fixed_size.test_optimized_ir())
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

### Runtime Errors

Errors made:

1. forgot the entry in ir.cpp

```cpp
  // Set insertion point to the module body
  builder.setInsertionPointToStart(module.getBody());
```

Without this, the JIT will fail and test won't give the right result.

2. function name mismatch

```bash
ImportError: /home/zhe/mlir-example/ch.1.Fixed-size/../build/x64-release/ch.1.Fixed-size/ch1_fixed_size.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZN4mlir23applyOptimizationPassesENS_8ModuleOpE
```


This error is caused by a typo in the function name in lowering.cpp. Ironically, the build was successful, and the error was a runtime error.
