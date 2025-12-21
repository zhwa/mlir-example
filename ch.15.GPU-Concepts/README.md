# Chapter 15: GPU Programming with MLIR

**Learn GPU programming concepts through CPU emulation - no GPU hardware needed!**

## Quick Start

```bash
# Build
cd /home/zhe/mlir-example/build/x64-release
cmake --build . --target ch15

# Run all tests
cd /home/zhe/mlir-example/ch.15.GPU-Concepts
python3 test_all.py

# Or run specific phase
python3 test_all.py --phase 0  # 1D thread hierarchy
python3 test_all.py --phase 1  # 2D matrix multiplication
python3 test_all.py --phase 2  # Element-wise operations
python3 test_all.py --phase 3  # Softmax with reductions
```

Expected: ✅ **21/21 tests passing** (6 Phase 0 + 4 Phase 1 + 7 Phase 2 + 4 Phase 3)

## Status

**Phase 0**: ✅ Complete (6/6 tests)
- Vector addition with 1D GPU thread hierarchy
- Thread indexing and bounds checking

**Phase 1**: ✅ Complete (4/4 tests)
- 2D matrix multiplication
- 2D thread hierarchy (row/col indexing)
- Multiple matrix sizes (tiny, square, rectangular, non-aligned)

**Phase 2**: ✅ Complete (7/7 tests)
- GELU activation (with polynomial tanh approximation)
- Element-wise Add and Mul operations
- **Major achievement**: Solved MLIR 19.1.7 JIT constant pool bug!

**Phase 3**: ✅ Complete (4/4 tests)
- Softmax with block-level reductions (max, sum)
- Three-pass algorithm (max → exp → normalize)
- Taylor series approximation for exponential function
- Numerical stability (subtract max before exp)
- Applied all lessons from Phase 2 (no hangs!)

**Next**: Phase 4 - LayerNorm (multi-stage reductions)

## Documentation

📖 **[TUTORIAL.md](TUTORIAL.md)** - Complete guide with:
- GPU concepts (Grid, Blocks, Threads)
- Phase 0: 1D thread hierarchy
- Phase 1: 2D matrix multiplication
- Phase 2: Element-wise operations
- **Critical Bug**: MLIR 19 constant pool issue (complete debugging journey)
- Phase 3: Softmax with reductions and Taylor series
- Common mistakes and solutions
- Code walkthrough
- Testing guide

## What You'll Learn

**Phase 0 (1D)**:
- GPU thread hierarchy (Grid → Blocks → Threads)
- 1D parallel index calculations: `globalIdx = blockIdx * blockSize + threadIdx`
- Bounds checking for safety
- Common MLIR pitfalls

**Phase 1 (2D)**:
- 2D thread organization (blocks in 2D grid)
- 2D index calculations: `row = blockIdx.x * 16 + threadIdx.x`
- Matrix multiplication algorithm
- Debugging ABI issues (22nd parameter segfault)
- Optimization pitfalls (O2 infinite hang)

**Phase 2 (Element-wise)**:
- Perfect parallelism (no thread dependencies)
- GELU activation with polynomial approximation
- **Deep debugging**: 8-iteration investigation of MLIR 19 JIT bug
- Workaround: Passing float constants as function arguments
- Real-world compiler bug experience

**Phase 3 (Reductions)**:
- Block-level reductions (max, sum)
- Multi-pass algorithms (3 passes with synchronization)
- **Taylor series**: Approximating exponential function (math background!)
- Numerical stability techniques (subtract max before exp)
- Softmax for neural networks (attention, classification)
- Applying lessons learned (Phase 2 → Phase 3 success)

## Files

```
ch.15.GPU-Concepts/
├── README.md           # This file
├── TUTORIAL.md         # Complete learning guide (all phases + bug documentation)
├── CMakeLists.txt      # Build configuration
├── src/
│   └── bindings.cpp    # MLIR implementation + Python bindings
└── test_all.py         # Comprehensive test suite (all phases)
```

**Simplified Structure**: Merged all phase tests into one file, removed duplicate documentation.

## Educational Approach

We use **SCF loop emulation** instead of the `gpu` dialect for simplicity:
- More direct transformation (easier to understand)
- No complex lowering passes needed  
- Standard debugging tools work
- Same concepts transfer to real GPUs later

See [TUTORIAL.md](TUTORIAL.md) for detailed explanation.

## Key Concepts

**Thread Hierarchy**:
```
Grid → Blocks (outer loop) → Threads (inner loop)
```

**Index Calculation**:
```cpp
global_index = blockIdx * blockSize + threadIdx
```

**Bounds Checking** (critical!):
```cpp
if (global_index < N) {
  // Process element
}
```

## Future Phases

- Phase 1: 2D MatMul (2D grid indexing)
- Phase 2: Element-wise ops (GELU, etc.)
- Phase 3: Softmax (reductions)
- Phase 4: LayerNorm
- Phase 5: GPT integration
- Phase 6: KV cache

Total timeline: ~3 weeks

## Building

```bash
cd /home/zhe/mlir-example
cmake --preset x64-release
cd build/x64-release
cmake --build . --target ch15
```

## Testing

```bash
cd /home/zhe/mlir-example/ch.15.GPU-Concepts
python3 test_phase0.py
```

## Key Concepts Demonstrated

### 1. Thread Hierarchy
- **Grid**: Collection of blocks (e.g., N/256 blocks)
- **Block**: Collection of threads (e.g., 256 threads)
- **Thread**: Individual execution unit

### 2. Index Calculation
```
global_index = blockIdx * blockSize + threadIdx
```

### 3. Bounds Checking
```
if (global_index < N) {
  // Process element
}
```

Critical for non-aligned sizes (e.g., N=1337 with blockSize=256)

### 4. Parallel Patterns
- **Embarrassingly Parallel**: Vector addition (no dependencies)
- **Thread-per-element**: Each thread processes one output

## Educational Value

This approach teaches GPU concepts **better** than real GPU:
1. **Visible Transformation**: See GPU concepts → CPU loops
2. **Easy Debugging**: Use standard CPU debuggers (GDB, print statements)
3. **No Hardware Barrier**: Anyone can learn
4. **Transferable Knowledge**: Concepts apply to any GPU (CUDA, ROCm, Metal)

## Future Phases

- **Phase 1**: 2D MatMul (block/thread indexing in 2D)
- **Phase 2**: Element-wise ops (GELU, Add, Mul)
- **Phase 3**: Softmax (reductions, barriers)
- **Phase 4**: LayerNorm (variance reduction)
- **Phase 5**: Full GPT integration
- **Phase 6**: KV cache with GPU concepts

Total timeline: ~3 weeks

## References

- [PLAN.md](PLAN.md) - Detailed phase-by-phase plan
- [gpu_dialect.md](gpu_dialect.md) - Official MLIR GPU dialect documentation
- MLIR GPU Dialect: https://mlir.llvm.org/docs/Dialects/GPU/

## Notes for Future Transition to Real GPU

When GPU hardware becomes available, the transition is straightforward:
1. Keep the same algorithm structure
2. Switch from SCF loops to actual `gpu.launch`
3. Add GPU-specific lowering passes:
   - `gpu-kernel-outlining`
   - `convert-gpu-to-nvvm` (for NVIDIA)
   - `gpu-to-llvm`
4. Execute on actual GPU

The **concepts** learned here transfer directly!
