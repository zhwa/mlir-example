# Chapter 15: GPU Programming with MLIR

This chapter teaches GPU programming fundamentals through MLIR's GPU dialect. We generate and examine GPU IR to understand how real GPU programming works - without executing any code or requiring GPU hardware.

## Three Representative GPU Patterns:

1. **Vector Addition** - Basic 1D GPU parallelism
   - `gpu.launch_func` for kernel launches
   - `gpu.thread_id`, `gpu.block_id`, `gpu.block_dim` for thread indexing
   - Global index calculation: `blockIdx * blockDim + threadIdx`
   - Bounds checking for safety

2. **Matrix Multiplication** - 2D parallelism + shared memory
   - 2D thread blocks and grids
   - `gpu.alloc` for workgroup (shared) memory allocation
   - `gpu.barrier` for thread synchronization
   - Tiled computation pattern

3. **Softmax** - Reductions and multi-stage algorithms
   - Block-level cooperation
   - Multiple `gpu.barrier` calls for stage separation
   - Reduction patterns (max, sum)
   - Shared memory for intermediate results