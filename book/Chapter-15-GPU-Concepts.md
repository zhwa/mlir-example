# Chapter 15: GPU Programming Concepts

Chapters 1-14 built a complete GPT architecture using MLIR's high-level abstractions—Chapters 11-14 used the **Transformer dialect** with operations that automatically bufferize to efficient code, while Chapter 14 added **Transform Dialect optimizations** (tiling, fusion, vectorization) for production-grade performance. The implementations are functionally correct and optimized for CPU, but modern LLM serving requires GPU acceleration—thousands of parallel threads, high memory bandwidth, specialized tensor cores. Chapter 15 introduces GPU programming fundamentals through MLIR by examining **real GPU dialect IR**, teaching the **low-level concepts** underlying production GPU code generation.

**Learning Through IR Inspection**. This chapter takes a unique pedagogical approach: generate GPU dialect IR and examine its structure rather than executing code. You'll write MLIR code that produces actual GPU kernels using `gpu.launch_func`, `gpu.thread_id`, and `gpu.barrier`, then inspect the generated IR to understand GPU programming concepts. This eliminates GPU hardware requirements—no NVIDIA or AMD GPU needed, just generate and read IR. The approach teaches real GPU dialect (not SCF loop emulation) showing the same IR structure that IREE, Torch-MLIR, and XLA generate for production GPUs. When you understand how `gpu.thread_id`, `gpu.barrier`, and `gpu.alloc` appear in IR, you understand GPU programming regardless of whether the target is NVIDIA CUDA, AMD ROCm, or Vulkan SPIR-V. The IR is the universal representation; hardware-specific details happen in later lowering passes.

**Connection to Previous Chapters**. Chapters 11-14 used high-level abstractions (Transformer dialect with tensor operations) that would **automatically target GPUs** in production compilers:

```
Chapter 14 tensor operations (linalg.matmul, linalg.generic)
  ↓ Transform Dialect (tile, fuse, vectorize)
  ↓ GPU Dialect (gpu.launch, gpu.thread_id) ← Chapter 15 teaches this level
  ↓ NVVM/ROCDL (GPU-specific LLVM)
  ↓ PTX/AMDGPU Assembly
```

Chapter 15 teaches what happens **inside** that GPU code generation—the thread patterns, memory access patterns, and reduction algorithms that automated compilers generate. You write high-level tensor code (Chapter 14), the compiler generates low-level GPU kernels (Chapter 15 patterns). Understanding both levels makes you effective at optimizing production ML systems.

**Three Representative Kernels**. This chapter implements three simplified GPU kernels. Vector addition demonstrates 1D parallelism with thread indexing and bounds checking. Matrix multiplication shows 2D parallelism using shared memory (`gpu.alloc`) and synchronization (`gpu.barrier`). Softmax illustrates reductions, multi-stage algorithms, and block cooperation with multiple barriers. These implementations are educational rather than production-quality—they show essential patterns without optimization complexity. Production kernels add tiling loops, bank conflict avoidance, and register blocking, but the core concepts remain identical.

## 15.1 GPU Architecture and Programming Model

Modern GPUs are massively parallel processors optimized for throughput. Understanding GPU architecture—thread organization, memory hierarchies, execution model—is essential for reading GPU dialect IR.

**Thread Hierarchy**. GPUs organize computation in a three-level hierarchy:

```
Grid (entire kernel launch)
  └─ Blocks (256-1024 threads each, independent units)
      └─ Threads (individual execution units within a block)
```

When processing 10,000 elements with 256 threads per block, you need a grid of 40 blocks (⌈10,000 / 256⌉ = 40), creating 10,240 thread slots total where 240 remain unused and are handled by bounds checking. Blocks execute independently in any order with no communication between them, while threads within each block cooperate through shared memory and synchronization barriers. This design scales naturally: a small GPU with 10 SMs runs 10 blocks concurrently, while a large GPU with 80 SMs can run all 40 concurrently.

**Thread Indexing**. Every thread computes its global index from block and thread coordinates:

```
1D indexing:
  globalIdx = blockIdx.x × blockDim.x + threadIdx.x
  
  Example: Block 5, Thread 123, blockDim 256
  → globalIdx = 5 × 256 + 123 = 1,403

2D indexing (for matrices):
  row = blockIdx.y × blockDim.y + threadIdx.y
  col = blockIdx.x × blockDim.x + threadIdx.x
  
  Example: 1024×1024 matrix, 16×16 thread blocks
  Block (5,10), Thread (8,12)
  → row = 88, col = 172
```

In MLIR GPU dialect, these appear as:
```mlir
%threadIdx = gpu.thread_id x : index
%blockIdx = gpu.block_id x : index
%blockDim = gpu.block_dim x : index
```

**Bounds Checking**. Grid dimensions rarely align exactly with data sizes. For 10,000 elements with 256-thread blocks, we launch 10,240 threads. The last 240 must skip computation:

```mlir
%globalIdx = ... // Computed from block/thread indices
%inBounds = arith.cmpi ult, %globalIdx, %N : index
scf.if %inBounds {
  // Only execute if within data bounds
}
```

Without bounds checking, threads access invalid memory—undefined behavior.

## 15.2 Memory Hierarchies

GPU memory is hierarchical: **global memory** (slow, large), **shared memory** (fast, small), and **registers** (fastest, smallest). Understanding this hierarchy is critical for reading GPU dialect IR.

GPUs have three memory levels with different performance characteristics. Global memory (HBM/GDDR) provides 16-80 GB capacity with ~400-800 cycle latency (can spike higher under contention or uncoalesced access) and 1-2 TB/s bandwidth, accessible to all threads and represented in MLIR as `memref<?xf32>` with default address space (technically Address Space 0). Shared memory offers 64-128 KB per SM with ~20-40 cycle latency and aggregate bandwidth often exceeding 15-20 TB/s across the chip (an order of magnitude faster than global memory), scoped to threads within the same block and allocated in MLIR via `gpu.alloc()` producing `memref<NxM xf32, 3>` where address space 3 indicates workgroup memory (this convention is target-dependent: NVVM and ROCDL both use address space 3 for shared memory). Registers provide ~256 KB per SM (32-64 registers per thread) with very low latency (2-4 cycles read-after-write, though this is typically hidden via warp switching, giving the appearance of "zero-cycle" access), private to each thread, and managed automatically by the compiler as SSA values rather than memory-mapped addresses.

**Shared Memory in MLIR**. Shared memory allocations appear as `gpu.alloc`:

```mlir
// Allocate 16×16 tile in shared memory
%tile = gpu.alloc() : memref<16x16xf32, 2>  // 2 = workgroup address space

// Load data cooperatively
memref.store %value, %tile[%row, %col] : memref<16x16xf32, 2>

// Synchronize all threads in block
gpu.barrier

// All threads can now read the tile
%shared_value = memref.load %tile[%i, %j] : memref<16x16xf32, 2>
```

The address space (2 = workgroup) tells the compiler to allocate in fast on-chip shared memory, not slow global memory. Note that address space 3 is the convention for NVVM (NVIDIA) and ROCDL (AMD) targets, but the logical MLIR GPU dialect often uses address space 2 for workgroup memory before lowering.

**Memory Coalescing**. Global memory transfers happen in 32-128 byte chunks. When threads in a warp access **consecutive addresses**, hardware coalesces loads into a single transaction:

```
Good (coalesced): Thread 0→data[0], Thread 1→data[1], ..., Thread 31→data[31]
  Result: 1 memory transaction (128 bytes: 32× float32)

Bad (uncoalesced): Thread 0→data[0], Thread 1→data[stride], ..., Thread 31→data[31×stride]  
  Result: 32 separate transactions (each 32 bytes)
  Slowdown: 32× slower!
```

In IR, coalescing depends on the **pattern of memref.load indices**. Sequential patterns coalesce; strided patterns don't.

## 15.3 Vector Addition: 1D Parallelism

Vector addition demonstrates the simplest GPU pattern: **1D parallelism** where each thread processes one element independently. This kernel teaches thread indexing, bounds checking, and `gpu.launch_func`.

**CPU vs GPU Approach**. On a CPU, you'd write a sequential loop iterating through indices and computing `C[i] = A[i] + B[i]` for each element. On a GPU, each thread computes one element by calculating its index from block and thread coordinates, then performing `C[idx] = A[idx] + B[idx]` independently and in parallel.

**IR Structure and Key Patterns**. The vector addition implementation in [gpu_kernels.cpp](ch.15.GPU-Concepts/src/gpu_kernels.cpp) generates IR with two distinct parts: the kernel definition and the host launch code. The kernel definition starts by creating a `gpu.module` containing a `gpu.func` marked with the kernel attribute. This separation is fundamental—GPU code lives in `gpu.module`, host code in regular `func.func`.

Inside the kernel, the first step is obtaining thread coordinates from hardware. Three operations provide this information:

```mlir
%threadIdx = gpu.thread_id x : index    // Position within block (0-255)
%blockIdx = gpu.block_id x : index      // Which block this is
%blockDim = gpu.block_dim x : index     // Block size (256)
```

These values come from GPU hardware, not from loops. Each thread automatically knows its position. Computing the global index requires combining these coordinates following the standard pattern `globalIdx = blockIdx × blockDim + threadIdx`:

```mlir
%blockOffset = arith.muli %blockIdx, %blockDim : index
%globalIdx = arith.addi %blockOffset, %threadIdx : index
```

For a thread in block 5 at position 123 with block size 256, this computes 5 × 256 + 123 = 1,403.

**Bounds Checking Pattern**. Since grid dimensions rarely align exactly with array size, threads must check whether their computed index is valid before accessing memory:

```mlir
%inBounds = arith.cmpi ult, %globalIdx, %N : index
scf.if %inBounds {
  %a = memref.load %A[%globalIdx] : memref<?xf32>
  %b = memref.load %B[%globalIdx] : memref<?xf32>
  %sum = arith.addf %a, %b : f32
  memref.store %sum, %C[%globalIdx] : memref<?xf32>
}
```

Without this check, threads with indices beyond array bounds would access invalid memory. For 10,000 elements with 256-thread blocks, you launch 10,240 threads—the last 240 skip computation via this bounds check.

**Host Launch Code**. Separately from the kernel, the host code calculates grid dimensions and launches the kernel:

```mlir
func.func @main(%A: memref<?xf32>, %B: memref<?xf32>, 
                %C: memref<?xf32>, %N: index) {
  // Grid size: ceil(N / 256)
  %c255 = arith.constant 255 : index
  %temp = arith.addi %N, %c255 : index
  %numBlocks = arith.divui %temp, %c256 : index
  
  // Launch with 1D grid and 1D blocks
  gpu.launch_func @kernels::@vector_add
      blocks in (%numBlocks, %c1, %c1)      // Grid dimensions
      threads in (%c256, %c1, %c1)          // Block dimensions
      args(...)
}
```

The `blocks in` clause specifies grid dimensions (how many blocks), while `threads in` specifies block dimensions (how many threads per block). This example uses 1D organization (only the x dimension is non-1). The ceiling division `(N + 255) / 256` ensures enough threads to cover all elements.

**Lowering to Hardware**. This IR lowers to actual GPU code:

```
GPU Dialect → NVVM (NVIDIA) → PTX → CUDA Binary
            → ROCDL (AMD) → AMDGPU → ROCm Binary  
            → SPIRV (Vulkan) → SPIR-V Bytecode
```

All three targets understand the same GPU dialect concepts—thread indexing, bounds checking, memory access. The IR is universal; hardware-specific details emerge only in the final lowering stage.

## 15.4 Matrix Multiplication: 2D Parallelism and Shared Memory

Matrix multiplication demonstrates **2D parallelism**, **shared memory** (`gpu.alloc`), and **synchronization** (`gpu.barrier`). This kernel teaches the tiled algorithm pattern used in all production matrix multiplication kernels.

**Simplified Implementation**. Our educational implementation demonstrates the essential pattern: allocate shared memory tiles (16×16) in workgroup address space, load data from global memory into shared memory, synchronize with `gpu.barrier` to ensure all threads finished loading, read from the now-populated shared memory (which is much faster), store results, then synchronize again before the next iteration. Production implementations add tiling loops to process multiple tiles and accumulate results, but our version shows one tile to focus on shared memory and synchronization concepts.

**Key Implementation Points**:

The C++ code in [gpu_kernels.cpp] shows:

1. **2D thread organization**:
```cpp
Value threadX = builder.create<gpu::ThreadIdOp>(loc, indexType, gpu::Dimension::x);
Value threadY = builder.create<gpu::ThreadIdOp>(loc, indexType, gpu::Dimension::y);
```

2. **Shared memory allocation with address space**:
```cpp
auto addrSpace = builder.getI64IntegerAttr(
    static_cast<int64_t>(gpu::AddressSpace::Workgroup));
auto tileType = MemRefType::get({16, 16}, f32Type, 
    MemRefLayoutAttrInterface{}, addrSpace);
Value tileA = builder.create<gpu::AllocOp>(loc, tileType, ...);
```

3. **Cooperative loading pattern**:
```cpp
// Each thread loads one element: global → shared
builder.create<memref::StoreOp>(loc, aVal, tileA, ValueRange{threadY, threadX});
builder.create<gpu::BarrierOp>(loc);  // Synchronize
// All threads can now read from shared memory
```

**Generated IR Pattern** (key structure):

```mlir
gpu.module @kernels {
  gpu.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, 
                   %C: memref<?x?xf32>) kernel {
    // 2D thread indices
    %tx = gpu.thread_id x : index
    %ty = gpu.thread_id y : index
    %bx = gpu.block_id x : index
    %by = gpu.block_id y : index
    
    // Allocate shared memory (address space 2 = workgroup)
    %tileA = gpu.alloc() : memref<16x16xf32, 2>
    %tileB = gpu.alloc() : memref<16x16xf32, 2>
    
    // Compute global indices: row = blockY * 16 + threadY
    %row = arith.addi %blockOffsetY, %ty : index
    %col = arith.addi %blockOffsetX, %tx : index
    
    // Cooperative loading: global → shared
    %aVal = memref.load %A[%row, %col] : memref<?x?xf32>
    memref.store %aVal, %tileA[%ty, %tx] : memref<16x16xf32, 2>
    
    gpu.barrier  // Wait for all threads to finish loading
    
    // Read from shared memory (50-100× faster!)
    %tileVal = memref.load %tileA[%ty, %tx] : memref<16x16xf32, 2>
    memref.store %tileVal, %C[%row, %col] : memref<?x?xf32>
    
    gpu.barrier  // Synchronize before next iteration
    gpu.return
  }
}

func.func @main(...) {
  // Launch: 2D grid, 2D blocks (16×16 threads per block)
  gpu.launch_func @kernels::@matmul
      blocks in (%gridX, %gridY, %c1)     // 2D grid
      threads in (%c16, %c16, %c1)        // 16×16 thread blocks
      args(...)
}
```

The matrix multiplication kernel uses 2D indexing for both grid and block organization (`blocks in (gridX, gridY, 1)` and `threads in (16, 16, 1)`). Shared memory appears as `gpu.alloc() : memref<16x16xf32, 2>` where address space 2 indicates workgroup memory. The IR shows the memory hierarchy through type annotations: global loads use `memref<?x?xf32>` while shared memory uses `memref<16x16xf32, 2>`. The `gpu.barrier` operation ensures all threads finish loading data before any thread reads from shared memory, enabling the cooperative pattern where all 256 threads (16×16) load one element each into the shared tile. Shared memory provides significant speedup over global memory (~20-40 cycle latency versus ~400-800 cycles, roughly 10-20× faster), which is why production kernels load tiles into shared memory and reuse data multiple times to dramatically reduce global memory traffic.

## 15.5 Softmax: Reductions and Block Cooperation

Softmax demonstrates **reduction operations**, **multi-stage algorithms**, and **block cooperation** with multiple barriers. This kernel teaches the pattern used in all neural network operations requiring cross-element communication.

**Note**: Our implementation shows a **simplified reduction pattern** for educational purposes. Production softmax kernels use more sophisticated tree reductions and warp-level primitives.

**Softmax Algorithm**. For numerical stability, softmax uses a three-stage algorithm:

```
softmax(x)[i] = exp(x[i] - max(x)) / Σ exp(x[j] - max(x))
                                      j

Stage 1: Find max(x) across all elements
Stage 2: Compute exp(x[i] - max) and sum exponentials
Stage 3: Normalize by dividing by sum
```

**Parallel Reduction Pattern**. Serial CPU code finds the maximum by initializing with the first element then iterating through remaining elements, updating the max as it goes. On a GPU with tree reduction, each thread first computes a local maximum over its assigned subset of elements, then threads cooperate in shared memory to combine partial results, using `gpu.barrier` operations between stages to ensure correctness.

**Key Implementation Points**:

The C++ code in [gpu_kernels.cpp](ch.15.GPU-Concepts/src/gpu_kernels.cpp) demonstrates:

1. **Allocating shared memory for reductions**:
```cpp
auto addrSpace = builder.getI64IntegerAttr(
    static_cast<int64_t>(gpu::AddressSpace::Workgroup));
Value sharedMax = builder.create<gpu::AllocOp>(loc, sharedType, ...);
Value sharedSum = builder.create<gpu::AllocOp>(loc, sharedType, ...);
```

2. **Four-barrier pattern**:
```cpp
// Compute local results → Store to shared memory
builder.create<gpu::BarrierOp>(loc);  // Barrier 1
// Thread 0 reduces shared memory → Stores global result
builder.create<gpu::BarrierOp>(loc);  // Barrier 2
// All threads read global result → Compute next stage
// ... repeat for sum reduction ...
```

**Generated IR Pattern** (key structure):

```mlir
gpu.func @softmax(...) kernel {
  %shared_max = gpu.alloc() : memref<256xf32, 2>
  %shared_sum = gpu.alloc() : memref<256xf32, 2>
  
  // Each thread: compute local max → store
  memref.store %local_max, %shared_max[%tx]
  gpu.barrier  // Wait for all threads
  
  // Thread 0: reduce to global max (simplified in example)
  // In production: tree reduction loop here
  gpu.barrier  // Wait for thread 0
  
  // All threads: read global_max, compute exp and local sum
  %global_max = memref.load %shared_max[%c0]
  memref.store %local_sum, %shared_sum[%tx]
  gpu.barrier  // Wait for all threads
  
  // Thread 0: reduce to global sum (simplified in example)
  // In production: tree reduction loop here
  gpu.barrier  // Wait for thread 0
  
  // All threads: normalize using global_sum
  %global_sum = memref.load %shared_sum[%c0]
  // ... normalization ...
}
```

The softmax kernel uses four barriers positioned after computing the local max, after computing the global max, after computing local sums, and after computing the global sum. Shared memory allocated via `gpu.alloc() : memref<256xf32, 2>` enables thread communication, where address space 2 indicates workgroup memory accessible to all threads in the block. Each stage of the multi-stage algorithm depends on the previous barrier completing. The cooperative pattern follows a consistent rhythm: all threads contribute partial results, one thread (typically thread 0) performs the reduction, then all threads read the final result.

**Why Barriers Matter**. Without barriers, race conditions occur. For example, if Thread 0 begins the reduction phase before Thread 255 has finished writing its local maximum to `shared_max[255]`, Thread 0 will read stale or uninitialized data, leading to an incorrect global maximum:

```
Thread 0: starts reading shared_max[] for reduction
Thread 255: still writing shared_max[255] = 8.3 → Race condition!
Result: Thread 0 computes wrong maximum
```

`gpu.barrier` ensures **all threads** reach the barrier before **any thread** proceeds. This guarantees:
- All writes to shared memory before barrier are visible to all threads after barrier
- No thread can race ahead and read stale data

**Production Pattern**. Softmax's multi-stage reduction with barriers is the canonical GPU pattern for:
- Layer normalization (mean/variance reductions)
- Cross-entropy loss (log-sum-exp reduction)
- Attention scores (softmax over sequence length)
- Batch normalization (statistics over batch)

All follow: compute locally → barrier → reduce cooperatively → barrier → use result.

## 15.6 Running the Examples

Build and run from the repository root:

```bash
python3 ch.15.GPU-Concepts/test_gpu.py
```

The test outputs GPU dialect IR for all three kernels with annotated explanations. You'll see the actual IR structure showing `gpu.thread_id`, `gpu.alloc`, `gpu.barrier`, and `gpu.launch_func` operations. The output demonstrates how thread indexing, shared memory, and synchronization appear in practice.

When inspecting the IR, the lowering path is: GPU dialect → NVVM/ROCDL/SPIRV → PTX/AMDGPU/SPIR-V → hardware machine code.

## 15.7 Summary

This chapter demonstrated GPU programming fundamentals through IR generation. The three kernels (vector addition, matrix multiplication, softmax) show 1D/2D parallelism, shared memory usage, and multi-barrier synchronization patterns.

The GPU dialect operations you generated—`gpu.thread_id`, `gpu.block_id`, `gpu.alloc`, `gpu.barrier`, `gpu.launch_func`—are identical to what production ML compilers emit. When PyTorch compiles `F.softmax(x)` or TensorFlow compiles a matmul, the underlying IR uses these same patterns. Production systems automate the generation; understanding the IR helps you diagnose performance and write custom optimizations.

GPU programming centers on parallelism (thousands of threads), memory hierarchy (global→shared→registers), and synchronization (barriers). These concepts are universal across CUDA, HIP, Metal, and MLIR GPU dialect.