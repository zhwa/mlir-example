# Chapter 15: GPU Programming Concepts

Chapters 1-14 built a complete GPT architecture using MLIR's high-level abstractions—Chapters 11-14 used the **Transformer dialect with tensor-first operations** that automatically bufferize to efficient code, while Chapter 14 added **Transform Dialect optimizations** (tiling, fusion, vectorization) for production-grade performance. The implementations are functionally correct and optimized for CPU, but modern LLM serving requires GPU acceleration—thousands of parallel threads, high memory bandwidth, specialized tensor cores. Chapter 15 introduces GPU programming fundamentals through MLIR, teaching the **low-level concepts** underlying production GPU code generation.

**Two Levels of GPU Programming**. This chapter takes a unique pedagogical approach distinct from Chapters 11-14:

**Chapters 11-14 (High-level)**: Tensor operations → Linalg → Transform Dialect → (would target GPU dialect) → GPU code. This is how production ML compilers (IREE, Torch-MLIR, XLA) work—you write high-level tensor operations, the compiler automatically generates optimized GPU kernels.

**Chapter 15 (Low-level)**: Direct memref manipulation with explicit thread indexing. We implement GPU concepts on CPU through **thread emulation** using MLIR's SCF (Structured Control Flow) dialect to emulate GPU thread hierarchies (grids, blocks, threads) as nested loops. This approach teaches the **hardware concepts and patterns** that automated compilers generate behind the scenes.

This chapter eliminates GPU hardware requirements while teaching genuine GPU programming patterns—the same concepts used in CUDA, ROCm, and Metal. When you understand thread indexing, memory coalescing, and reduction algorithms through CPU emulation, you understand them period. The transition to real GPU code is syntactic, not conceptual. **Educational value**: See what Transform Dialect and production compilers do automatically when targeting GPUs.

**Connection to Previous Chapters**. Chapters 11-14 used high-level abstractions (Transformer dialect with tensor operations) that would **automatically target GPUs** in production compilers:

```
Chapter 14 tensor operations (linalg.matmul, linalg.generic)
  ↓ Transform Dialect (tile, fuse, vectorize)
  ↓ GPU Dialect (gpu.launch, gpu.thread_id) ← automatic code generation
  ↓ NVVM/ROCDL (GPU-specific LLVM)
  ↓ PTX/AMDGPU Assembly
```

Chapter 15 teaches what happens **inside** that GPU code generation—the thread patterns, memory access patterns, and reduction algorithms that automated compilers generate. You write high-level tensor code (Chapter 14), the compiler generates low-level GPU kernels (Chapter 15 patterns). Understanding both levels makes you effective at optimizing production ML systems.

**Why Memref Instead of Tensors?** This chapter uses **memref directly** (raw memory with explicit indexing) rather than tensor operations because:
1. **Teaching GPU concepts**: Thread indexing, bounds checking, memory coalescing are explicit and visible
2. **CPU emulation**: Nested loops directly map to GPU thread hierarchies
3. **Hardware focus**: See the patterns that Transform Dialect and GPU dialect generate automatically

Production ML compilers use the Chapter 14 approach (high-level tensors) for portability and optimization. Chapter 15 teaches the low-level patterns they generate.

**Why GPU Programming Matters for LLMs**. GPT-3 has 175 billion parameters (700 GB at float32). A single forward pass touches every parameter—reading 700 GB of data. CPU memory bandwidth: 50-100 GB/s (7-14 seconds per forward pass). GPU memory bandwidth: 1-2 TB/s (0.35-0.7 seconds per forward pass). This 10-20× bandwidth advantage makes GPUs essential for LLM inference. Understanding GPU architecture—memory hierarchies, thread organization, access patterns—is prerequisite for building production serving systems.

Chapter 15 is structured as seven progressive phases: vector operations (1D parallelism), matrix multiplication (2D parallelism), element-wise operations (GELU, bias), softmax (reductions), layer normalization (multi-stage reductions), transpose (memory patterns), attention mechanism (combining primitives), and complete transformer (nano GPT with KV cache). Each phase introduces new GPU concepts while building toward a working transformer. By chapter's end, you'll have a complete nano GPT implementation that demonstrates production GPU patterns—all running on CPU for universal accessibility.

**A Note on AOT vs JIT Compilation**. Chapters 1-14 used **JIT-style workflow** with Python bindings—build IR at runtime, compile to LLVM, execute via ExecutionEngine, return results to Python. This enables rapid prototyping and testing. Chapter 15 switches to **AOT (Ahead-Of-Time) workflow**—generate MLIR at build time, compile to object files, link into standalone C++ test binary. This change reflects production reality: serving systems compile models once, execute many times. AOT workflow matches production compilers (IREE, XLA, TVM) and enables better debugging (inspect assembly, use gdb, profile with perf). The trade-off is losing Python integration, but gaining reliability and matching production deployment patterns. Both approaches use MLIR compilation; the difference is **when** and **how** code generation happens.

**Note**: The original development used JIT but encountered LLVM 21 ORC JIT bugs with LayerNorm operations. Switching to AOT resolved these issues—demonstrating why production systems prefer AOT compilation for reliability.

## 15.1 GPU Architecture Fundamentals

Modern GPUs are massively parallel processors optimized for throughput, not latency. Understanding GPU architecture—how it differs from CPUs, why it excels at ML workloads—is essential for effective GPU programming.

**CPU vs GPU Design Philosophy**. CPUs optimize for serial execution: large caches (32 MB L3), complex control logic (branch prediction, out-of-order execution, speculative execution), few powerful cores (8-64 cores). A CPU core is a Formula 1 race car—sophisticated, fast, expensive. CPUs excel at irregular workloads with unpredictable branches and data dependencies.

GPUs optimize for parallel throughput: small caches per core (128 KB shared across 64 threads), simple control logic (no branch prediction, in-order execution), thousands of simple cores (2,000-10,000 streaming multiprocessors). A GPU core is a bicycle—simple, efficient, cheap. GPUs excel at regular workloads with predictable patterns and massive parallelism.

**Memory Bandwidth vs Compute**. Modern GPUs (NVIDIA A100, H100) provide 1-2 TB/s memory bandwidth—10-20× faster than CPU DRAM (50-100 GB/s). This bandwidth advantage is critical for LLM inference:

```
GPT-3 forward pass (175B parameters, float32):
  Data volume: 175B × 4 bytes = 700 GB
  
CPU (100 GB/s bandwidth):
  Time: 700 GB / 100 GB/s = 7 seconds
  
GPU (1.5 TB/s bandwidth):
  Time: 700 GB / 1500 GB/s = 0.47 seconds
  
Speedup: 7 / 0.47 ≈ 15×
```

For large models, **memory bandwidth dominates compute**. GPUs provide both higher bandwidth (20×) and higher compute (100×), but bandwidth is often the bottleneck. This is why techniques like KV caching (Chapter 14) and paged attention (Chapter 16) are crucial—they reduce memory transfers, not compute.

**SIMT Execution Model**. GPUs use SIMT (Single Instruction, Multiple Threads): one instruction stream controls multiple threads executing the same operation on different data. This is conceptually similar to SIMD (Single Instruction, Multiple Data) but more flexible:

- **SIMD (CPU)**: One instruction, 8-16 data elements (AVX-512: 16× float32)
- **SIMT (GPU)**: One instruction, 32-64 threads (NVIDIA warp: 32 threads, AMD wavefront: 64 threads)

SIMT provides programmer flexibility: threads can diverge (different control flow paths), though performance suffers when they do. The GPU hardware groups threads into **warps** (NVIDIA) or **wavefronts** (AMD)—typically 32 threads executing lockstep. When all threads in a warp follow the same path, hardware is fully utilized. When threads diverge (if-else branches), hardware serializes execution—a performance penalty.

**Thread Hierarchy: Grid, Block, Thread**. GPUs organize computation in a three-level hierarchy:

```
Grid (entire kernel launch)
  └─ Blocks (256-1024 threads each, fixed size)
      └─ Threads (individual execution units)
```

**Example: Processing 10,000 elements with 256 threads per block**:
- Need ⌈10,000 / 256⌉ = 40 blocks
- Each block has 256 threads
- Total: 40 × 256 = 10,240 thread slots (240 unused)
- Grid dimensions: 1D grid of 40 blocks

Blocks are **independent** and can execute in any order—this enables scalability across different GPU sizes. A GPU with 10 SMs (Streaming Multiprocessors) executes 10 blocks concurrently; a GPU with 80 SMs executes all 40 blocks concurrently. Threads within a block can **cooperate** (shared memory, synchronization); threads in different blocks cannot.

**Index Calculation**. Every thread computes its global index from grid and block coordinates:

```cpp
// 1D indexing
globalIdx = blockIdx.x * blockDim.x + threadIdx.x

// Example: Block 5, Thread 123, blockDim 256
globalIdx = 5 * 256 + 123 = 1,403
```

For 2D operations (matrices), use 2D indexing:

```cpp
// 2D indexing
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x

// Example: Process 1024×1024 matrix with 16×16 thread blocks
// Need 64×64 blocks (1024/16 = 64 in each dimension)
// Thread at block (5,10), local position (8,12):
row = 5 * 16 + 8 = 88
col = 10 * 16 + 12 = 172
// This thread processes element [88, 172]
```

**Bounds Checking**. Grid dimensions don't always align with data sizes. For 10,000 elements with 256-thread blocks, we launch 10,240 threads (40 × 256). The last 240 threads must skip computation:

```cpp
int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

if (globalIdx < N) {  // Critical bounds check!
    output[globalIdx] = process(input[globalIdx]);
}
```

Without bounds checking, threads access invalid memory—crashes, corruption, or silent errors. Always include bounds checks unless grid dimensions exactly match data dimensions (rare in practice).

## 15.2 Memory Hierarchies

GPU memory organization is hierarchical: registers (fastest, smallest), shared memory (fast, small), global memory (slow, large). Understanding this hierarchy and using it correctly is critical for performance.

**Memory Hierarchy Overview**:

```
Registers (per-thread storage)
  Size: ~256 KB per SM (32-64 registers per thread)
  Latency: 0 cycles (immediate access)
  Scope: Private to thread
  
Shared Memory (per-block storage)
  Size: 64-128 KB per SM (shared across block threads)
  Latency: ~5-10 cycles
  Scope: Shared within block, requires synchronization
  
L1/L2 Cache (hardware-managed)
  Size: 128 KB L1 per SM, 40-60 MB L2 shared
  Latency: ~20-100 cycles
  Scope: Hardware-managed, programmer-transparent
  
Global Memory (main GPU memory)
  Size: 16-80 GB (HBM2/HBM3)
  Latency: ~300-500 cycles
  Bandwidth: 1-2 TB/s
  Scope: Accessible by all threads
```

**Latency Comparison**. Consider a simple operation: `c = a + b`:

- **Registers**: 0 cycles if `a`, `b`, `c` already in registers
- **Shared memory**: ~5-10 cycles to load from shared memory
- **Global memory**: ~300-500 cycles to load from DRAM

This 50-100× latency difference makes memory access patterns critical. A kernel spending 90% of time waiting for memory loads is memory-bound, not compute-bound—optimizing compute does nothing.

**Memory Coalescing**. Global memory transfers happen in 32-byte or 128-byte chunks. When threads in a warp access consecutive memory addresses, the hardware **coalesces** loads into a single transaction:

```cpp
// Good: Coalesced access (consecutive addresses)
__global__ void coalesced_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];  // Thread 0 reads data[0], thread 1 reads data[1], ...
        // Process val
    }
}

// Bad: Strided access (non-consecutive addresses)
__global__ void strided_kernel(float* data, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx * stride];  // Thread 0 reads data[0], thread 1 reads data[stride], ...
        // Process val (many separate memory transactions!)
    }
}
```

**Coalesced access** (consecutive addresses): 32 threads load 32 consecutive float32 values (128 bytes) in 1 transaction.

**Strided access** (e.g., stride=32): 32 threads load values 128 bytes apart, requiring 32 separate transactions—32× slower!

For matrices, row-major layout enables coalescing for row-wise access but not column-wise access. This is why transpose operations are expensive—they convert coalesced reads to uncoalesced writes (or vice versa).

**Shared Memory for Data Reuse**. Global memory is expensive (500 cycles). If multiple threads need the same data, load it once into shared memory:

```cpp
__shared__ float sharedData[256];  // Shared across block

// Load data cooperatively
sharedData[threadIdx.x] = globalData[blockIdx.x * 256 + threadIdx.x];
__syncthreads();  // Ensure all threads finished loading

// All threads can now access sharedData with ~5 cycle latency
float value = sharedData[someIndex];
```

Matrix multiplication uses shared memory extensively: load a tile of A and B into shared memory, compute partial results (many reuses), then proceed to next tile. This reduces global memory traffic by 10-100×.

**Register Pressure**. Each thread has limited registers (32-64 registers per thread on modern GPUs). Using too many variables spills registers to local memory (stored in global memory)—destroying performance:

```cpp
// High register pressure (many variables)
float a1, a2, a3, ..., a64;  // 64 float32 variables = 64 registers
// If GPU supports 32 registers/thread, 32 variables spill to slow memory!
```

Compilers try to minimize register usage, but complex kernels naturally require many temporaries. Sometimes reducing parallelism (fewer threads per block) allows more registers per thread, improving performance despite less parallelism.

## 15.3 Thread Organization and Warps

GPU threads are not independent—they execute in groups called warps (NVIDIA) or wavefronts (AMD). Understanding warp execution is essential for writing efficient GPU code.

**Warp Execution Model**. A **warp** is a group of 32 threads (NVIDIA) or 64 threads (AMD) that execute the same instruction simultaneously. Hardware schedules warps, not individual threads. Within a warp:

- All threads execute the same instruction (SIMT: Single Instruction, Multiple Threads)
- Threads can read/write different data (different register values)
- Threads can follow different control flow paths (divergence)

**Example: 256-thread block**:
- NVIDIA GPU: 8 warps (256 / 32 = 8)
- AMD GPU: 4 wavefronts (256 / 64 = 4)

Each warp executes independently. The GPU schedules warps to hide memory latency—while one warp waits for memory, another warp computes.

**Warp Divergence**. When threads in a warp follow different control flow paths, hardware **serializes** execution:

```cpp
// Divergent code (poor performance)
if (threadIdx.x < 16) {
    // Path A: threads 0-15
    expensiveOperationA();
} else {
    // Path B: threads 16-31
    expensiveOperationB();
}
```

Execution timeline for warp:
1. Threads 0-15 execute Path A, threads 16-31 are idle (masked)
2. Threads 16-31 execute Path B, threads 0-15 are idle (masked)
3. Total time: time(A) + time(B)

If both paths take 100 cycles, divergence costs 200 cycles instead of 100 cycles. The warp is only 50% utilized.

**Minimizing Divergence**. Organize data and algorithms to avoid divergence:

```cpp
// Good: All threads follow same path
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {  // Entire warps are in-bounds or out-of-bounds (mostly)
    output[idx] = process(input[idx]);
}
```

When block size (256) is a multiple of warp size (32), entire warps are either fully in-bounds or fully out-of-bounds. Only the last partial warp has divergence—minimal impact.

**Warp-Level Primitives**. Modern GPUs provide warp-level operations for efficient communication:

```cpp
// Warp shuffle: exchange data between threads in a warp (no memory access!)
float value = __shfl_xor_sync(0xFFFFFFFF, myValue, 1);  // Swap with thread XOR 1

// Warp reduction: sum all values in a warp
float sum = __reduce_add_sync(0xFFFFFFFF, myValue);
```

These operations have ~1-2 cycle latency—much faster than shared memory (~5-10 cycles) or global memory (~500 cycles). Modern reduction algorithms use warp shuffles extensively.

**Occupancy**. Occupancy is the ratio of active warps to maximum possible warps:

```
Occupancy = (active warps per SM) / (maximum warps per SM)
```

Higher occupancy provides more warps to hide memory latency. If one warp stalls on memory, the SM switches to another warp—maintaining high throughput. Factors limiting occupancy:

- **Registers per thread**: Using 64 registers/thread allows fewer threads per SM
- **Shared memory per block**: Using 48 KB shared memory allows fewer blocks per SM
- **Thread block size**: Small blocks (64 threads) leave SM underutilized

Typical target: 50-75% occupancy. Higher isn't always better—sometimes reducing occupancy (more registers per thread) improves performance by reducing memory traffic.

## 15.4 Emulating GPU Threads on CPU

MLIR's SCF dialect enables GPU-style programming on CPU through nested loops that emulate thread hierarchies. This section shows the translation from GPU concepts to MLIR IR.

**GPU Kernel Structure**. A typical GPU kernel has this structure:

```cuda
__global__ void vector_add(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Launch: <<<numBlocks, threadsPerBlock>>>
dim3 numBlocks((N + 255) / 256);
dim3 threadsPerBlock(256);
vector_add<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
```

**MLIR Emulation with Nested Loops**. Translate grid/block/thread hierarchy to nested `scf.for` loops:

```cpp
// C++ API: Build kernel IR
void buildVectorAddKernel(OpBuilder& builder, Location loc,
                          Value A, Value B, Value C, Value N) {
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value c256 = builder.create<arith::ConstantIndexOp>(loc, 256);
    Value c255 = builder.create<arith::ConstantIndexOp>(loc, 255);
    
    // Grid size: ceil(N / 256)
    Value N_plus_255 = builder.create<arith::AddIOp>(loc, N, c255);
    Value numBlocks = builder.create<arith::DivUIOp>(loc, N_plus_255, c256);
    
    // Outer loop: blocks (emulate GPU grid)
    auto blockLoop = builder.create<scf::ForOp>(loc, c0, numBlocks, c1);
    builder.setInsertionPointToStart(blockLoop.getBody());
    Value blockIdx = blockLoop.getInductionVar();
    
    // Inner loop: threads (emulate GPU block)
    auto threadLoop = builder.create<scf::ForOp>(loc, c0, c256, c1);
    builder.setInsertionPointToStart(threadLoop.getBody());
    Value threadIdx = threadLoop.getInductionVar();
    
    // Compute global index
    Value blockOffset = builder.create<arith::MulIOp>(loc, blockIdx, c256);
    Value globalIdx = builder.create<arith::AddIOp>(loc, blockOffset, threadIdx);
    
    // Bounds check
    Value inBounds = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalIdx, N
    );
    
    auto ifOp = builder.create<scf::IfOp>(loc, inBounds, false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    
    // Kernel body: C[idx] = A[idx] + B[idx]
    Value a = builder.create<memref::LoadOp>(loc, A, globalIdx);
    Value b = builder.create<memref::LoadOp>(loc, B, globalIdx);
    Value sum = builder.create<arith::AddFOp>(loc, a, b);
    builder.create<memref::StoreOp>(loc, sum, C, globalIdx);
}
```

**Generated MLIR IR**:

```mlir
func.func @vector_add(%A: memref<?xf32>, %B: memref<?xf32>, 
                      %C: memref<?xf32>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %c255 = arith.constant 255 : index
  
  // Grid size
  %temp = arith.addi %N, %c255 : index
  %numBlocks = arith.divui %temp, %c256 : index
  
  // Block loop (GPU grid)
  scf.for %blockIdx = %c0 to %numBlocks step %c1 {
    // Thread loop (GPU block)
    scf.for %threadIdx = %c0 to %c256 step %c1 {
      // Global index
      %blockOffset = arith.muli %blockIdx, %c256 : index
      %globalIdx = arith.addi %blockOffset, %threadIdx : index
      
      // Bounds check
      %inBounds = arith.cmpi ult, %globalIdx, %N : index
      
      scf.if %inBounds {
        %a = memref.load %A[%globalIdx] : memref<?xf32>
        %b = memref.load %B[%globalIdx] : memref<?xf32>
        %c = arith.addf %a, %b : f32
        memref.store %c, %C[%globalIdx] : memref<?xf32>
      }
    }
  }
  return
}
```

**Why This Works**. The nested loop structure precisely mirrors GPU execution:

- **Outer loop** (`blockIdx`): Represents independent blocks (can execute in any order)
- **Inner loop** (`threadIdx`): Represents threads within a block (execute serially on CPU, parallel on GPU)
- **Index calculation**: Identical to GPU (`blockIdx * blockDim + threadIdx`)
- **Bounds check**: Required for both CPU and GPU

On CPU, loops execute serially (block 0, then block 1, ...). On GPU, blocks execute in parallel. The **semantics are identical**—only execution model differs. This is why CPU emulation teaches real GPU programming.

**2D Thread Organization**. For matrix operations, use 4 nested loops (2D grid × 2D block):

```cpp
// Grid dimensions: ceil(M/16) × ceil(N/16)
Value c16 = builder.create<arith::ConstantIndexOp>(loc, 16);
Value c15 = builder.create<arith::ConstantIndexOp>(loc, 15);

Value gridDimX = builder.create<arith::DivUIOp>(loc,
    builder.create<arith::AddIOp>(loc, M, c15), c16);
Value gridDimY = builder.create<arith::DivUIOp>(loc,
    builder.create<arith::AddIOp>(loc, N, c15), c16);

// Outer loops: 2D grid
auto blockLoopX = builder.create<scf::ForOp>(loc, c0, gridDimX, c1);
builder.setInsertionPointToStart(blockLoopX.getBody());
Value blockIdxX = blockLoopX.getInductionVar();

auto blockLoopY = builder.create<scf::ForOp>(loc, c0, gridDimY, c1);
builder.setInsertionPointToStart(blockLoopY.getBody());
Value blockIdxY = blockLoopY.getInductionVar();

// Inner loops: 2D block (16×16 threads)
auto threadLoopX = builder.create<scf::ForOp>(loc, c0, c16, c1);
builder.setInsertionPointToStart(threadLoopX.getBody());
Value threadIdxX = threadLoopX.getInductionVar();

auto threadLoopY = builder.create<scf::ForOp>(loc, c0, c16, c1);
builder.setInsertionPointToStart(threadLoopY.getBody());
Value threadIdxY = threadLoopY.getInductionVar();

// Compute global indices
Value row = builder.create<arith::AddIOp>(loc,
    builder.create<arith::MulIOp>(loc, blockIdxX, c16), threadIdxX);
Value col = builder.create<arith::AddIOp>(loc,
    builder.create<arith::MulIOp>(loc, blockIdxY, c16), threadIdxY);
```

This 4-level nesting (block X, block Y, thread X, thread Y) emulates GPU's 2D thread organization. Each thread processes one matrix element at position `(row, col)`.

## 15.5 Basic Operations: Vector and Matrix

With GPU concepts established, we implement fundamental operations: vector addition (1D parallelism) and matrix multiplication (2D parallelism). These operations form building blocks for neural networks.

**Vector Addition: 1D Parallelism**. Vector addition is the simplest GPU operation—embarrassingly parallel with no data dependencies:

```
C[i] = A[i] + B[i]  for i = 0, 1, ..., N-1
```

Each element is independent. With 256-thread blocks:

```
Threads:  |0|1|2|...|255|256|257|...|511|...
          └─ Block 0 ─┘└─── Block 1 ────┘
          
Thread 0:   C[0] = A[0] + B[0]
Thread 1:   C[1] = A[1] + B[1]
...
Thread 256: C[256] = A[256] + B[256]
```

**Implementation Pattern**:

```cpp
void buildVectorAddKernel(OpBuilder& builder, Location loc,
                          Value A, Value B, Value C, Value N) {
    // Constants
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value c256 = builder.create<arith::ConstantIndexOp>(loc, 256);
    Value c255 = builder.create<arith::ConstantIndexOp>(loc, 255);
    
    // Grid size: ceil(N / 256)
    Value numBlocks = builder.create<arith::DivUIOp>(loc,
        builder.create<arith::AddIOp>(loc, N, c255), c256);
    
    // Block loop
    auto blockLoop = builder.create<scf::ForOp>(loc, c0, numBlocks, c1);
    builder.setInsertionPointToStart(blockLoop.getBody());
    Value blockIdx = blockLoop.getInductionVar();
    
    // Thread loop
    auto threadLoop = builder.create<scf::ForOp>(loc, c0, c256, c1);
    builder.setInsertionPointToStart(threadLoop.getBody());
    Value threadIdx = threadLoop.getInductionVar();
    
    // Global index
    Value blockOffset = builder.create<arith::MulIOp>(loc, blockIdx, c256);
    Value globalIdx = builder.create<arith::AddIOp>(loc, blockOffset, threadIdx);
    
    // Bounds check
    Value inBounds = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalIdx, N
    );
    
    auto ifOp = builder.create<scf::IfOp>(loc, inBounds, false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    
    // Compute: C[i] = A[i] + B[i]
    Value a = builder.create<memref::LoadOp>(loc, A, globalIdx);
    Value b = builder.create<memref::LoadOp>(loc, B, globalIdx);
    Value sum = builder.create<arith::AddFOp>(loc, a, b);
    builder.create<memref::StoreOp>(loc, sum, C, globalIdx);
}
```

**Key Pattern Elements**:
1. **Grid calculation**: `ceil(N / blockSize)` ensures all elements processed
2. **Index arithmetic**: `blockIdx * blockSize + threadIdx` maps thread to element
3. **Bounds check**: `if (idx < N)` handles non-aligned sizes safely
4. **Memory access**: Coalesced loads (consecutive indices → consecutive memory)

**Matrix Multiplication: 2D Parallelism**. Matrix multiplication requires 2D thread organization:

```
C[i,j] = Σ(k=0 to K-1) A[i,k] * B[k,j]

Input shapes: A[M, K], B[K, N]
Output shape: C[M, N]
```

Each output element C[i,j] requires K multiply-adds (dot product). With 16×16 thread blocks:

```
Thread Block (16×16):
┌────────────────┐
│ (0,0) .. (0,15)│  Each thread computes one C[i,j]
│  ...      ...  │
│(15,0) ..(15,15)│
└────────────────┘

Grid dimensions: ceil(M/16) × ceil(N/16) blocks
Total threads: M × N (one per output element)
```

**Implementation**:

```cpp
void buildMatMulKernel(OpBuilder& builder, Location loc,
                       Value A, Value B, Value C, Value M, Value N, Value K) {
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value c16 = builder.create<arith::ConstantIndexOp>(loc, 16);
    Value c15 = builder.create<arith::ConstantIndexOp>(loc, 15);
    
    // Grid dimensions
    Value gridDimX = builder.create<arith::DivUIOp>(loc,
        builder.create<arith::AddIOp>(loc, M, c15), c16);
    Value gridDimY = builder.create<arith::DivUIOp>(loc,
        builder.create<arith::AddIOp>(loc, N, c15), c16);
    
    // 2D grid loops
    auto blockLoopX = builder.create<scf::ForOp>(loc, c0, gridDimX, c1);
    builder.setInsertionPointToStart(blockLoopX.getBody());
    Value blockIdxX = blockLoopX.getInductionVar();
    
    auto blockLoopY = builder.create<scf::ForOp>(loc, c0, gridDimY, c1);
    builder.setInsertionPointToStart(blockLoopY.getBody());
    Value blockIdxY = blockLoopY.getInductionVar();
    
    // 2D block loops (16×16 threads)
    auto threadLoopX = builder.create<scf::ForOp>(loc, c0, c16, c1);
    builder.setInsertionPointToStart(threadLoopX.getBody());
    Value threadIdxX = threadLoopX.getInductionVar();
    
    auto threadLoopY = builder.create<scf::ForOp>(loc, c0, c16, c1);
    builder.setInsertionPointToStart(threadLoopY.getBody());
    Value threadIdxY = threadLoopY.getInductionVar();
    
    // Compute global indices
    Value row = builder.create<arith::AddIOp>(loc,
        builder.create<arith::MulIOp>(loc, blockIdxX, c16), threadIdxX);
    Value col = builder.create<arith::AddIOp>(loc,
        builder.create<arith::MulIOp>(loc, blockIdxY, c16), threadIdxY);
    
    // Bounds check
    Value rowValid = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, row, M);
    Value colValid = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, col, N);
    Value valid = builder.create<arith::AndIOp>(loc, rowValid, colValid);
    
    auto ifOp = builder.create<scf::IfOp>(loc, valid, false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    
    // Initialize accumulator
    Value zero = builder.create<arith::ConstantFloatOp>(
        loc, APFloat(0.0f), builder.getF32Type()
    );
    
    // Reduction loop: sum over k dimension
    auto kLoop = builder.create<scf::ForOp>(loc, c0, K, c1, ValueRange{zero});
    builder.setInsertionPointToStart(kLoop.getBody());
    Value k = kLoop.getInductionVar();
    Value accum = kLoop.getRegionIterArgs()[0];
    
    // Load A[row, k] and B[k, col]
    Value aVal = builder.create<memref::LoadOp>(loc, A, ValueRange{row, k});
    Value bVal = builder.create<memref::LoadOp>(loc, B, ValueRange{k, col});
    
    // Multiply and accumulate
    Value prod = builder.create<arith::MulFOp>(loc, aVal, bVal);
    Value newAccum = builder.create<arith::AddFOp>(loc, accum, prod);
    
    builder.create<scf::YieldOp>(loc, newAccum);
    builder.setInsertionPointAfter(kLoop);
    
    // Store result
    Value result = kLoop.getResult(0);
    builder.create<memref::StoreOp>(loc, result, C, ValueRange{row, col});
}
```

**Key Differences from Vector Addition**:
1. **4 nested loops**: 2D grid × 2D block (vs 1D grid × 1D block)
2. **2D indexing**: `row = blockIdx.x * 16 + threadIdx.x`, `col = blockIdx.y * 16 + threadIdx.y`
3. **Reduction loop**: Inner loop over K dimension accumulates dot product
4. **Loop-carried dependency**: `scf.ForOp` with iter_args for accumulator

This pattern—4 nested loops for 2D parallelism, inner reduction loop—is fundamental to GPU matrix operations. Attention mechanism, convolution, and transformer blocks all follow this structure.

## 15.6 Reductions and Synchronization

Reduction operations (sum, max, mean) require coordination across threads. Understanding reduction algorithms and synchronization primitives is essential for implementing operations like softmax and layer normalization.

**The Reduction Problem**. Compute sum of N elements using P threads:

```
Input:  [1, 2, 3, 4, 5, 6, 7, 8]  (N = 8)
Output: 36  (sum of all elements)
```

Naive approach: one thread reads all elements (serial, no parallelism). Parallel approach: divide work among threads, combine partial results. For GPU, use **tree reduction**:

```
Step 0: [1, 2, 3, 4, 5, 6, 7, 8]  (8 values)
Step 1: [3, 7, 11, 15]             (4 values, combine pairs)
Step 2: [10, 26]                   (2 values, combine pairs)
Step 3: [36]                       (1 value, final result)
```

Each step halves active threads: 8→4→2→1. Total operations: 7 additions (vs 7 for serial). Speedup comes from parallelism: 4 additions happen simultaneously in step 1.

**Softmax Reduction Pattern**. Softmax requires two reductions: max (for numerical stability) and sum (for normalization):

```
softmax(x)[i] = exp(x[i] - max(x)) / Σ exp(x[j] - max(x))
                                      j
```

**Algorithm**:
1. **Reduction 1**: Find max(x) across all elements
2. **Element-wise**: Compute `exp(x[i] - max)`
3. **Reduction 2**: Sum all exponentials
4. **Element-wise**: Divide by sum

For GPU implementation with N elements and P threads:

```cpp
// Phase 1: Find maximum
float maxVal = -INFINITY;
for (int k = globalIdx; k < N; k += P) {
    maxVal = max(maxVal, input[k]);
}
// Reduce maxVal across threads (tree reduction or atomic max)

// Phase 2: Compute exp(x - max) and sum
float localSum = 0.0f;
for (int k = globalIdx; k < N; k += P) {
    float expVal = exp(input[k] - maxVal);
    temp[k] = expVal;  // Store for later
    localSum += expVal;
}
// Reduce localSum across threads

// Phase 3: Normalize
for (int k = globalIdx; k < N; k += P) {
    output[k] = temp[k] / totalSum;
}
```

**MLIR Implementation** (simplified for 1D softmax):

```cpp
void buildSoftmaxKernel(OpBuilder& builder, Location loc,
                        Value input, Value output, Value N) {
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    
    // Phase 1: Find maximum
    Value negInf = builder.create<arith::ConstantFloatOp>(
        loc, APFloat::getInf(APFloat::IEEEsingle(), /*Negative=*/true),
        builder.getF32Type()
    );
    
    auto maxLoop = builder.create<scf::ForOp>(loc, c0, N, c1, ValueRange{negInf});
    builder.setInsertionPointToStart(maxLoop.getBody());
    Value idx = maxLoop.getInductionVar();
    Value currentMax = maxLoop.getRegionIterArgs()[0];
    
    Value val = builder.create<memref::LoadOp>(loc, input, idx);
    Value newMax = builder.create<arith::MaximumFOp>(loc, currentMax, val);
    
    builder.create<scf::YieldOp>(loc, newMax);
    builder.setInsertionPointAfter(maxLoop);
    Value maxVal = maxLoop.getResult(0);
    
    // Phase 2: Compute exp(x - max) and sum
    Value zero = builder.create<arith::ConstantFloatOp>(
        loc, APFloat(0.0f), builder.getF32Type()
    );
    
    Value tempBuffer = builder.create<memref::AllocaOp>(
        loc, MemRefType::get({ShapedType::kDynamic}, builder.getF32Type()), ValueRange{N}
    );
    
    auto expSumLoop = builder.create<scf::ForOp>(loc, c0, N, c1, ValueRange{zero});
    builder.setInsertionPointToStart(expSumLoop.getBody());
    Value idx2 = expSumLoop.getInductionVar();
    Value currentSum = expSumLoop.getRegionIterArgs()[0];
    
    Value val2 = builder.create<memref::LoadOp>(loc, input, idx2);
    Value shifted = builder.create<arith::SubFOp>(loc, val2, maxVal);
    Value expVal = builder.create<math::ExpOp>(loc, shifted);
    
    builder.create<memref::StoreOp>(loc, expVal, tempBuffer, idx2);
    Value newSum = builder.create<arith::AddFOp>(loc, currentSum, expVal);
    
    builder.create<scf::YieldOp>(loc, newSum);
    builder.setInsertionPointAfter(expSumLoop);
    Value sumVal = expSumLoop.getResult(0);
    
    // Phase 3: Normalize
    auto normLoop = builder.create<scf::ForOp>(loc, c0, N, c1);
    builder.setInsertionPointToStart(normLoop.getBody());
    Value idx3 = normLoop.getInductionVar();
    
    Value expVal3 = builder.create<memref::LoadOp>(loc, tempBuffer, idx3);
    Value normalized = builder.create<arith::DivFOp>(loc, expVal3, sumVal);
    builder.create<memref::StoreOp>(loc, normalized, output, idx3);
}
```

This is a **serial reduction** (CPU-style). On GPU, we'd use parallel tree reduction with synchronization barriers. The algorithmic structure remains identical—three phases with reductions.

**Layer Normalization**. Layer norm requires computing mean and variance:

```
mean = (1/N) Σ x[i]
variance = (1/N) Σ (x[i] - mean)²
output[i] = (x[i] - mean) / sqrt(variance + ε)
```

**Multi-stage reduction**:
1. **Reduction**: Sum all elements → mean = sum / N
2. **Element-wise**: Compute `(x[i] - mean)²`
3. **Reduction**: Sum squared differences → variance = sum / N
4. **Element-wise**: Normalize using mean and variance

Each stage depends on the previous stage's result—no opportunity for fusion without complex synchronization.

## 15.7 Memory Access Patterns

Efficient GPU kernels require careful attention to memory access patterns. Random access kills performance; coalesced access achieves maximum bandwidth.

**Transpose Problem**. Matrix transpose swaps rows and columns:

```
Input (row-major):        Output (row-major):
[1, 2, 3]                 [1, 4]
[4, 5, 6]        →        [2, 5]
                          [3, 6]
```

**Naive Transpose** (poor performance):

```cpp
// Each thread handles one output element
int row = blockIdx.y * 16 + threadIdx.y;
int col = blockIdx.x * 16 + threadIdx.x;

if (row < M && col < N) {
    output[col * M + row] = input[row * N + col];  // Write stride M
}
```

**Problem**: Writing to `output` has stride M—threads write to addresses M elements apart. For a 1024×1024 matrix (M=1024), threads in a warp write to addresses 1024 floats (4096 bytes) apart—no coalescing! This requires 32 separate memory transactions instead of 1.

**Optimized Transpose with Shared Memory**:

```cpp
__shared__ float tile[16][17];  // 17 to avoid bank conflicts

// Load tile from input (coalesced reads)
int inRow = blockIdx.y * 16 + threadIdx.y;
int inCol = blockIdx.x * 16 + threadIdx.x;
if (inRow < M && inCol < N) {
    tile[threadIdx.y][threadIdx.x] = input[inRow * N + inCol];
}
__syncthreads();

// Store tile to output (coalesced writes, transposed indices)
int outRow = blockIdx.x * 16 + threadIdx.y;
int outCol = blockIdx.y * 16 + threadIdx.x;
if (outRow < N && outCol < M) {
    output[outRow * M + outCol] = tile[threadIdx.x][threadIdx.y];  // Note swapped indices
}
```

**How it works**:
1. Load 16×16 tile into shared memory (coalesced: consecutive threads read consecutive addresses)
2. Synchronize (ensure all threads finished loading)
3. Write tile to output with transposed indices (coalesced: consecutive threads write consecutive addresses)

Shared memory access is random (reading `tile[x][y]` instead of `tile[y][x]`), but shared memory is fast (5-10 cycles) and has no coalescing requirement. We trade uncoalesced global writes (500 cycles, 32× penalty) for random shared memory access (10 cycles, no penalty)—50× speedup!

**Attention Score Computation**. Computing attention scores Q @ K^T requires careful memory layout:

```
Q: [seq_len, d_model]  (row-major)
K: [seq_len, d_model]  (row-major)
K^T: [d_model, seq_len]  (logical transpose)
Scores: [seq_len, seq_len]
```

**Option 1**: Materialize K^T explicitly (transpose kernel), then matrix multiply. Clean but requires extra memory and kernel launch.

**Option 2**: Implicit transpose during multiplication—each thread computes one score by reading row of Q and column of K (accessing K in column-major order):

```cpp
// Thread computes scores[i, j]
int i = blockIdx.y * 16 + threadIdx.y;  // Query index
int j = blockIdx.x * 16 + threadIdx.x;  // Key index

float score = 0.0f;
for (int k = 0; k < d_model; k++) {
    score += Q[i * d_model + k] * K[j * d_model + k];  // K accessed in row order (same as Q)
}
```

Wait—K is accessed in row order? Yes! We want `K^T[k, j] = K[j, k]`. Reading `K[j * d_model + k]` gives us the j-th row, k-th element, which is exactly `K^T[k, j]`. Both Q and K reads are coalesced (consecutive k values). No explicit transpose needed!

**General Principle**: Prefer coalesced global memory access over random access. Use shared memory to reorganize data when necessary. Design algorithms to access data in memory order (row-major traversal for row-major arrays).

## 15.8 MLIR GPU Dialect (Preview)

MLIR provides a dedicated GPU dialect for explicit GPU programming. While this chapter uses SCF emulation (CPU execution), understanding the GPU dialect prepares you for real GPU code generation.

**GPU Dialect Operations**. The `gpu` dialect models GPU-specific concepts:

```mlir
// GPU kernel definition
gpu.module @kernels {
  gpu.func @vector_add(%A: memref<?xf32>, %B: memref<?xf32>, 
                       %C: memref<?xf32>, %N: index) kernel {
    // Thread index (no loops needed!)
    %idx = gpu.thread_id x : index
    %blockDim = gpu.block_dim x : index
    %blockIdx = gpu.block_id x : index
    
    // Global index
    %blockOffset = arith.muli %blockIdx, %blockDim : index
    %globalIdx = arith.addi %blockOffset, %idx : index
    
    // Bounds check
    %inBounds = arith.cmpi ult, %globalIdx, %N : index
    scf.if %inBounds {
      %a = memref.load %A[%globalIdx] : memref<?xf32>
      %b = memref.load %B[%globalIdx] : memref<?xf32>
      %c = arith.addf %a, %b : f32
      memref.store %c, %C[%globalIdx] : memref<?xf32>
    }
    gpu.return
  }
}

// Host code: launch kernel
func.func @main(%A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>, %N: index) {
  %c256 = arith.constant 256 : index
  %numBlocks = arith.divui %N, %c256 : index
  
  // GPU kernel launch
  gpu.launch_func @kernels::@vector_add
      blocks in (%numBlocks, %c1, %c1)
      threads in (%c256, %c1, %c1)
      args(%A : memref<?xf32>, %B : memref<?xf32>, %C : memref<?xf32>, %N : index)
  
  return
}
```

**Key Differences from SCF Emulation**:

1. **No explicit loops**: `gpu.thread_id` and `gpu.block_id` are built-in operations (hardware provides indices)
2. **Kernel function**: Marked with `kernel` attribute, separate from host code
3. **Launch syntax**: `gpu.launch_func` specifies grid/block dimensions explicitly
4. **Module separation**: GPU code in `gpu.module`, host code in regular `func.func`

**GPU Dialect Lowering**. MLIR compiles GPU dialect to:

- **NVVM dialect** (NVIDIA): Lowers to PTX assembly, then CUDA driver compiles to GPU machine code
- **ROCDL dialect** (AMD): Lowers to AMDGPU assembly, then ROCm compiles to GPU machine code
- **SPIRV dialect** (Vulkan/OpenCL): Lowers to SPIR-V bytecode for cross-platform execution

The lowering pipeline:

```
GPU dialect → (NVVM / ROCDL / SPIRV) → Assembly → Machine Code
```

Each target has slightly different features (NVIDIA: warp shuffle, AMD: wavefront, SPIR-V: subgroup operations), but the core concepts (thread hierarchy, memory hierarchy, coalescing) are universal.

**When to Use GPU Dialect**. The GPU dialect is appropriate when:

- Targeting actual GPU hardware (NVIDIA, AMD, Intel)
- Need GPU-specific features (shared memory, warp primitives)
- Want explicit control over thread organization
- Generating code for production deployment

For this book's educational goals, SCF emulation is more accessible—no GPU required, same concepts, easier debugging. Chapter 16 references GPU dialect when discussing production serving architectures.

## 15.9 Building a Transformer Block

Armed with GPU programming fundamentals, we implement a complete transformer block: the core building block for GPT architecture. This section demonstrates how primitive operations (matmul, element-wise, reductions) compose into complex neural network layers.

**Transformer Block Structure**:

```
Input: x [seq_len, d_model]
  ↓
1. Layer Normalization
  ↓
2. Self-Attention (Q@K^T → scale → mask → softmax → @V)
  ↓
3. Residual Connection (add input)
  ↓
4. Layer Normalization
  ↓
5. Feed-Forward Network (Linear → GELU → Linear)
  ↓
6. Residual Connection (add input after step 3)
  ↓
Output: x' [seq_len, d_model]
```

Each step uses GPU kernels developed in previous sections. Composing them requires managing intermediate buffers and orchestrating kernel launches.

**Self-Attention Kernel**. The attention mechanism combines multiple operations:

```
Q = x @ W_q    [seq_len, d_model] @ [d_model, d_model] → [seq_len, d_model]
K = x @ W_k    [seq_len, d_model] @ [d_model, d_model] → [seq_len, d_model]
V = x @ W_v    [seq_len, d_model] @ [d_model, d_model] → [seq_len, d_model]

scores = Q @ K^T    [seq_len, d_model] @ [d_model, seq_len] → [seq_len, seq_len]
scores = scores / sqrt(d_model)    Element-wise scale
scores = apply_mask(scores)        Causal mask (zero out future positions)
weights = softmax(scores)          Row-wise softmax [seq_len, seq_len]
output = weights @ V               [seq_len, seq_len] @ [seq_len, d_model] → [seq_len, d_model]
```

**Feed-Forward Network**:

```
hidden = x @ W1    [seq_len, d_model] @ [d_model, 4*d_model] → [seq_len, 4*d_model]
hidden = hidden + bias1             Bias addition (broadcast)
hidden = GELU(hidden)               Element-wise activation
output = hidden @ W2                [seq_len, 4*d_model] @ [4*d_model, d_model] → [seq_len, d_model]
output = output + bias2             Bias addition
```

**Implementation Strategy**: Each operation becomes a kernel (or kernel sequence). The host code orchestrates execution:

```cpp
// Pseudocode: Transformer block execution
transformer_block(x, weights, output) {
    // Layer Norm 1
    layernorm_kernel(x, ln1_gamma, ln1_beta, x_norm);
    
    // Self-Attention
    matmul_kernel(x_norm, W_q, Q);
    matmul_kernel(x_norm, W_k, K);
    matmul_kernel(x_norm, W_v, V);
    
    attention_kernel(Q, K, V, attn_out);  // Combined scores + softmax + output
    
    // Residual 1
    add_kernel(x, attn_out, x_res1);
    
    // Layer Norm 2
    layernorm_kernel(x_res1, ln2_gamma, ln2_beta, x_norm2);
    
    // Feed-Forward
    matmul_kernel(x_norm2, W1, ffn_hidden);
    bias_add_kernel(ffn_hidden, bias1, ffn_hidden);
    gelu_kernel(ffn_hidden, ffn_hidden);
    matmul_kernel(ffn_hidden, W2, ffn_out);
    bias_add_kernel(ffn_out, bias2, ffn_out);
    
    // Residual 2
    add_kernel(x_res1, ffn_out, output);
}
```

Each kernel launch is asynchronous on GPU—the GPU processes queued operations while the CPU continues enqueueing. This **asynchronous execution** overlaps CPU overhead with GPU computation, maximizing throughput.

**KV Caching for Generation**. Chapter 14 introduced KV caching for efficient autoregressive generation. The GPU implementation follows the same algorithm but with parallel thread execution:

```cpp
// Cached attention: process one new token, reuse previous K/V
kv_cached_attention_kernel(
    query,         // [1, d_model] - new token only
    key_cache,     // [max_seq_len, d_model] - accumulated keys
    value_cache,   // [max_seq_len, d_model] - accumulated values
    pos,           // Current position
    output         // [1, d_model]
) {
    // Each thread processes one output dimension
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= d_model) return;
    
    // Compute attention scores for new query against all cached keys
    float score_sum = 0.0f;
    float max_score = -INFINITY;
    
    // Pass 1: Find max score (for numerical stability)
    for (int j = 0; j <= pos; j++) {
        float score = 0.0f;
        for (int k = 0; k < d_model; k++) {
            score += query[k] * key_cache[j * d_model + k];
        }
        score /= sqrt((float)d_model);
        max_score = max(max_score, score);
    }
    
    // Pass 2: Compute exp(score - max) and sum
    float temp_scores[MAX_SEQ_LEN];
    for (int j = 0; j <= pos; j++) {
        float score = 0.0f;
        for (int k = 0; k < d_model; k++) {
            score += query[k] * key_cache[j * d_model + k];
        }
        score = score / sqrt((float)d_model) - max_score;
        temp_scores[j] = exp(score);
        score_sum += temp_scores[j];
    }
    
    // Pass 3: Compute weighted sum of values
    float output_val = 0.0f;
    for (int j = 0; j <= pos; j++) {
        float weight = temp_scores[j] / score_sum;
        output_val += weight * value_cache[j * d_model + d];
    }
    
    output[d] = output_val;
}
```

This kernel processes one new token in O(seq_len) time instead of O(seq_len²) for full recomputation. For 100-token generation: naive approach requires 5,050 attention computations (100 + 99 + ... + 1), cached approach requires 100 attention computations (1 per token). The 50× speedup is algorithmic, independent of GPU hardware.

## 15.10 Summary

Chapter 15 introduced GPU programming fundamentals through CPU emulation using MLIR's SCF dialect. We progressed from basic concepts (thread hierarchies, memory hierarchies, index calculation) through primitive operations (vector addition, matrix multiplication, reductions) to complete transformer blocks with KV caching.

**Key Concepts Mastered**:

- **Thread Organization**: Grid, blocks, threads—three-level hierarchy for massive parallelism
- **Memory Hierarchy**: Registers, shared memory, global memory—50-500× latency differences
- **Index Calculation**: `blockIdx * blockDim + threadIdx` maps threads to data
- **Bounds Checking**: Essential for non-aligned data sizes
- **Memory Coalescing**: Consecutive threads accessing consecutive memory—critical for bandwidth
- **Reduction Algorithms**: Multi-stage patterns for computing aggregates (sum, max, softmax)
- **Warp Execution**: 32-thread groups executing lockstep, divergence costs

**Patterns Established**:

- **1D Parallelism**: Single loop over threads (vector operations)
- **2D Parallelism**: Double nested loops (matrix operations)
- **Reduction Pattern**: Initialize accumulator, loop with accumulation, return result
- **Multi-stage Operations**: Softmax (max → exp → sum → normalize), layer norm (mean → variance → normalize)

**GPU Emulation Value**. CPU emulation with nested loops teaches genuine GPU programming:
- Same indexing arithmetic (`blockIdx * blockSize + threadIdx`)
- Same bounds checking requirements
- Same memory access patterns (coalesced vs strided)
- Same algorithmic structures (reduction trees, multi-stage operations)

The only difference is execution model: CPU runs loops serially, GPU runs them in parallel. The code structure is identical. When you transition to real GPU code (CUDA, ROCm, Metal), you're writing the same kernels with different syntax.

**Connecting the Dots: Two Paths to GPU Execution**

**Path 1 (Chapters 11-14 extended to GPU)**:
```
Write: tensor operations (Chapter 13/14 style)
  ↓
Optimize: Transform Dialect (Chapter 14)
  ↓
Generate: GPU dialect (automatic)
  ↓
Execute: GPU hardware
```
This is the production ML compiler approach (IREE, Torch-MLIR, XLA).

**Path 2 (Chapter 15 direct GPU programming)**:
```
Write: memref + explicit thread indexing (Chapter 15 style)
  ↓
Lower: GPU dialect (manual control)
  ↓
Execute: GPU hardware (or CPU emulation)
```
This is the manual GPU programming approach (CUDA/HIP kernels).

Production systems use **Path 1** (automatic generation from high-level ops). Chapter 15 teaches **Path 2** patterns so you understand what Path 1 generates.

Chapter 15 switched from JIT to AOT compilation workflow. Production ML systems compile models once, serve many times. AOT workflow reflects this reality and matches production compiler architecture.

**Looking Ahead to Production Deployment**. Real production LLM serving combines concepts from Chapters 14 and 15: high-level tensor operations (Chapter 14) compiled to efficient GPU kernels (Chapter 15 patterns) with advanced serving techniques like continuous batching and paged attention. The next step would integrate these foundations with production serving frameworks (vLLM, TensorRT-LLM, SGLang) that automate the high-level → GPU code generation while providing batching, scheduling, and memory management.

Chapter 15 completed the GPU programming foundation. You understand thread hierarchies, memory hierarchies, and parallel algorithms. Chapter 16 applies these concepts to build a complete serving engine—the final step from theory to production system.