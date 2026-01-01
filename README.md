# MLIR Learn-by-Doing Tutorial

A hands-on tutorial demonstrating MLIR JIT compilation, progressing from basic matrix operations through neural network operators to production-grade LLM serving.

**What You'll Build**:
- Chapters 1-4: MLIR fundamentals (JIT compilation, dynamic shapes, bufferization)
- Chapters 5-7: Neural operations (softmax, computation graphs, operator composition)
- Chapters 8-9: Custom dialects (TableGen, universal bindings)
- Chapters 10-11: Optimization passes and attention mechanisms
- Chapters 12-14: Complete GPT model with KV-caching and optimizations
- Chapter 15: GPU programming concepts
- Chapter 16: NANO LLM serving engine

## Quick Start

### Prerequisites

```bash
# Ubuntu/WSL2 - LLVM 21 Required

# Add LLVM 21 repository
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
sudo add-apt-repository -y "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-21 main"
sudo apt update

# Install LLVM/MLIR 21
sudo apt install -y llvm-21 llvm-21-dev llvm-21-runtime
sudo apt install -y mlir-21-tools libmlir-21-dev
sudo apt install -y clang-21

# Install build tools and Python dependencies
sudo apt install -y cmake python3 python3-dev python3-numpy ninja-build libzstd-dev libffi-dev pkg-config
```

### Build All Chapters

```bash
# Configure (one time)
cmake --preset x64-release

# Build
cmake --build --preset x64-release
```

Outputs: `build/x64-release/ch.*/ch*_*.so`

### Test a Chapter

```bash
cd ch.1.Fixed-size
python3 test_jit.py
```

## Chapter Progression

### Chapter 1: Fixed-Size Matrices
**Goal**: Understand MLIR basics with hardcoded 8×32 × 32×16 multiplication

**Key Concepts**:
- MLIR IR generation (`linalg.matmul`)
- Optimization pipeline (affine, linalg, SCF)
- JIT execution with mlir::ExecutionEngine
- Python binding basics

**Documentation**: `ch.1.Fixed-size/README.md`, `MLIR_CODING_PATTERN.md`

### Chapter 2: Dynamic Shapes
**Goal**: Support arbitrary matrix dimensions at runtime

**Key Concepts**:
- Dynamic shape syntax: `tensor<?x?xf32>`
- Runtime dimension passing (21-parameter memref ABI)
- Shape-agnostic IR generation
- Out-parameter bufferization pattern

### Chapter 3: JIT Caching
**Goal**: Optimize performance by caching compiled functions

**Key Concepts**:
- Single compilation for all shapes
- Function pointer caching
- Performance measurement (first call vs cached)
- Shape flexibility without recompilation

### Chapter 4: Tensor Bufferization
**Goal**: Master advanced MLIR bufferization strategies

**Key Concepts**:
- Tensor vs memref semantics
- One-shot bufferization
- Buffer-results-to-out-params pass
- Clean Python API design

**Documentation**: `ch.4.Tensor-bufferization/README.md`, `BUFFERIZATION_GUIDE.md`

### Chapter 5: Vector Operations & SCF Dialect
**Goal**: Learn structured control flow with explicit loops

**Kernel**: SAXPY (Scalar A times X Plus Y): `C[i] = α·A[i] + B[i]`

**Key Concepts**:
- SCF (Structured Control Flow) dialect
- `scf.for` loops with induction variables
- `memref.dim` for dynamic size queries
- Comparison: SCF explicit loops vs Linalg high-level ops

**Documentation**: `ch.5.Vector-ops/README.md`

### Chapter 6: Softmax with Math Dialect
**Goal**: Implement softmax using Math dialect for mathematical functions

**Kernel**: Softmax: `output[i] = exp(input[i] - max) / sum(exp(...))`

**Key Concepts**:
- Math dialect operations (`math.exp`)
- Multi-pass algorithms (find max, compute exp/sum, normalize)
- Numerical stability techniques
- Loop-carried variables in `scf.for`
- Math-to-LLVM and Math-to-Libm lowering strategies

**Documentation**: `ch.6.Softmax/README.md`

### Chapter 7: Neural Operations (Operator Composition)
**Goal**: Build computation graphs with deferred execution before introducing custom dialects

**Operations**: Element-wise (add, mul), matrix multiplication, activations (ReLU, softmax)

**Key Concepts**:
- Computation graph with symbolic operation tracking
- Deferred execution model (build → compile → execute)
- Recursive IR generation from graph structure
- Operation composition for multi-layer networks
- MLIR calling convention for static memrefs (5 params for 1D, 7 for 2D)
- Critical lesson: Math dialect lowering pass order (math-to-llvm THEN math-to-libm)

**Documentation**: `ch.7.Neural-ops/README.md`

### Chapter 8: Universal Bindings with libffi
**Goal**: Implement production-grade universal Python bindings

**Key Concepts**:
- libffi-based marshalling (eliminates per-kernel wrappers)
- Dynamic FFI dispatch for ANY function signature
- ~60% code reduction vs explicit parameter cases
- Python string-based custom dialect workflow
- Comparison with industrial compilers (IREE)

**Documentation**: `ch.8.Custom-dialect/README.md`

### Chapter 9: TableGen Custom Dialect
**Goal**: Build production-grade custom dialect using TableGen/ODS

**Key Concepts**:
- TableGen dialect definition (NNDialect.td, NNOps.td)
- Operation Definition Specification (ODS)
- OpBuilder for programmatic IR construction
- Memref-based operations (nn.add, nn.mul, nn.matmul, nn.relu)
- Flat directory structure (inc/, src/)
- PyTorch-like API (`ch9.forward()`)

**Documentation**: `ch.9.TableGen-dialect/README.md`, `TUTORIAL.md`

### Chapter 10: Optimization Passes
**Goal**: Compare baseline vs optimized lowering pipelines

**Key Concepts**:
- Linalg generalization and fusion
- Performance measurement and comparison
- Reusing Chapter 9's NN dialect
- LLVM 21 API updates

### Chapter 11: Multi-Head Attention with Transformer Dialect
**Goal**: Implement multi-head scaled dot-product attention with custom dialect

**Operations**: matmul, add, mul, softmax, transpose, attention

**Key Concepts**:
- TableGen without attributes (LLVM 21 BytecodeOpInterface issue)
- Minimal includes pattern (only mlir/IR/OpBase.td)
- Multi-pass softmax with numerical stability
- C++ reference implementation for validation
- LLVM 21 compatibility (API changes in ExecutionEngine, cast methods)
- Lowering transformer ops to standard MLIR (scf.for loops, memref operations)

**Documentation**: `ch.11.Attention/README.md`

### Chapter 12: Transformer Block
**Goal**: Build complete transformer block with feedforward network

**Operations**: Multi-head attention, layer normalization, feedforward MLP (2 linear layers)

**Key Concepts**:
- Transformer block composition (attention + FFN)
- Layer normalization implementation
- Residual connections
- Complete forward pass for transformer architecture

**Documentation**: `ch.12.Transformer/README.md`

### Chapter 13: GPT Model (Inference-Only)
**Goal**: Build complete GPT-2 style autoregressive language model

**Components**: Token embedding, position embedding, transformer blocks, language model head

**Key Concepts**:
- Full GPT architecture assembly
- Weight loading from pretrained models
- Autoregressive text generation
- Greedy and temperature-based sampling
- PyTorch weight compatibility

**Documentation**: `ch.13.GPT/README.md`

### Chapter 14: Optimized GPT Inference
**Goal**: Production-grade GPT inference with KV-caching and optimization passes

**Optimizations**:
- KV-cache for O(1) decode (vs O(n²) recomputation)
- Prefill vs decode mode separation
- MLIR optimization passes (linalg fusion, vectorization)
- Batch inference support

**Documentation**: `ch.14.GPT-Optimized/README.md`, `TUTORIAL.md`

### Chapter 15: GPU Programming Concepts
**Goal**: Learn GPU programming by examining GPU dialect IR (no GPU hardware required)

**Key Concepts**:
- GPU dialect IR structure (gpu.module, gpu.func, gpu.launch_func)
- Thread indexing (gpu.thread_id, gpu.block_id, gpu.block_dim)
- Memory hierarchies (global memory vs workgroup/shared memory)
- Synchronization primitives (gpu.barrier)
- Three representative patterns: 1D parallelism, 2D+shared memory, reductions

**Implementation**: Python module generating GPU IR for three kernels (vector add, matmul, softmax)

**Documentation**: `ch.15.GPU-Concepts/README.md`

### Chapter 16: Nano LLM Serving
**Goal**: Build production-style LLM serving engine with modern optimizations

**6 Implementation Phases**:
1. **Request & Batch Abstraction**: Multi-request parallel processing
2. **KV Cache Pool (C++)**: Paged memory management (10-30x capacity)
3. **Prefill vs Decode**: Separate scheduling for different phases
4. **Chunked Prefill**: Fair scheduling for long contexts
5. **Radix Cache**: Automatic prefix sharing (40-60% hit rate = 2-3x speedup)
6. **Continuous Batching**: Dynamic request scheduling (2-5x throughput)

**Performance**: 100-500x faster than naive implementations through:
- Radix attention (SGLang-style prefix caching)
- Continuous batching (vLLM-style dynamic scheduling)
- Chunked prefill (TensorRT-LLM-style fairness)

**Documentation**: `ch.16.Nano-Serving/README.md`, `ch.16.Nano-Serving/TUTORIAL.md` (1,300+ lines)

## Key Implementation Details

### Python Binding Layer (`bindings.cpp`)

```cpp
py::array_t<float> gemm(py::array_t<float> A, py::array_t<float> B) {
  // Extract dimensions
  int64_t M = A.shape[0];
  int64_t K = A.shape[1];
  int64_t N = B.shape[1];

  // Allocate output array (hidden from user!)
  auto C = py::array_t<float>({M, N});

  // Call JIT function with C as out-parameter
  mlir::executeGemm(A.ptr, B.ptr, C.ptr, M, N, K);

  return C;  // Return to Python
}
```

The Python binding:
1. Accepts two NumPy arrays (A, B)
2. **Automatically allocates** the output array C
3. Calls the JIT function with C as an out-parameter
4. Returns C to Python

This design gives users a clean API while using efficient out-parameter style internally.

### JIT Function Signature (Internal)

After MLIR bufferization and lowering, the compiled function has this signature:

```cpp
// LLVM-level signature (after buffer-results-to-out-params pass)
void gemm(
    float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // A memref
    float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,  // B memref
    float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t   // C memref
);
```

Each `memref<?x?xf32>` expands to 7 parameters:
- `ptr, ptr, offset, size0, size1, stride0, stride1`

Total: 3 memrefs × 7 params = **21 parameters**

The function takes C as the third memref parameter (out-parameter style) and returns `void`.

### Why Out-Parameters?

MLIR's bufferization pipeline uses the `buffer-results-to-out-params` pass to convert:

```mlir
// Before: Return value
func.func @gemm(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) 
    -> memref<?x?xf32>

// After: Out-parameter
func.func @gemm(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                %arg2: memref<?x?xf32>)
```

This is necessary because:
1. **Memory management**: Returning a memref would require the caller to know where it was allocated
2. **ABI compatibility**: Struct returns (memref descriptors) have complex calling conventions
3. **Performance**: Out-parameters avoid copying large descriptors on the stack

## Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│ Python User Code                                        │
│   C = gemm.gemm(A, B)  ← Clean API!                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Python Binding (bindings.cpp)                           │
│   • Allocates C internally                              │
│   • Calls executeGemm(A, B, C, M, N, K)                 │
│   • Returns C to Python                                 │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ JIT Executor (jit.cpp)                                  │
│   • Expands memrefs to 21 parameters                    │
│   • Calls: gemm(A[7], B[7], C[7])                       │
│   • C is out-parameter (void return)                    │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ MLIR Bufferization Pipeline (lowering.cpp)              │
│   1. One-Shot Bufferize (tensor → memref)               │
│   2. Buffer-Results-To-Out-Params (return → out-param)  │
│   3. Bufferization-To-MemRef (finalize memrefs)         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Generated Machine Code                                  │
│   • malloc buffer                                       │
│   • memset(buffer, 0)                                   │
│   • matmul(A, B, buffer)                                │
│   • memcpy(buffer, C_out_param)                         │
└─────────────────────────────────────────────────────────┘
```

The user sees a clean functional API, but internally it uses efficient imperative memory operations with out-parameters.

## Python API Examples

All chapters export the same clean API:

```python
import numpy as np

# Chapter 1 (fixed size only)
import ch1_fixed_size
A = np.ones((8, 32), dtype=np.float32)
B = np.ones((32, 16), dtype=np.float32)
C = ch1_fixed_size.gemm(A, B)  # Returns (8, 16)

# Chapters 2-4 (any size)
import ch2_dynamic_size
A = np.random.rand(100, 50).astype(np.float32)
B = np.random.rand(50, 25).astype(np.float32)
C = ch2_dynamic_size.gemm(A, B)  # Returns (100, 25)

# Chapter 13-14: GPT Inference
import ch14_gpt_optimized as gpt
weights = gpt.load_gpt2_weights()
output = gpt.generate("Hello world", max_tokens=50, temperature=0.8)

# Chapter 16: LLM Serving Engine
from nano_engine import NanoServingEngine, SamplingParams
engine = NanoServingEngine(config, weights, kv_cache_pages=400)
prompts = [[1, 2, 3], [4, 5, 6]]
params = [SamplingParams(max_tokens=50, temperature=0.8) for _ in prompts]
results = engine.generate(prompts, params)  # Batch inference with all optimizations
```