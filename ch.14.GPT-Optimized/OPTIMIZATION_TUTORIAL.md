# Chapter 14: Compiler Optimizations for Deep Learning - A Complete Tutorial

## Educational Overview

This chapter explores **high-level compiler optimizations** for neural network inference. We progressively optimize a GPT model by leveraging MLIR's powerful optimization infrastructure, demonstrating how modern compilers achieve significant speedups through automatic transformations.

**Learning Goals:**
- Understand why IR abstraction level matters for optimization
- Learn how compilers automatically fuse operations and eliminate redundancy
- Explore vectorization strategies (SIMD) for numerical code
- Implement algorithmic optimizations (KV cache) with compiler support
- Measure and analyze optimization impact

**Achievement:** 3-5x speedup on forward pass, 10-20x on generation

---

## Part 1: Why Optimize? Understanding the Baseline

### The Performance Problem

In Chapter 13, we built a functional GPT model that generates text correctly. However, it's **slow** (~500ms per forward pass, ~9 seconds to generate 20 tokens). Why?

**Problem 1: Low-level IR prevents optimization**
```cpp
// Chapter 13: Direct SCF loops for matrix multiplication
rewriter.create<scf::ForOp>(...);  // for i in rows
  rewriter.create<scf::ForOp>(...);  // for j in cols
    rewriter.create<scf::ForOp>(...);  // for k in common
      builder.create<memref::LoadOp>(A, {i, k});
      builder.create<memref::LoadOp>(B, {k, j});
      // Manual accumulation with FMA
```

**Why this is slow:**
- Compiler sees individual loop iterations, not the "matrix multiply" pattern
- No automatic tiling for cache efficiency
- No SIMD vectorization (processes 1 float at a time, not 8)
- Operations can't fuse (each has separate loop structure)

**Problem 2: Redundant computation in generation**
```python
# Generate 20 tokens: O(N²) complexity per token!
for new_token in range(20):
    # Recompute attention for ALL previous tokens AGAIN
    attention = softmax(Q @ K.T)  # K grows by 1 each iteration
    # This is wasteful - old tokens don't change!
```

### The Optimization Strategy

We'll solve these problems through **three key techniques:**

1. **Higher-level IR (Linalg)**: Let compiler recognize patterns
2. **Automatic fusion**: Eliminate intermediate memory operations
3. **Algorithmic optimization (KV cache)**: Avoid redundant computation

---

## Part 2: Linalg - Teaching the Compiler About Linear Algebra

### What is Linalg?

**Linalg** is MLIR's "Linear Algebra" dialect - a higher-level IR specifically designed for matrix/tensor operations. Instead of expressing operations as raw loops, we use **semantic operations** that describe *what* we want, not *how* to compute it.

**Key insight:** The more semantics the compiler understands, the better it can optimize!

### Example: Matrix Multiplication

**Low-level approach (Chapter 13):**
```cpp
// Compiler sees: "Three nested loops with loads/stores"
// It doesn't know this is a matmul!
for (i = 0; i < M; i++) {
  for (j = 0; j < N; j++) {
    sum = 0;
    for (k = 0; k < K; k++) {
      sum += A[i][k] * B[k][j];
    }
    C[i][j] = sum;
  }
}
```

**High-level approach (Chapter 14 - Linalg):**
```cpp
// Compiler sees: "This is a matrix multiplication!"
auto matmulOp = rewriter.create<linalg::MatmulOp>(
  loc, 
  ValueRange{lhs, rhs},     // Inputs
  ValueRange{output}        // Output
);
```

**What the compiler gains:**
- **Pattern recognition**: "I know efficient algorithms for matmul!"
- **Automatic tiling**: Break into cache-friendly blocks (32x32x32)
- **Vectorization opportunities**: Use SIMD instructions (AVX2: 8 floats at once)
- **Fusion potential**: Combine with adjacent ops (matmul + bias + activation)

### Element-wise Operations with Linalg.Generic

For operations without named ops (like GELU), we use `linalg.generic` - a flexible way to express any operation over tensors:

```cpp
// Element-wise GELU: y = 0.5 * x * (1 + tanh(...))
auto geluOp = rewriter.create<linalg::GenericOp>(
  loc,
  /*inputs=*/ValueRange{input},
  /*outputs=*/ValueRange{output},
  /*indexingMaps=*/identityMap,          // One-to-one element mapping
  /*iteratorTypes=*/parallel,            // All dimensions parallel
  /*body=*/[&](OpBuilder &b, Location loc, ValueRange args) {
    // Compute GELU for single element
    Value x = args[0];
    Value result = compute_gelu(x);
    b.create<linalg::YieldOp>(loc, result);
  }
);
```

**Key concepts:**
- **Indexing maps**: Describe how inputs/outputs relate (identity = same shape)
- **Iterator types**: Parallel (independent) or reduction (accumulation)
- **Body region**: Scalar computation applied to each element

### Implementation Pattern

**Step 1: Add Linalg dialect to your context**
```cpp
context_.loadDialect<linalg::LinalgDialect>();
```

**Step 2: Rewrite lowering passes**
```cpp
struct MatMulOpLowering : public OpConversionPattern<MatMulOp> {
  LogicalResult matchAndRewrite(
      MatMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    // Old way: Create 3 nested SCF loops manually
    // New way: Single linalg op
    auto matmul = rewriter.create<linalg::MatmulOp>(
      op.getLoc(),
      ValueRange{adaptor.getLhs(), adaptor.getRhs()},
      ValueRange{adaptor.getOutput()}
    );
    
    rewriter.replaceOp(op, matmul.getResults());
    return success();
  }
};
```

**Step 3: Test correctness**
All numerical outputs should match Chapter 13 exactly (within floating-point tolerance).

---

## Part 3: Operation Fusion - Eliminating Intermediate Memory

### The Memory Bottleneck

Modern CPUs are **memory-bound** for neural network inference. Arithmetic is fast (~1 cycle for multiply-add), but loading data from memory is slow (~100+ cycles for cache miss).

**Inefficient pattern:**
```cpp
// Three separate operations = three memory round-trips
tmp1 = matmul(X, W)       // Write tmp1 to memory
tmp2 = add(tmp1, bias)    // Read tmp1, write tmp2 to memory  
output = gelu(tmp2)       // Read tmp2, write output to memory
```

**Each intermediate result:**
- Must be written to memory (cache or RAM)
- Must be read back for the next operation
- Consumes memory bandwidth

### What is Fusion?

**Fusion** means combining multiple operations into a single computational kernel that processes data in one pass, eliminating intermediate memory traffic.

**Fused pattern:**
```cpp
// Single fused operation - no intermediate memory!
output = matmul_bias_gelu(X, W, bias)  // Compute all in registers
```

**How it works:**
```cpp
for (i, j) {
  // Compute matmul output for one position
  sum = 0;
  for (k) {
    sum += X[i][k] * W[k][j];
  }
  // Immediately apply bias (no memory write!)
  sum = sum + bias[j];
  // Immediately apply GELU (no memory write!)
  output[i][j] = 0.5 * sum * (1 + tanh(...));
}
// Only write final result
```

### Enabling Fusion with Linalg

Linalg ops are designed to be **fusible**. The compiler can automatically recognize fusion opportunities and apply them.

**Enable fusion in your pipeline:**
```cpp
PassManager pm(&context_);

// 1. Lower to Linalg ops (Chapter 14 lowering)
pm.addNestedPass<func::FuncOp>(createLowerTransformerToLinalgPass());

// 2. FUSION PASS - This is where magic happens!
pm.addPass(createLinalgElementwiseOpFusionPass());

// 3. Canonicalization cleans up
pm.addPass(createCanonicalizerPass());
```

**What the fusion pass does:**
- Identifies "producer-consumer" chains (matmul → add → gelu)
- Checks if operations can be fused (compatible iteration spaces)
- Combines them into a single `linalg.generic` operation
- Eliminates intermediate allocations

**Verifying fusion:**
```bash
# Before fusion: separate operations
%0 = linalg.matmul ins(%A, %B) outs(%tmp1)
%1 = linalg.add ins(%tmp1, %bias) outs(%tmp2)
%2 = linalg.gelu ins(%tmp2) outs(%output)

# After fusion: single operation
%0 = linalg.generic ... {
  ^bb0(%a, %b, %bias):
    %matmul_result = ...
    %add_result = arith.addf %matmul_result, %bias
    %gelu_result = ... // GELU computation
    linalg.yield %gelu_result
} ins(%A, %B, %bias) outs(%output)
```

**Expected speedup:** 1.2-1.5x (memory bandwidth savings)

**Note:** For memref-based linalg (bufferized), fusion is limited because buffers are pre-allocated. Tensor-based linalg has better fusion potential.

---

## Part 4: Loop Optimizations - LICM and Canonicalization

### Loop Invariant Code Motion (LICM)

**Problem:** Redundant computation inside loops
```cpp
for (i = 0; i < N; i++) {
  constant_value = load(addr);  // Same every iteration!
  result[i] = array[i] + constant_value;
}
```

**Solution:** Hoist invariant code outside the loop
```cpp
constant_value = load(addr);  // Computed once
for (i = 0; i < N; i++) {
  result[i] = array[i] + constant_value;
}
```

**Enable LICM:**
```cpp
pm.addPass(createLoopInvariantCodeMotionPass());
```

### Canonicalization

**Canonicalization** simplifies IR by applying algebraic identities and pattern rewrites:
- `x + 0 → x`
- `x * 1 → x`
- Constant folding: `2 * 3 → 6`
- Dead code elimination

**Always run after transformations:**
```cpp
pm.addPass(createCanonicalizerPass());  // Clean up IR
```

---

## Part 5: Vectorization - SIMD for Parallel Computing

### What is Vectorization?

Modern CPUs have **SIMD** (Single Instruction Multiple Data) instructions that operate on multiple values simultaneously:
- AVX2: Process 8 floats (256 bits) in one instruction
- AVX-512: Process 16 floats (512 bits) in one instruction

**Scalar (slow):**
```cpp
// Process one float at a time
for (int i = 0; i < N; i++) {
  result[i] = a[i] + b[i];  // 1 addition per iteration
}
```

**Vectorized (fast):**
```cpp
// Process 8 floats at a time (AVX2)
for (int i = 0; i < N; i += 8) {
  __m256 va = _mm256_load_ps(&a[i]);     // Load 8 floats
  __m256 vb = _mm256_load_ps(&b[i]);     // Load 8 floats
  __m256 vr = _mm256_add_ps(va, vb);     // Add 8 floats in one instruction!
  _mm256_store_ps(&result[i], vr);       // Store 8 floats
}
// 8x fewer iterations, 8x more work per iteration
```

### MLIR Vector Dialect

MLIR's **Vector dialect** provides a target-independent way to express SIMD operations. The compiler lowers these to architecture-specific instructions (AVX2, NEON, etc.).

**Setup Vector infrastructure:**
```cpp
// 1. Load dialect
context_.loadDialect<vector::VectorDialect>();

// 2. Add conversion passes
pm.addPass(createConvertVectorToLLVMPass());  // Vector → LLVM SIMD
pm.addPass(createConvertVectorToSCFPass());   // Vector → loops (fallback)
```

### Two Approaches to Vectorization

**Approach A: Automatic (Limited in MLIR 19)**
MLIR 19 doesn't have general auto-vectorization passes. The compiler will vectorize some linalg ops opportunistically, but it's not guaranteed.

**Approach B: Explicit with Transform Dialect (Recommended)**
Use the **Transform dialect** to explicitly specify vectorization strategies:

```cpp
// Tile matmul into smaller blocks, then vectorize each block
transform.sequence {
  %matmul = transform.structured.match ops{["linalg.matmul"]}
  
  // Tile into 32x32x32 blocks
  %tiled = transform.structured.tile %matmul [32, 32, 32]
  
  // Vectorize each block
  %vectorized = transform.structured.vectorize %tiled
}
```

**Expected speedup:** 2-3x (8x theoretical, but memory-bound operations don't scale linearly)

**Implementation timeline:**
- **Phase 4:** Load Vector dialect infrastructure ✅ Complete
- **Phase 6:** Apply explicit vectorization strategies (deferred after KV cache)

---

## Part 6: Algorithmic Optimization - KV Cache

### The Generation Problem

Text generation is **autoregressive**: each new token depends on all previous tokens. This creates a performance crisis:

**Without cache (naive):**
```python
tokens = [tok0]
for step in range(20):  # Generate 20 tokens
    # Process ENTIRE sequence from scratch every time!
    # Step 1: Process 1 token   → O(1²) = 1
    # Step 2: Process 2 tokens  → O(2²) = 4
    # Step 3: Process 3 tokens  → O(3²) = 9
    # ...
    # Step 20: Process 20 tokens → O(20²) = 400
    # Total: O(N³) complexity! (~500 operations for 20 tokens)
    
    hidden = model.forward(tokens)
    next_token = sample(hidden[-1])
    tokens.append(next_token)
```

**Problem:** We're recomputing attention for old tokens **that never change**!

### The KV Cache Solution

**Key insight:** In self-attention, each token's Key and Value vectors are computed from that token's representation, which doesn't change after computation.

**Attention computation:**
```python
Q = hidden @ Wq  # Query for new token
K = hidden @ Wk  # Keys for all tokens (including old ones)
V = hidden @ Wv  # Values for all tokens (including old ones)

scores = Q @ K.T  # Attention scores
attn = softmax(scores) @ V  # Weighted sum of values
```

**Optimization:** Cache K and V for processed tokens!
```python
# Initialize caches [max_seq_len, d_model]
k_cache = zeros(max_seq_len, d_model)
v_cache = zeros(max_seq_len, d_model)

# First token: compute and cache
k_cache[0] = new_token @ Wk
v_cache[0] = new_token @ Wv
Q = new_token @ Wq
scores = Q @ k_cache[0:1].T  # Only 1 cached token
attn = softmax(scores) @ v_cache[0:1]

# Second token: reuse cached K/V!
k_cache[1] = new_token @ Wk  # Only compute for NEW token
v_cache[1] = new_token @ Wv
Q = new_token @ Wq
scores = Q @ k_cache[0:2].T  # Reuse both cached tokens
attn = softmax(scores) @ v_cache[0:2]
```

**Complexity improvement:**
- Without cache: O(N²) per token → O(N³) total
- With cache: O(N) per token → O(N²) total
- **Speedup: 10-100x for generation!**

### Implementation Strategy

**C++ function for cached attention:**
```cpp
py::array_t<float> gpt_attention_cached(
    py::array_t<float> new_token_hidden,  // [1, d_model] - just new token
    py::array_t<float> k_cache,           // [max_seq, d_model] - updated in-place
    py::array_t<float> v_cache,           // [max_seq, d_model] - updated in-place
    int cache_pos,                        // Where to write new K/V
    py::array_t<float> Wq, py::array_t<float> bq,
    py::array_t<float> Wk, py::array_t<float> bk,
    py::array_t<float> Wv, py::array_t<float> bv,
    py::array_t<float> Wo, py::array_t<float> bo,
    int n_heads) {
  
  // 1. Project new token to Q/K/V
  // 2. Update k_cache[cache_pos] and v_cache[cache_pos]
  // 3. Compute attention using full cache (0:cache_pos+1)
  // 4. Return output projection
}
```

**Python generation with cache:**
```python
def generate_cached(prompt_tokens, model, max_new_tokens=20):
    # Initialize per-layer caches
    k_caches = [np.zeros((max_seq, d_model)) for _ in range(n_layers)]
    v_caches = [np.zeros((max_seq, d_model)) for _ in range(n_layers)]
    
    # Process prompt (fill caches)
    hidden = embed(prompt_tokens[0])
    for layer in range(n_layers):
        hidden = cached_attention(hidden, k_caches[layer], v_caches[layer], 0, ...)
        hidden = ffn(hidden)
    
    # Generate tokens incrementally
    for pos in range(1, max_new_tokens):
        hidden = embed(next_token)
        for layer in range(n_layers):
            hidden = cached_attention(hidden, k_caches[layer], v_caches[layer], pos, ...)
            hidden = ffn(hidden)
        next_token = sample(hidden)
```

**Testing strategy:**
1. Verify numerical correctness: cached output == non-cached output
2. Test incremental updates: process tokens one-by-one
3. Measure speedup: time cached vs non-cached generation

**Expected results:**
- 20-token generation: 10-20x faster
- 100-token generation: 50-100x faster
- Memory overhead: 2 * n_layers * max_seq * d_model floats (~50MB for typical config)

---

## Part 7: The Complete Optimization Pipeline

### Putting It All Together

Here's the complete compilation pipeline for Chapter 14:

```cpp
bool lowerToLLVM(ModuleOp module) {
  PassManager pm(&context_);
  
  // ===== Stage 1: High-level Dialect Lowering =====
  // Custom ops → Linalg ops (enables optimization)
  pm.addNestedPass<func::FuncOp>(createLowerTransformerToLinalgPass());
  pm.addPass(createCanonicalizerPass());
  
  // ===== Stage 2: Linalg Optimizations =====
  // Fuse element-wise operations
  pm.addPass(createLinalgElementwiseOpFusionPass());
  pm.addPass(createCanonicalizerPass());
  
  // ===== Stage 3: Loop Optimizations =====
  // Convert linalg → loops for further optimization
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createLoopInvariantCodeMotionPass());  // LICM
  pm.addPass(createCanonicalizerPass());
  
  // ===== Stage 4: Vectorization (Phase 6 - Future) =====
  // TODO: Add Transform dialect vectorization
  // pm.addPass(createLinalgVectorizationWithTransform());
  
  // ===== Stage 5: Standard Lowering =====
  // Vector → LLVM SIMD
  pm.addPass(createConvertVectorToLLVMPass());
  
  // Math/SCF/MemRef → LLVM
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  
  // Cleanup
  pm.addPass(createReconcileUnrealizedCastsPass());
  
  return succeeded(pm.run(module));
}
```

### Required Headers and Libraries

**Headers:**
```cpp
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/Vector/Transforms/Passes.h>
#include <mlir/Transforms/Passes.h>  // LICM, Canonicalization
```

**CMakeLists.txt:**
```cmake
target_link_libraries(ch14 PRIVATE
  # Linalg
  MLIRLinalgDialect
  MLIRLinalgTransforms
  
  # Vector
  MLIRVectorDialect
  MLIRVectorTransforms
  MLIRVectorToLLVMPass
  MLIRVectorToSCF
  
  # Standard passes
  MLIRTransforms
  
  # ... other libraries
)
```

---

## Part 8: Measuring Performance

### Benchmarking Best Practices

**1. Warm up the JIT:**
```python
# First run compiles, subsequent runs measure execution
for _ in range(10):
    model.forward(input)  # Warm-up

start = time.perf_counter()
for _ in range(100):
    result = model.forward(input)
end = time.perf_counter()
avg_time = (end - start) / 100
```

**2. Compare apples-to-apples:**
- Same inputs
- Same hardware
- Same environment (temperature, background processes)
- Multiple runs for statistical significance

**3. Profile bottlenecks:**
```bash
# Linux perf
perf stat -e cycles,instructions,cache-misses python benchmark.py

# Verify SIMD usage
objdump -d ch14.so | grep vmov  # Look for AVX instructions
```

### Expected Results Summary

| Optimization | Speedup | Why |
|-------------|---------|-----|
| **Linalg rewrite** | 1.0x | Foundation - no speedup yet |
| **Operation fusion** | 1.2-1.5x | Reduced memory traffic |
| **LICM** | 1.1x | Hoisted invariant loads |
| **Vectorization** | 2-3x | SIMD (8 floats at once) |
| **Combined forward pass** | **3-5x** | Multiplicative effects |
| **KV cache (generation)** | **10-100x** | Algorithmic improvement |

---

## Summary: What We Learned

### Key Takeaways

1. **IR abstraction level matters**: High-level ops (linalg) enable optimizations that low-level loops (SCF) cannot.

2. **Compiler optimizations compound**: Fusion + LICM + vectorization = multiplicative speedup.

3. **Memory is the bottleneck**: Most speedups come from reducing memory traffic, not compute.

4. **Algorithmic wins dominate**: KV cache (10-100x) beats micro-optimizations (2-3x) by orders of magnitude.

5. **Measure everything**: "In theory, theory and practice are the same. In practice, they are not." Always benchmark!

### The Optimization Hierarchy

```
Level 1: Algorithmic (KV cache)          → 10-100x speedup
Level 2: Compiler (fusion, vectorization) → 3-5x speedup
Level 3: Hardware (GPU, TPU)             → 10-100x speedup
Level 4: Distributed (multi-node)        → Nx speedup (N nodes)
```

**Lesson:** Start with the highest-leverage optimizations first!

### Next Steps

**Phase 6: Explicit Vectorization**
- Use Transform dialect for tile-and-vectorize
- Target matmul and element-wise ops
- Goal: Additional 2-3x speedup

**Beyond Chapter 14:**
- Multi-threaded execution (OpenMP)
- GPU acceleration (CUDA/ROCm)
- Mixed precision (FP16/BF16)
- Quantization (INT8)
- Distributed inference

---

## References and Further Reading

### MLIR Documentation
- [Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [Transform Dialect](https://mlir.llvm.org/docs/Dialects/Transform/)
- [Vector Dialect](https://mlir.llvm.org/docs/Dialects/Vector/)
- [Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)

### Academic Papers
- "Polyhedral Compilation for Deep Learning" (Verdoolaege et al.)
- "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning" (Chen et al.)
- "IREE: A Retargetable MLIR-based Compiler" (Google)

### Tools and Projects
- **IREE**: Production MLIR compiler for ML
- **Torch-MLIR**: PyTorch frontend for MLIR
- **OpenXLA**: Unified ML compiler platform

---

*Happy optimizing! Remember: premature optimization is the root of all evil, but knowing how to optimize is the root of all performance.*
