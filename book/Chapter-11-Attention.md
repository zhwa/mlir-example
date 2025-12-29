# Chapter 11: Attention Mechanisms in MLIR

Chapters 5-10 built the foundation: operations (Chapter 7), custom dialects (Chapters 8-9), and optimization pipelines (Chapter 10). Now we apply these techniques to modern deep learning's core innovation: **attention mechanisms**. Introduced by Vaswani et al. in "Attention is All You Need" (2017), attention transformed natural language processing and became the foundation of transformers, BERT, GPT, and essentially all state-of-the-art language models. This chapter implements scaled dot-product attention—the mathematical heart of transformers—in MLIR, demonstrating how compiler infrastructure handles complex ML operations.

Unlike previous chapters that explained algorithms (softmax in Chapter 6, neural operations in Chapter 7), we assume you know attention theory. If you need a refresher, consult the original Vaswani paper or any modern NLP textbook. Our focus: **MLIR implementation**—how to represent attention operations in IR, lower them efficiently, and compose them into reusable components. We'll create a `transformer` dialect containing operations like `matmul`, `softmax`, and `transpose`, then lower them to standard MLIR dialects for execution. By chapter's end, you'll have working attention that matches Python's output numerically, compiled entirely through MLIR's JIT infrastructure.

The chapter progresses from understanding attention's computational structure (Q, K, V matrices and their transformations) through dialect design (what operations attention needs, how to specify them in TableGen) to implementation (lowering patterns, numerical stability, debugging). We'll see attention's irregular data access patterns—unlike element-wise operations from Chapter 7, attention involves matrix transposes, reductions across different dimensions, and cascaded operations where one's output feeds another. These patterns challenge naive compilation but showcase MLIR's strength: representing complex operations at high levels while generating efficient code at low levels. Let's begin by examining what attention computes and why its structure matters for compilation.

## 11.1 Scaled Dot-Product Attention: The Core Operation

Before implementing attention in MLIR, let's establish the computational structure precisely. Attention mechanisms compute **contextualized representations**: given a query, they retrieve relevant information from a set of keys and values. The mathematics is elegant but involves several distinct operations, each with different computational characteristics.

**The Attention Formula**. Scaled dot-product attention takes three inputs:
- **Queries (Q)**: Shape `(seq_len, d_k)` - what we're looking for
- **Keys (K)**: Shape `(seq_len, d_k)` - what's available to match against
- **Values (V)**: Shape `(seq_len, d_v)` - the actual information to retrieve

The computation proceeds in four stages:

```
1. Similarity scores:  scores = Q @ K^T           # (seq_len, seq_len)
2. Scaling:            scaled = scores / √d_k     # numerical stability
3. Normalization:      weights = softmax(scaled)  # (seq_len, seq_len)
4. Weighted sum:       output = weights @ V       # (seq_len, d_v)
```

Each stage has distinct characteristics:
- **Stage 1 (matmul)**: O(seq_len² × d_k) compute, creates large intermediate (seq_len²)
- **Stage 2 (scaling)**: O(seq_len²) element-wise, memory-bound
- **Stage 3 (softmax)**: O(seq_len²), requires reductions (max, sum), numerically sensitive
- **Stage 4 (matmul)**: O(seq_len² × d_v) compute, produces final output

**Why This Structure Matters**. From a compiler perspective, attention is interesting because:

1. **Mixed Operation Types**: Compute-intensive matrix multiplications (stages 1, 4) and memory-bound element-wise operations (stage 2) require different optimization strategies.

2. **Large Intermediates**: The scores matrix is `seq_len × seq_len`—for 512-token sequences, that's 262,144 elements. For 2048 tokens (common in modern LLMs), it's 4,194,304 elements. This memory pressure drives optimization (fusion, tiling).

3. **Data Dependencies**: Stage 2 depends on stage 1, stage 3 on stage 2, stage 4 on stage 3. No parallelism **between** stages, but high parallelism **within** each stage.

4. **Numerical Stability**: Naive softmax overflows for large scores. The stable formulation requires finding the maximum first—a global reduction—then computing exponentials relative to that maximum. This multi-pass pattern appears throughout ML.

**Concrete Example**. Consider a small attention instance with `seq_len=4`, `d_k=3`:

```python
Q = [[1.0, 0.0, 0.0],    # Query: looking for first dimension
     [0.0, 1.0, 0.0],    # Query: looking for second dimension
     [0.0, 0.0, 1.0],    # Query: looking for third dimension
     [0.3, 0.3, 0.3]]    # Query: looking for all dimensions equally

K = Q  # Keys match queries (self-attention)

V = [[1.0, 2.0],         # Values to retrieve
     [3.0, 4.0],
     [5.0, 6.0],
     [7.0, 8.0]]
```

**Stage 1: Scores** (Q @ K^T):

```
[[1.0, 0.0, 0.0, 0.3],   # Query 1 matches key 1 strongly
 [0.0, 1.0, 0.0, 0.3],   # Query 2 matches key 2 strongly
 [0.0, 0.0, 1.0, 0.3],   # Query 3 matches key 3 strongly
 [0.3, 0.3, 0.3, 0.27]]  # Query 4 matches all keys equally
```

The diagonal dominance (1.0 values) indicates each query strongly attends to its corresponding position—typical of self-attention without causal masking.

**Stage 2: Scaling** (scores / √3 ≈ scores / 1.732):

```
[[0.577, 0.0, 0.0, 0.173],
 [0.0, 0.577, 0.0, 0.173],
 [0.0, 0.0, 0.577, 0.173],
 [0.173, 0.173, 0.173, 0.156]]
```

Scaling prevents large dot products from pushing softmax into saturation (where gradients vanish during training).

**Stage 3: Weights** (softmax per row):

```
[[0.536, 0.154, 0.154, 0.156],   # Query 1 attends mostly to position 1
 [0.154, 0.536, 0.154, 0.156],   # Query 2 attends mostly to position 2
 [0.154, 0.154, 0.536, 0.156],   # Query 3 attends mostly to position 3
 [0.269, 0.269, 0.269, 0.193]]   # Query 4 distributes attention
```

Softmax converts scores to probabilities: each row sums to 1.0, weights are positive, higher scores get exponentially more weight.

**Stage 4: Output** (weights @ V):

```
[[2.62, 3.62],   # Weighted sum emphasizing value 1
 [4.31, 5.31],   # Weighted sum emphasizing value 2
 [5.93, 6.93],   # Weighted sum emphasizing value 3
 [4.0, 5.0]]     # Balanced weighted sum
```

Each output position is a weighted combination of all values, with weights determined by query-key similarity.

**Why We Care About These Details**. When implementing attention in MLIR, we must:
- Allocate buffers for intermediates (scores, scaled scores, weights)
- Ensure operations execute in the correct order (dependencies)
- Handle numerical edge cases (softmax stability, division by zero)
- Optimize memory traffic (fusion opportunities between stages)

The computation graph structure dictates IR design: we need operations for each stage, types representing shapes, and lowering patterns converting high-level attention to efficient low-level code. Let's design a dialect capturing this structure.

## 11.2 The Transformer Dialect: Operations for Attention

Chapter 9 taught TableGen-based dialect definition. Now we apply that knowledge to create a `transformer` dialect containing operations needed for attention. The design philosophy: operations should match attention's computational stages closely enough to enable optimization, but be general enough for reuse across different transformer architectures.

**Dialect Design Principles**. When creating domain-specific operations:

1. **Granularity**: Too fine-grained (one operation per arithmetic instruction) loses optimization opportunities. Too coarse-grained (one operation for entire attention) limits flexibility. We choose **medium granularity**: operations matching stages (matmul, softmax, transpose).

2. **Shape Polymorphism**: Operations should handle 2D and 3D tensors (batched attention) uniformly. The same `matmul` works for single-instance and batched cases.

3. **Type System**: Use memrefs for in-place computation (avoid allocations in generated code). Shapes can be dynamic (runtime-determined sequence lengths).

4. **Composability**: Operations compose to build higher-level primitives. Attention is `matmul` + `transpose` + `scale` + `softmax` + `matmul` again.

**The Transformer Dialect Definition**. From [inc/TransformerDialect.td](ch.11.Attention/inc/TransformerDialect.td):

```tablegen
def Transformer_Dialect : Dialect {
  let name = "transformer";
  let summary = "Operations for transformer attention mechanisms";
  let cppNamespace = "::mlir::transformer";

  let description = [{
    The transformer dialect provides operations for implementing
    attention mechanisms and transformer blocks. Operations are designed
    to be lowered to standard MLIR dialects (arith, scf, memref) for
    compilation to efficient code.
  }];
}
```

Standard dialect boilerplate: name, namespace, description. The key is `cppNamespace`—generated C++ code lives in `mlir::transformer` to avoid naming conflicts.

**Base Operation Class**. Like Chapter 9's NN dialect:

```tablegen
class Transformer_Op<string mnemonic, list<Trait> traits = []>
    : Op<Transformer_Dialect, mnemonic, traits>;
```

All transformer operations inherit from this, automatically inheriting the dialect and getting the `transformer.` prefix.

**Core Operations**. We define five operations matching attention's needs:

### 11.2.1 Matrix Multiplication

```tablegen
def Transformer_MatmulOp : Transformer_Op<"matmul"> {
  let summary = "Matrix multiplication";
  let description = [{
    Performs matrix multiplication: C = A @ B

    Supports 2D and 3D tensors (batched matmul).
    For 2D: (M, K) @ (K, N) -> (M, N)
    For 3D: (B, M, K) @ (B, K, N) -> (B, M, N)
  }];

  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);
  let assemblyFormat = "$lhs `,` $rhs `,` $output attr-dict `:` type($lhs) `,` type($rhs) `,` type($output)";
}
```

**Design Choices**:
- **Three operands**: Two inputs, one output (out-parameter pattern from Chapter 7). Avoids allocation inside the operation.
- **AnyMemRef**: Accepts memrefs of any rank/type. Verification (if needed) happens at lowering time.
- **Assembly format**: Custom syntax `transformer.matmul %A, %B, %C : memref<...>, memref<...>, memref<...>`. Mirrors operation semantics directly.


### 11.2.2 Transpose

```tablegen
def Transformer_TransposeOp : Transformer_Op<"transpose"> {
  let summary = "Transpose last two dimensions";
  let description = [{
    Transposes the last two dimensions of a tensor.

    For 2D: (M, N) -> (N, M)
    For 3D: (B, M, N) -> (B, N, M)
  }];

  let arguments = (ins AnyMemRef:$input, AnyMemRef:$output);
  let assemblyFormat = "$input `,` $output attr-dict `:` type($input) `,` type($output)";
}
```

Attention requires transposing K to compute Q @ K^T. Note the specification: "last two dimensions." This handles batched attention (3D tensors) automatically—batch dimension remains unchanged, inner matrix transposes.

### 11.2.3 Softmax

```tablegen
def Transformer_SoftmaxOp : Transformer_Op<"softmax"> {
  let summary = "Softmax activation along last dimension";
  let description = [{
    Applies numerically stable softmax along the last dimension:
    
    softmax(x)[i] = exp(x[i] - max(x)) / sum_j(exp(x[j] - max(x)))
  }];

  let arguments = (ins AnyMemRef:$input, AnyMemRef:$output);
  let assemblyFormat = "$input `,` $output attr-dict `:` type($input) `,` type($output)";
}
```

The description specifies **numerically stable** softmax—max subtraction prevents overflow. This is crucial: without it, exp(large_number) produces infinity, breaking attention. The implementation (Section 11.3) shows the three-pass algorithm from Chapter 6.

### 11.2.4 Element-Wise Operations

```tablegen
def Transformer_AddOp : Transformer_Op<"add"> {
  let summary = "Element-wise addition";
  let description = [{ output = lhs + rhs }];
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);
  let assemblyFormat = "$lhs `,` $rhs `,` $output attr-dict `:` type($lhs) `,` type($rhs) `,` type($output)";
}

def Transformer_MulOp : Transformer_Op<"mul"> {
  let summary = "Element-wise multiplication";
  let description = [{ output = lhs * rhs }];
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);
  let assemblyFormat = "$lhs `,` $rhs `,` $output attr-dict `:` type($lhs) `,` type($rhs) `,` type($output)";
}
```

Addition handles residual connections (Chapter 12). Multiplication implements scaling (dividing by √d_k). Both are rank-generic: work on any-shaped memrefs with compatible shapes.


## 11.3 Lowering Patterns: Leveraging the Linalg Dialect

Operations defined, we now implement lowering—converting high-level `transformer.*` operations to MLIR's **Linalg dialect**. Linalg provides structured operations for linear algebra that MLIR's optimization passes understand deeply. By lowering to linalg instead of raw loops, we get free optimizations: tiling, fusion, vectorization, parallelization.

**Design Philosophy**: The transformer dialect is a **thin wrapper** around linalg. Operations provide domain-specific names (`transformer.matmul` is clearer than `linalg.matmul` in attention code), but immediately lower to linalg for optimization. This combines usability with performance.

### 11.3.1 Matrix Multiplication Lowering

Matrix multiplication C = A @ B lowers to two linalg operations:

```mlir
linalg.fill ins(%zero : f32) outs(%C : memref<?x?xf32>)
linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
              outs(%C : memref<?x?xf32>)
```

**Zero Initialization**: `linalg.fill` sets all elements of C to zero before accumulation. Skipping this produces undefined behavior.

**Structured Operation**: `linalg.matmul` is a **named operation**—MLIR knows its semantics (`C[i,j] += A[i,k] * B[k,j]`). Optimization passes can tile it, fuse it with consumers, vectorize it, or parallelize it automatically.

**The Lowering Pattern**. From [src/TransformerToStandard.cpp](ch.11.Attention/src/TransformerToStandard.cpp):

```cpp
struct MatmulOpLowering : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op,
                                  PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value output = op.getOutput();

    // Zero-initialize output
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(0.0));
    rewriter.create<linalg::FillOp>(loc, zero, output);

    // Matrix multiplication
    rewriter.create<linalg::MatmulOp>(
        loc, ValueRange{lhs, rhs}, ValueRange{output});

    rewriter.eraseOp(op);
    return success();
  }
};
```

Compare to manual loop lowering: ~115 lines of nested `scf.for` loops with index management, load/store operations, and accumulation logic—now 11 lines. The linalg operation encapsulates all that complexity.

**Execution**: Later passes convert `linalg.matmul` to loops (via `createConvertLinalgToLoopsPass()`), generating the same nested structure—but optimization passes run **before** that conversion, applying transformations impossible with raw loops.

### 11.3.2 Transpose Lowering

Transpose swaps the last two dimensions. Linalg provides `linalg.transpose` with a permutation map:

```mlir
linalg.transpose ins(%input : memref<?x?xf32>)
                  outs(%output : memref<?x?xf32>)
                  permutation = [1, 0]
```

The permutation `[1, 0]` means "dimension 0 of output comes from dimension 1 of input, dimension 1 of output comes from dimension 0 of input"—swapping them.

**The Lowering Pattern**:

```cpp
struct TransposeOpLowering : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                  PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();

    auto inputType = cast<MemRefType>(input.getType());
    int64_t rank = inputType.getRank();

    // Permutation: swap last two dimensions
    SmallVector<int64_t> perm(rank);
    for (int64_t i = 0; i < rank; ++i)
      perm[i] = i;
    std::swap(perm[rank - 2], perm[rank - 1]);

    rewriter.create<linalg::TransposeOp>(
        loc, input, output, perm);

    rewriter.eraseOp(op);
    return success();
  }
};
```

This pattern constructs a permutation vector `[0, 1, ..., rank-3, rank-1, rank-2]` (identity except swapping the last two dimensions). Works for 2D (rank 2) and 3D (rank 3, batched attention) tensors uniformly.

Compare to manual lowering: ~50 lines of nested loops with careful index swapping—now 20 lines generating a declarative operation.

### 11.3.3 Softmax Lowering: Reductions and Broadcasting

Softmax combines reductions (finding max, summing) with element-wise operations (exp, division). Linalg handles both via `linalg.reduce` and `linalg.generic`.

**The Algorithm** (per row):

```
1. max_val = reduce_max(input along last dimension)
2. shifted = input - max_val  (broadcast max_val)
3. exp_vals = exp(shifted)
4. sum_exp = reduce_sum(exp_vals along last dimension)
5. output = exp_vals / sum_exp  (broadcast sum_exp)
```

**Step 1: Find Maximum**

```mlir
linalg.reduce { arith.maximumf }
  ins(%input : memref<?x?xf32>)
  outs(%max_vals : memref<?xf32>)
  dimensions = [1]
```

Reduces along dimension 1 (last dimension for 2D), applying `arith.maximumf` to accumulate. Result: one max per row.

**Step 2: Subtract Max (Broadcasting)**

```mlir
linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0)>,      // max_vals: broadcast d0
    affine_map<(d0, d1) -> (d0, d1)>,  // input: full tensor
    affine_map<(d0, d1) -> (d0, d1)>   // output: full tensor
  ],
  iterator_types = ["parallel", "parallel"]
}
ins(%max_vals, %input : memref<?xf32>, memref<?x?xf32>)
outs(%shifted : memref<?x?xf32>) {
^bb0(%max: f32, %inp: f32, %out: f32):
  %sub = arith.subf %inp, %max : f32
  linalg.yield %sub : f32
}
```

The affine map `(d0, d1) -> (d0)` **broadcasts** `max_vals`: for each `(i, j)`, `max_vals[i]` repeats across all `j`. This implements `shifted[i, j] = input[i, j] - max_vals[i]`.

**Steps 3-5**: Follow the same pattern—reduce for sum, generic for exp and division with broadcasting.

**The Lowering Pattern** (abbreviated):

```cpp
struct SoftmaxOpLowering : public OpRewritePattern<SoftmaxOp> {
  LogicalResult matchAndRewrite(SoftmaxOp op,
                                  PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();
    
    auto inputType = cast<MemRefType>(input.getType());
    auto shape = inputType.getShape();
    int64_t rank = inputType.getRank();
    
    // Allocate temporaries for max and sum
    auto reducedShape = SmallVector<int64_t>(shape.begin(), shape.end() - 1);
    auto maxType = MemRefType::get(reducedShape, rewriter.getF32Type());
    Value maxVals = rewriter.create<memref::AllocOp>(loc, maxType);
    
    // Step 1: Reduce max along last dimension
    rewriter.create<linalg::ReduceOp>(
        loc, ValueRange{input}, ValueRange{maxVals},
        ArrayRef<int64_t>{rank - 1},
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value max = b.create<arith::MaximumFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, max);
        });
    
    // Step 2: Subtract max (broadcasting) and compute exp
    // ... (linalg.generic with affine maps for broadcasting)
    
    // Step 3: Reduce sum
    // ... (linalg.reduce with addf)
    
    // Step 4: Divide by sum (broadcasting)
    // ... (linalg.generic with broadcasting)
    
    rewriter.eraseOp(op);
    return success();
  }
};
```

Compare to manual lowering: ~105 lines of three-pass nested loops with temporary buffers—now 75 lines of declarative linalg operations. More importantly, linalg's structured form enables **fusion**: later passes can merge operations, eliminating intermediate buffers.

### 11.3.4 Element-Wise Operations

Addition and multiplication are element-wise: `output[i, j] = lhs[i, j] + rhs[i, j]`. Linalg generic handles these:

```mlir
linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,  // lhs
    affine_map<(d0, d1) -> (d0, d1)>,  // rhs
    affine_map<(d0, d1) -> (d0, d1)>   // output
  ],
  iterator_types = ["parallel", "parallel"]
}
ins(%lhs, %rhs : memref<?x?xf32>, memref<?x?xf32>)
outs(%output : memref<?x?xf32>) {
^bb0(%l: f32, %r: f32, %out: f32):
  %sum = arith.addf %l, %r : f32
  linalg.yield %sum : f32
}
```

**Identity Maps**: All three operands use `(d0, d1) -> (d0, d1)`—no broadcasting, direct element correspondence.

**Parallel Iteration**: Both dimensions are `"parallel"`, meaning iterations are independent (no carried dependencies). MLIR can parallelize this automatically.

**The Lowering Patterns**:

```cpp
struct AddOpLowering : public OpRewritePattern<AddOp> {
  LogicalResult matchAndRewrite(AddOp op,
                                  PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value output = op.getOutput();
    
    auto outputType = cast<MemRefType>(output.getType());
    int64_t rank = outputType.getRank();
    
    // Build identity affine maps for all operands
    SmallVector<AffineMap> indexingMaps;
    for (int i = 0; i < 3; ++i)
      indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));
    
    SmallVector<utils::IteratorType> iterators(
        rank, utils::IteratorType::parallel);
    
    rewriter.create<linalg::GenericOp>(
        loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{output},
        indexingMaps, iterators,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });
    
    rewriter.eraseOp(op);
    return success();
  }
};
```

Multiplication follows identically, replacing `arith.AddFOp` with `arith.MulFOp`.

Compare to manual lowering: ~50 lines of recursive loop generation for arbitrary ranks—now 25 lines of generic operation. Rank-polymorphism comes naturally with linalg.

### 11.3.5 From Linalg to Loops: The Pass Pipeline

Linalg operations don't execute directly—they lower to loops. The pass pipeline:

```cpp
pm.addPass(createLowerTransformerToStandardPass());  // transformer -> linalg
pm.addPass(createConvertLinalgToLoopsPass());        // linalg -> scf.for
pm.addPass(createConvertSCFToCFPass());              // scf -> control flow
// ... remaining passes to LLVM IR
```

**Key Insight**: Optimization passes run **between** transformer→linalg and linalg→loops. Linalg's structured form enables:

- **Tiling**: Break operations into cache-friendly blocks
- **Fusion**: Merge producer-consumer operations, eliminating loads/stores
- **Vectorization**: Generate SIMD instructions (AVX, NEON)
- **Parallelization**: Distribute across threads

Manual loop lowering bypasses these. By lowering to linalg, we get these optimizations **for free** as MLIR evolves.

**Registering the Linalg Dialect**. From [src/bindings.cpp](ch.11.Attention/src/bindings.cpp):

```cpp
context_.getOrLoadDialect<linalg::LinalgDialect>();
```

The linalg dialect must be registered before lowering patterns can emit linalg operations. Forgetting this produces "unknown dialect" errors at runtime.

**Lowering Pass Organization**. All patterns register in a single pass:

```cpp
void populateLowerTransformerToStandardPatterns(RewritePatternSet &patterns) {
  patterns.add<MatmulOpLowering,
               TransposeOpLowering,
               SoftmaxOpLowering,
               AddOpLowering,
               MulOpLowering>(patterns.getContext());
}

std::unique_ptr<Pass> createLowerTransformerToStandardPass() {
  return std::make_unique<LowerTransformerToStandardPass>();
}
```

The pass applies all patterns in a single traversal. MLIR's pattern rewriting framework handles orchestration: which patterns to try, in what order, how to handle failures. We provide the logic; MLIR provides the infrastructure.

## 11.4 Python API: Building Computation Graphs

Lowering patterns convert operations to loops, but where do those operations come from? Users don't write MLIR IR directly—they use Python APIs building computation graphs symbolically. This section examines Chapter 11's Python API design, showing how operator overloading and deferred execution create PyTorch-like ergonomics while generating MLIR underneath.

**The Tensor Abstraction**. Python users work with a `Tensor` class wrapping NumPy arrays:

```python
import ch11
import numpy as np

Q = ch11.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
K = ch11.Tensor(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))

# Build computation graph (no execution yet)
result_tensor = ch11.attention(Q, K, K)

# Compile and execute
output = ch11.forward(result_tensor)  # Returns NumPy array
```

The `Tensor` class doesn't compute immediately. Instead, it records operations in a **computation graph**—a directed acyclic graph (DAG) where nodes represent operations and edges represent data dependencies. This deferred execution pattern matches PyTorch's JIT and TensorFlow's graph mode: build a symbolic representation, compile once, execute many times.

**Computation Graph Representation**. From [src/bindings.cpp](ch.11.Attention/src/bindings.cpp):

```cpp
enum class OpType {
  Input,      // Leaf node: data provided by user
  Matmul,     // Binary operation: lhs @ rhs
  Add,        // Binary operation: lhs + rhs
  Transpose,  // Unary operation: transpose(input)
  Softmax,    // Unary operation: softmax(input)
  Scale       // Unary operation: input * scale_factor
};

struct GraphNode {
  OpType type;
  std::vector<std::shared_ptr<GraphNode>> inputs;  // Dependencies
  py::array_t<float> data;                          // For Input nodes
  float scale_factor = 1.0f;                        // For Scale nodes
  std::vector<int64_t> shape;                       // Output shape

  GraphNode(OpType t) : type(t) {}
};
```

Each node stores:
- **Type**: What operation this represents
- **Inputs**: Pointers to input nodes (empty for `Input` nodes, non-empty for operations)
- **Data**: For leaf nodes (`Input`), the actual NumPy array
- **Shape**: The output shape of this operation (inferred from inputs)

**Graph Construction API**. The `Tensor` class provides methods building nodes:

```cpp
class Tensor {
public:
  std::shared_ptr<GraphNode> node;

  Tensor(py::array_t<float> data) {
    node = std::make_shared<GraphNode>(OpType::Input);
    node->data = data;
    auto buf = data.request();
    node->shape.resize(buf.ndim);
    for (int i = 0; i < buf.ndim; i++) {
      node->shape[i] = static_cast<int64_t>(buf.shape[i]);
    }
  }

  Tensor(std::shared_ptr<GraphNode> n) : node(n) {}

  // Operator overloading for arithmetic
  Tensor add(const Tensor& other) const {
    auto result_node = std::make_shared<GraphNode>(OpType::Add);
    result_node->inputs = {node, other.node};
    result_node->shape = node->shape;  // Assumes compatible shapes
    return Tensor(result_node);
  }

  Tensor mul(const Tensor& other) const {
    auto result_node = std::make_shared<GraphNode>(OpType::Mul);
    result_node->inputs = {node, other.node};
    result_node->shape = node->shape;
    return Tensor(result_node);
  }

  const std::vector<int64_t>& shape() const { return node->shape; }
};
```

Each method creates a new node pointing to input nodes, returns a new `Tensor` wrapping that node. No computation happens—we're just building the graph.

**Python Bindings**. pybind11 exposes these methods to Python:

```cpp
py::class_<Tensor>(m, "Tensor")
  .def(py::init<py::array_t<float>>())
  .def("__add__", &Tensor::add)
  .def("__mul__", &Tensor::mul)
  .def("shape", &Tensor::shape);
```

Python's `a + b` calls `Tensor.__add__()`, which calls the C++ `add()` method. This provides natural syntax hiding graph construction complexity.

**Higher-Level Operations**. Operations like matmul, transpose, softmax are module-level functions:

```cpp
Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
  auto result_node = std::make_shared<GraphNode>(OpType::Matmul);
  result_node->inputs = {lhs.node, rhs.node};
  
  // Shape inference: (M, K) @ (K, N) -> (M, N)
  result_node->shape = {lhs.shape()[0], rhs.shape()[1]};
  return Tensor(result_node);
}

Tensor transpose(const Tensor& input) {
  auto result_node = std::make_shared<GraphNode>(OpType::Transpose);
  result_node->inputs = {input.node};
  
  // Shape inference: (M, N) -> (N, M)
  auto& in_shape = input.shape();
  result_node->shape = {in_shape[1], in_shape[0]};
  return Tensor(result_node);
}

Tensor softmax(const Tensor& input) {
  auto result_node = std::make_shared<GraphNode>(OpType::Softmax);
  result_node->inputs = {input.node};
  result_node->shape = input.shape();  // Softmax preserves shape
  return Tensor(result_node);
}

Tensor scale(const Tensor& input, float factor) {
  auto result_node = std::make_shared<GraphNode>(OpType::Scale);
  result_node->inputs = {input.node};
  result_node->scale_factor = factor;
  result_node->shape = input.shape();
  return Tensor(result_node);
}
```

These functions follow the same pattern: create node, set inputs, infer output shape, return tensor. Shape inference is crucial—we need shapes to generate correct MLIR IR (allocating buffers, setting loop bounds).

**Composing Attention**. With primitives defined, attention becomes straightforward:

```cpp
Tensor attention(const Tensor& Q, const Tensor& K, const Tensor& V) {
  // scores = Q @ K^T
  auto K_T = transpose(K);
  auto scores = matmul(Q, K_T);
  
  // scaled_scores = scores / sqrt(d_k)
  int64_t d_k = Q.shape()[1];  // Key dimension
  float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
  auto scaled_scores = scale(scores, scale_factor);
  
  // attn_weights = softmax(scaled_scores)
  auto attn_weights = softmax(scaled_scores);
  
  // output = attn_weights @ V
  auto output = matmul(attn_weights, V);
  
  return output;
}
```

This is **compositional**: each operation returns a tensor, which feeds into the next operation. The final graph has 6 nodes:
1. Input (Q)
2. Input (K)  
3. Input (V)
4. Transpose (K)
5. Matmul (Q, K^T)
6. Scale (scores)
7. Softmax (scaled scores)
8. Matmul (weights, V)

Plus three more input nodes (Q, K, V). The graph structure encodes dependencies: matmul depends on transpose, softmax depends on scale, final matmul depends on softmax.

**Why Graphs?** You might wonder: why not compile each operation individually? Several reasons:

1. **Optimization Opportunities**: With the full graph, we can fuse operations (Chapter 10's techniques), reorder for better cache locality, eliminate redundant computation.

2. **Memory Planning**: Knowing all operations upfront lets us plan buffer allocation—reuse buffers for intermediate results when possible.

3. **Batching**: If multiple inputs flow through the same graph, compile once, execute repeatedly (amortizing compilation cost).

4. **Debugging**: Graph visualization shows computation structure, helping identify errors (wrong operation ordering, shape mismatches).

Chapter 11's implementation is simple: compile on every `forward()` call. Production systems (PyTorch JIT, TensorFlow) cache compiled graphs, only recompiling when graph structure changes. But the principle is identical.

## 11.5 JIT Compilation: From Graphs to Native Code

Computation graph built, we now compile it to executable code. This section traces the compilation pipeline: graph → MLIR IR → LLVM IR → native machine code → execution via libffi. Each stage transforms the representation, eventually producing code running on the CPU.

**The forward() Entry Point**. Python users call:

```python
output = ch11.forward(result_tensor)
```

The C++ implementation:

```cpp
py::array_t<float> forward(const Tensor& input) {
  // Extract computation graph
  auto& graph = input.node;
  
  // Build MLIR module containing the computation
  MLIRContext& context = getCompiler().getContext();
  OpBuilder builder(&context);
  auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
  
  // Convert graph to MLIR function
  builder.setInsertionPointToEnd(module.getBody());
  buildGraphFunction(builder, module, graph);
  
  // Compile to native code
  void* funcPtr = getCompiler().compileAndGetFunctionPtr(module, "graph_func");
  
  // Execute via libffi
  py::array_t<float> result = executeFunctionViaLibffi(funcPtr, graph);
  
  return result;
}
```

Four stages: graph → MLIR, MLIR → LLVM, LLVM → native, native → execution. Let's examine each.

**Stage 1: Graph to MLIR**. The `buildGraphFunction()` traverses the graph, emitting MLIR operations:

```cpp
void buildGraphFunction(OpBuilder& builder, ModuleOp module,
                         std::shared_ptr<GraphNode> outputNode) {
  auto loc = builder.getUnknownLoc();
  
  // Collect all input nodes
  std::vector<std::shared_ptr<GraphNode>> inputs;
  std::unordered_map<GraphNode*, Value> nodeToValue;
  collectInputs(outputNode, inputs);
  
  // Build function signature
  SmallVector<Type> argTypes;
  for (auto& input : inputs) {
    auto memrefType = MemRefType::get(input->shape, builder.getF32Type());
    argTypes.push_back(memrefType);
  }
  
  // Output type
  auto outputType = MemRefType::get(outputNode->shape, builder.getF32Type());
  argTypes.push_back(outputType);
  
  // Create function
  auto funcType = builder.getFunctionType(argTypes, {});
  auto func = builder.create<func::FuncOp>(loc, "graph_func", funcType);
  auto* entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  
  // Map input nodes to function arguments
  for (size_t i = 0; i < inputs.size(); ++i) {
    nodeToValue[inputs[i].get()] = entryBlock->getArgument(i);
  }
  Value outputBuffer = entryBlock->getArgument(inputs.size());
  
  // Emit operations for each node
  emitNode(builder, outputNode, nodeToValue, outputBuffer);
  
  builder.create<func::ReturnOp>(loc);
  module.push_back(func);
}
```

**Key Steps**:

1. **Collect Inputs**: Traverse graph depth-first, identifying all `Input` nodes. These become function arguments.

2. **Build Signature**: Each input becomes a memref argument. The output (where results are written) is the last argument.

3. **Create Function**: Build `func.func @graph_func(...)` with the computed signature.

4. **Emit Operations**: Walk the graph, emitting transformer operations in topological order (dependencies before dependents).

**Emitting Operations**. The `emitNode()` function is recursive:

```cpp
Value emitNode(OpBuilder& builder, std::shared_ptr<GraphNode> node,
               std::unordered_map<GraphNode*, Value>& nodeToValue,
               Value outputBuffer) {
  // Check if already emitted
  if (nodeToValue.count(node.get())) {
    return nodeToValue[node.get()];
  }
  
  auto loc = builder.getUnknownLoc();
  
  switch (node->type) {
    case OpType::Input:
      // Already in nodeToValue (mapped to function arguments)
      assert(false && "Input nodes should be pre-mapped");
      
    case OpType::Matmul: {
      // Emit input nodes first (recursion)
      Value lhs = emitNode(builder, node->inputs[0], nodeToValue, outputBuffer);
      Value rhs = emitNode(builder, node->inputs[1], nodeToValue, outputBuffer);
      
      // Allocate buffer for result
      auto resultType = MemRefType::get(node->shape, builder.getF32Type());
      Value result = builder.create<memref::AllocOp>(loc, resultType);
      
      // Emit matmul operation
      builder.create<MatmulOp>(loc, lhs, rhs, result);
      
      nodeToValue[node.get()] = result;
      return result;
    }
    
    case OpType::Transpose: {
      Value input = emitNode(builder, node->inputs[0], nodeToValue, outputBuffer);
      auto resultType = MemRefType::get(node->shape, builder.getF32Type());
      Value result = builder.create<memref::AllocOp>(loc, resultType);
      builder.create<TransposeOp>(loc, input, result);
      nodeToValue[node.get()] = result;
      return result;
    }
    
    case OpType::Softmax: {
      Value input = emitNode(builder, node->inputs[0], nodeToValue, outputBuffer);
      auto resultType = MemRefType::get(node->shape, builder.getF32Type());
      Value result = builder.create<memref::AllocOp>(loc, resultType);
      builder.create<SoftmaxOp>(loc, input, result);
      nodeToValue[node.get()] = result;
      return result;
    }
    
    // ... other cases ...
  }
}
```

The pattern: emit dependencies recursively, allocate buffer for this operation's result, emit the transformer operation, cache the result value. This ensures topological ordering—operations emit only after their inputs are available.

**Generated MLIR**. For the attention graph, the emitted IR looks like:

```mlir
func.func @graph_func(%Q: memref<4x8xf32>, %K: memref<4x8xf32>,
                       %V: memref<4x8xf32>, %output: memref<4x8xf32>) {
  %K_T = memref.alloc() : memref<8x4xf32>
  transformer.transpose %K, %K_T : memref<4x8xf32>, memref<8x4xf32>
  
  %scores = memref.alloc() : memref<4x4xf32>
  transformer.matmul %Q, %K_T, %scores : memref<4x8xf32>, memref<8x4xf32>, memref<4x4xf32>
  
  %scale_buf = memref.alloc() : memref<4x4xf32>
  %scale = arith.constant 0.353553 : f32  // 1/sqrt(8)
  // ... scale operation ...
  
  %attn = memref.alloc() : memref<4x4xf32>
  transformer.softmax %scale_buf, %attn : memref<4x4xf32>, memref<4x4xf32>
  
  transformer.matmul %attn, %V, %output : memref<4x4xf32>, memref<4x8xf32>, memref<4x8xf32>
  
  func.return
}
```

High-level transformer operations, explicit buffer allocations, correct dependency ordering. This IR is ready for lowering.

**Stage 2: Lowering to LLVM**. The compiler applies passes from Section 11.3:

```cpp
bool TransformerCompiler::lowerToLLVM(ModuleOp module) {
  PassManager pm(&context_);

  // Lower transformer dialect to linalg
  pm.addNestedPass<func::FuncOp>(createLowerTransformerToStandardPass());
  
  // Lower linalg to loops
  pm.addPass(createConvertLinalgToLoopsPass());
  
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // Lower standard dialects to LLVM
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertMathToLibmPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  return succeeded(pm.run(module));
}
```

First, transformer ops → linalg (Section 11.3's patterns). Then, linalg → scf.for loops (via `createConvertLinalgToLoopsPass()`). Finally, loops → LLVM IR (standard MLIR passes). Canonicalization and CSE (common subexpression elimination) clean up between major transformations. After this pipeline, the module contains only LLVM dialect operations—pointers, branches, arithmetic instructions.

**Stage 3: LLVM IR to Native Code**. MLIR's ExecutionEngine compiles LLVM dialect to machine code:

```cpp
void* TransformerCompiler::compileAndGetFunctionPtr(ModuleOp module,
                                                      const std::string& funcName) {
  registerBuiltinDialectTranslation(*module.getContext());
  registerLLVMDialectTranslation(*module.getContext());

  if (!lowerToLLVM(module)) {
    llvm::errs() << "Failed to lower to LLVM dialect\n";
    return nullptr;
  }

  ExecutionEngineOptions options;
  auto transformer = makeOptimizingTransformer(3, 0, nullptr);
  options.transformer = std::move(transformer);

  auto maybeEngine = ExecutionEngine::create(module, options);
  if (!maybeEngine) {
    llvm::errs() << "Failed to create ExecutionEngine\n";
    return nullptr;
  }

  auto engine = std::move(*maybeEngine);
  auto expectedFPtr = engine->lookup(funcName);
  if (!expectedFPtr) {
    llvm::errs() << "Failed to lookup function: " << funcName << "\n";
    return nullptr;
  }

  // Keep engine alive
  engines_.emplace_back(engine.release());
  return reinterpret_cast<void*>(*expectedFPtr);
}
```

The `ExecutionEngine::create()` call invokes LLVM's JIT compiler (OrcJIT). It parses LLVM IR, optimizes (the `makeOptimizingTransformer(3, ...)` sets optimization level 3—aggressive), generates machine code for the target architecture, loads it into memory. The `lookup()` finds the function's entry point address—a raw pointer to executable code.

**Stage 4: Execution via libffi**. With a function pointer, how do we call it? The signature is:

```c
void graph_func(float* Q_data, intptr_t Q_offset, intptr_t Q_size0, intptr_t Q_size1, intptr_t Q_stride0, intptr_t Q_stride1,
                float* K_data, intptr_t K_offset, ...,
                float* V_data, intptr_t V_offset, ...,
                float* out_data, intptr_t out_offset, ...);
```

MLIR's memref calling convention (Chapter 7) passes memrefs as expanded arguments: pointer, offset, sizes, strides. For 2D memrefs, that's 7 arguments per memref. Manually calling this is tedious and error-prone. Enter **libffi**:

```cpp
py::array_t<float> executeFunctionViaLibffi(void* funcPtr,
                                             std::shared_ptr<GraphNode> outputNode) {
  // Collect input data arrays
  std::vector<py::array_t<float>> inputs;
  collectInputData(outputNode, inputs);
  
  // Allocate output array
  py::array_t<float> output(outputNode->shape);
  
  // Prepare libffi call
  std::vector<ffi_type*> arg_types;
  std::vector<void*> arg_values;
  
  // Marshal each input memref
  for (auto& arr : inputs) {
    marshal_memref_2d(arg_types, arg_values, arr);
  }
  
  // Marshal output memref
  marshal_memref_2d(arg_types, arg_values, output);
  
  // Build libffi CIF (Call Interface)
  ffi_cif cif;
  if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, arg_types.size(),
                    &ffi_type_void, arg_types.data()) != FFI_OK) {
    throw std::runtime_error("ffi_prep_cif failed");
  }
  
  // Call function
  ffi_call(&cif, FFI_FN(funcPtr), nullptr, arg_values.data());
  
  return output;
}
```

The `marshal_memref_2d()` function expands NumPy arrays into memref descriptor arguments:

```cpp
void marshal_memref_2d(std::vector<ffi_type*>& arg_types,
                        std::vector<void*>& arg_values,
                        py::array_t<float> arr) {
  auto buf = arr.request();
  float* data = static_cast<float*>(buf.ptr);
  
  // Allocate persistent storage for arguments
  static std::vector<void*> persistent_args;
  persistent_args.push_back(data);
  persistent_args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(0)));  // offset
  persistent_args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[0])));
  persistent_args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[1])));
  persistent_args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[1])));  // stride0
  persistent_args.push_back(reinterpret_cast<void*>(static_cast<intptr_t>(1)));   // stride1
  
  // Add types
  arg_types.push_back(&ffi_type_pointer);  // data
  arg_types.push_back(&ffi_type_sint64);   // offset
  arg_types.push_back(&ffi_type_sint64);   // size0
  arg_types.push_back(&ffi_type_sint64);   // size1
  arg_types.push_back(&ffi_type_sint64);   // stride0
  arg_types.push_back(&ffi_type_sint64);   // stride1
  
  // Add value pointers
  for (size_t i = persistent_args.size() - 6; i < persistent_args.size(); ++i) {
    arg_values.push_back(&persistent_args[i]);
  }
}
```

libffi handles calling conventions, register allocation, stack management. We provide types and values; it invokes the function correctly.

**End-to-End Flow**. Putting it all together:

```
Python: ch11.forward(attention_tensor)
  ↓
C++: Build MLIR from computation graph
  ↓
MLIR: transformer.matmul, transformer.softmax, ...
  ↓
Lowering: SCF loops, arith operations, memref accesses
  ↓
LLVM: Load/store instructions, floating-point arithmetic, branches
  ↓
JIT: Compile to x86-64 machine code
  ↓
libffi: Call with NumPy array data
  ↓
Execution: Native code runs on CPU
  ↓
Python: Returns NumPy array with results
```

Each stage is modular: replace the lowering patterns, and code generation changes; swap ExecutionEngine for a different backend, and target architecture changes. MLIR's infrastructure handles the plumbing.

## 11.6 Composing Scaled Dot-Product Attention

With lowering and JIT compilation established, we can now execute complete attention. This section demonstrates composing primitives into scaled dot-product attention, tracing both the Python API and generated MLIR IR. The key insight: complex operations emerge from simple compositions—compiler infrastructure handles optimization and execution.

**Python Interface**. From the user's perspective, attention is a single function call:

```python
import ch11
import numpy as np

# Create inputs (4 tokens, 8-dimensional embeddings)
Q = ch11.Tensor(np.random.randn(4, 8).astype(np.float32))
K = ch11.Tensor(np.random.randn(4, 8).astype(np.float32))
V = ch11.Tensor(np.random.randn(4, 8).astype(np.float32))

# Build attention computation graph
result = ch11.attention(Q, K, V)

# Compile and execute
output = ch11.forward(result)  # Returns NumPy array (4, 8)

print(output.shape)  # (4, 8)
print(output)        # Weighted combination of V vectors
```

The `ch11.attention()` function (from Section 11.4) builds a 6-node computation graph:

```cpp
Tensor attention(const Tensor& Q, const Tensor& K, const Tensor& V) {
  auto K_T = transpose(K);                     // (8, 4)
  auto scores = matmul(Q, K_T);                // (4, 4)
  
  int64_t d_k = Q.shape()[1];
  float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
  auto scaled = scale(scores, scale_factor);   // (4, 4)
  
  auto weights = softmax(scaled);              // (4, 4)
  auto output = matmul(weights, V);            // (4, 8)
  
  return output;
}
```

**Generated MLIR IR**. Section 11.5's IRBuilder produces:

```mlir
func.func @graph_func(%Q: memref<4x8xf32>, %K: memref<4x8xf32>, 
                       %V: memref<4x8xf32>, %output: memref<4x8xf32>) {
  // Step 1: Transpose K
  %K_T = memref.alloc() : memref<8x4xf32>
  transformer.transpose %K, %K_T : memref<4x8xf32>, memref<8x4xf32>
  
  // Step 2: Compute Q @ K^T
  %scores = memref.alloc() : memref<4x4xf32>
  transformer.matmul %Q, %K_T, %scores : memref<4x8xf32>, memref<8x4xf32>, memref<4x4xf32>
  
  // Step 3: Scale by 1/sqrt(d_k)
  %scale_buf = memref.alloc() : memref<4x4xf32>
  %scale_factor = arith.constant 0.353553 : f32  // 1/sqrt(8)
  transformer.mul %scores, %scale_factor, %scale_buf : memref<4x4xf32>, f32, memref<4x4xf32>
  
  // Step 4: Apply softmax
  %attn_weights = memref.alloc() : memref<4x4xf32>
  transformer.softmax %scale_buf, %attn_weights : memref<4x4xf32>, memref<4x4xf32>
  
  // Step 5: Multiply attention weights by V
  transformer.matmul %attn_weights, %V, %output : memref<4x4xf32>, memref<4x8xf32>, memref<4x8xf32>
  
  func.return
}
```

Five transformer operations, four intermediate buffers. This is the **before-lowering** IR—still high-level, still compositional. 

**Two-Stage Lowering**: Section 11.3's patterns lower transformer operations to **linalg** operations first, then MLIR's `createConvertLinalgToLoopsPass()` converts linalg to scf.for loops. The intermediate linalg representation enables optimization passes (tiling, fusion, vectorization) before final loop generation.

**After First Lowering** (transformer → linalg, abbreviated):

```mlir
func.func @graph_func(%Q: memref<4x8xf32>, %K: memref<4x8xf32>,
                       %V: memref<4x8xf32>, %output: memref<4x8xf32>) {
  %K_T = memref.alloc() : memref<8x4xf32>
  
  // Transpose: structured operation with permutation
  linalg.transpose ins(%K : memref<4x8xf32>)
                   outs(%K_T : memref<8x4xf32>)
                   permutation = [1, 0]
  
  // Matmul: Q @ K_T -> scores
  %scores = memref.alloc() : memref<4x4xf32>
  %zero = arith.constant 0.0 : f32
  linalg.fill ins(%zero : f32) outs(%scores : memref<4x4xf32>)
  linalg.matmul ins(%Q, %K_T : memref<4x8xf32>, memref<8x4xf32>)
                outs(%scores : memref<4x4xf32>)
  
  // Scale: element-wise multiply via linalg.generic
  %scale_buf = memref.alloc() : memref<4x4xf32>
  %scale_factor = arith.constant 0.353553 : f32
  linalg.generic { /* identity maps, parallel iterators */ }
    ins(%scores, %scale_factor : memref<4x4xf32>, f32)
    outs(%scale_buf : memref<4x4xf32>) {
    ^bb0(%score: f32, %factor: f32, %out: f32):
      %scaled = arith.mulf %score, %factor : f32
      linalg.yield %scaled : f32
  }
  
  // Softmax: reduce max, generic exp, reduce sum, generic divide
  %attn_weights = memref.alloc() : memref<4x4xf32>
  %max_vals = memref.alloc() : memref<4xf32>
  linalg.reduce { arith.maximumf }
    ins(%scale_buf : memref<4x4xf32>)
    outs(%max_vals : memref<4xf32>)
    dimensions = [1]
  
  // ... (exp, sum, divide steps via linalg.generic)
  
  // Final matmul: attn_weights @ V -> output
  linalg.fill ins(%zero : f32) outs(%output : memref<4x8xf32>)
  linalg.matmul ins(%attn_weights, %V : memref<4x4xf32>, memref<4x8xf32>)
                outs(%output : memref<4x8xf32>)
  
  func.return
}
```

This linalg representation is **semantically meaningful** to MLIR's optimizer—it knows `linalg.matmul` performs matrix multiplication, enabling transformations like tiling or fusion that raw loops wouldn't support.

**After Second Lowering** (linalg → scf.for loops, abbreviated):

```mlir
func.func @graph_func(%Q: memref<4x8xf32>, %K: memref<4x8xf32>,
                       %V: memref<4x8xf32>, %output: memref<4x8xf32>) {
  %K_T = memref.alloc() : memref<8x4xf32>
  
  // Transpose: swap indices when copying
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c8 step %c1 {
      %val = memref.load %K[%i, %j] : memref<4x8xf32>
      memref.store %val, %K_T[%j, %i] : memref<8x4xf32>
    }
  }
  
  // Matmul: Q @ K_T -> scores
  %scores = memref.alloc() : memref<4x4xf32>
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      %zero = arith.constant 0.0 : f32
      %sum = scf.for %k = %c0 to %c8 step %c1 iter_args(%s = %zero) -> (f32) {
        %a = memref.load %Q[%i, %k] : memref<4x8xf32>
        %b = memref.load %K_T[%k, %j] : memref<8x4xf32>
        %prod = arith.mulf %a, %b : f32
        %new_sum = arith.addf %s, %prod : f32
        scf.yield %new_sum : f32
      }
      memref.store %sum, %scores[%i, %j] : memref<4x4xf32>
    }
  }
  
  // Scale: element-wise multiply by 0.353553
  %scale_buf = memref.alloc() : memref<4x4xf32>
  %scale_factor = arith.constant 0.353553 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      %val = memref.load %scores[%i, %j] : memref<4x4xf32>
      %scaled = arith.mulf %val, %scale_factor : f32
      memref.store %scaled, %scale_buf[%i, %j] : memref<4x4xf32>
    }
  }
  
  // Softmax: three-pass algorithm (max, exp, normalize)
  %attn_weights = memref.alloc() : memref<4x4xf32>
  %max_buf = memref.alloc() : memref<4xf32>
  %sum_buf = memref.alloc() : memref<4xf32>
  
  // Pass 1: Find max per row
  %neg_inf = arith.constant 0xFF800000 : f32  // -inf
  scf.for %i = %c0 to %c4 step %c1 {
    %max = scf.for %j = %c0 to %c4 step %c1 iter_args(%m = %neg_inf) -> (f32) {
      %val = memref.load %scale_buf[%i, %j] : memref<4x4xf32>
      %new_max = arith.maximumf %m, %val : f32
      scf.yield %new_max : f32
    }
    memref.store %max, %max_buf[%i] : memref<4xf32>
  }
  
  // Pass 2: Compute exp(x - max) and sum
  %zero_f = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    %max = memref.load %max_buf[%i] : memref<4xf32>
    %sum = scf.for %j = %c0 to %c4 step %c1 iter_args(%s = %zero_f) -> (f32) {
      %val = memref.load %scale_buf[%i, %j] : memref<4x4xf32>
      %adjusted = arith.subf %val, %max : f32
      %exp_val = math.exp %adjusted : f32
      memref.store %exp_val, %attn_weights[%i, %j] : memref<4x4xf32>  // Temporary storage
      %new_sum = arith.addf %s, %exp_val : f32
      scf.yield %new_sum : f32
    }
    memref.store %sum, %sum_buf[%i] : memref<4xf32>
  }
  
  // Pass 3: Normalize
  scf.for %i = %c0 to %c4 step %c1 {
    %sum = memref.load %sum_buf[%i] : memref<4xf32>
    scf.for %j = %c0 to %c4 step %c1 {
      %exp_val = memref.load %attn_weights[%i, %j] : memref<4x4xf32>
      %normalized = arith.divf %exp_val, %sum : f32
      memref.store %normalized, %attn_weights[%i, %j] : memref<4x4xf32>
    }
  }
  
  // Final matmul: attn_weights @ V -> output
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c8 step %c1 {
      %zero = arith.constant 0.0 : f32
      %sum = scf.for %k = %c0 to %c4 step %c1 iter_args(%s = %zero) -> (f32) {
        %a = memref.load %attn_weights[%i, %k] : memref<4x4xf32>
        %b = memref.load %V[%k, %j] : memref<4x8xf32>
        %prod = arith.mulf %a, %b : f32
        %new_sum = arith.addf %s, %prod : f32
        scf.yield %new_sum : f32
      }
      memref.store %sum, %output[%i, %j] : memref<4x8xf32>
    }
  }
  
  func.return
}
```

**From 5 Operations to ~100 Lines of Loops**. The transformer dialect's five operations expanded into:
- 2 nested loops (transpose)
- 3 nested loops (matmul for scores)
- 2 nested loops (scaling)
- 6 nested loops (softmax's three passes, 2 loops each)
- 3 nested loops (matmul for output)

16 loops total, plus ~20 memory allocations (intermediate buffers). This is the code actually running—explicit iteration, explicit arithmetic, explicit memory accesses.

**Memory Usage**. For seq_len=4, d_k=8:
- K_T: 8×4 = 32 floats = 128 bytes
- scores: 4×4 = 16 floats = 64 bytes
- scale_buf: 4×4 = 16 floats = 64 bytes
- attn_weights: 4×4 = 16 floats = 64 bytes
- max_buf: 4 floats = 16 bytes
- sum_buf: 4 floats = 16 bytes
- **Total**: 352 bytes

Small. But for seq_len=512, d_k=64:
- K_T: 512×64 = 32,768 floats = 128 KB
- scores: 512×512 = 262,144 floats = 1 MB
- scale_buf: 1 MB
- attn_weights: 1 MB
- **Total**: ~3.3 MB per attention operation

Multiply by number of attention heads (8-12 typical), multiply by layers (12-24 typical), and memory becomes non-trivial. Chapter 14 discusses optimizations (fusing operations, eliminating intermediates, tiling for cache locality).

**Optimization Opportunities**. Even without Chapter 14's techniques, MLIR applies standard optimizations:

1. **Loop Fusion**: The scale operation could fuse into softmax's first pass—read from `scores`, scale, compute max. One fewer buffer.

2. **Buffer Reuse**: `scores` and `scale_buf` have disjoint lifetimes—after scaling completes, `scores` is dead. Reuse the same memory.

3. **Vectorization**: The element-wise loops (transpose, scale, softmax passes) are embarrassingly parallel. SIMD instructions can process 4-8 elements simultaneously.

4. **Parallelization**: Each row of attention weights is independent. Multi-thread the computation.

These optimizations are future work (Chapters 10, 14), but the point: **the IR structure enables optimization**. High-level operations preserve semantics; lowering patterns expose parallelism; standard passes apply transformations. Users write `ch11.attention(Q, K, V)`, compilers generate efficient code.

## 11.7 Numerical Validation: Testing Correctness

Attention implementation complete, how do we know it's correct? This section discusses numerical validation: comparing against reference implementations, handling floating-point precision, designing test cases catching bugs. Correctness is non-negotiable—performance optimizations mean nothing if results are wrong.

**Reference Implementation**. NumPy provides a straightforward attention:

```python
import numpy as np

def attention_reference(Q, K, V):
    """
    Reference attention implementation.
    Q, K, V: (seq_len, d_k) arrays
    Returns: (seq_len, d_k) array
    """
    # Step 1: Compute scores = Q @ K^T
    scores = Q @ K.T  # (seq_len, seq_len)
    
    # Step 2: Scale
    d_k = Q.shape[1]
    scaled_scores = scores / np.sqrt(d_k)
    
    # Step 3: Softmax
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=1, keepdims=True))
    attn_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Step 4: Weight V
    output = attn_weights @ V
    
    return output
```

Note the softmax implementation: subtract max before exponentiating (numerical stability, Section 11.1). This matches our MLIR implementation.

**Test Harness**. Compare MLIR output against NumPy:

```python
import ch11
import numpy as np

def test_attention():
    # Generate random inputs
    np.random.seed(42)  # Reproducibility
    seq_len, d_k = 4, 8
    Q = np.random.randn(seq_len, d_k).astype(np.float32)
    K = np.random.randn(seq_len, d_k).astype(np.float32)
    V = np.random.randn(seq_len, d_k).astype(np.float32)
    
    # Compute reference
    expected = attention_reference(Q, K, V)
    
    # Compute with MLIR
    Q_t = ch11.Tensor(Q)
    K_t = ch11.Tensor(K)
    V_t = ch11.Tensor(V)
    result_t = ch11.attention(Q_t, K_t, V_t)
    actual = ch11.forward(result_t)
    
    # Compare
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)
    print("✓ Attention test passed")

test_attention()
```

**Tolerance Thresholds**. The `assert_allclose()` call specifies:
- **rtol** (relative tolerance): 1e-4 (0.01% error allowed)
- **atol** (absolute tolerance): 1e-6 (for values near zero)

Why tolerances? Floating-point arithmetic is not exact:
- Different operation orderings can accumulate different rounding errors
- `exp()` and `sqrt()` use approximations (hardware or library implementations)
- Compiler optimizations might reorder operations (FMA instructions, for example)

For single-precision floats (f32), expecting exact bit-for-bit equality is unrealistic. Tolerances of 1e-4 to 1e-5 are standard—small enough to catch bugs, large enough to tolerate harmless differences.

**Edge Cases**. Random inputs catch many bugs, but structured test cases catch specific failure modes:

```python
def test_attention_identity():
    """Test that attention with identical Q, K, V behaves reasonably."""
    seq_len, d_k = 3, 4
    X = np.eye(seq_len, d_k, dtype=np.float32)  # Identity matrix
    
    X_t = ch11.Tensor(X)
    result_t = ch11.attention(X_t, X_t, X_t)
    actual = ch11.forward(result_t)
    expected = attention_reference(X, X, X)
    
    np.testing.assert_allclose(actual, expected, rtol=1e-4)
    print("✓ Identity test passed")

def test_attention_uniform():
    """Test with uniform values—all entries equal."""
    seq_len, d_k = 4, 4
    Q = np.ones((seq_len, d_k), dtype=np.float32)
    K = np.ones((seq_len, d_k), dtype=np.float32)
    V = np.ones((seq_len, d_k), dtype=np.float32)
    
    Q_t, K_t, V_t = ch11.Tensor(Q), ch11.Tensor(K), ch11.Tensor(V)
    result_t = ch11.attention(Q_t, K_t, V_t)
    actual = ch11.forward(result_t)
    expected = attention_reference(Q, K, V)
    
    # With uniform inputs, attention weights should be uniform (1/seq_len each)
    # Output should be row-wise average of V (which is all ones)
    np.testing.assert_allclose(actual, np.ones((seq_len, d_k)), rtol=1e-4)
    print("✓ Uniform test passed")

def test_attention_single_token():
    """Test with seq_len=1 (trivial case)."""
    d_k = 8
    Q = np.random.randn(1, d_k).astype(np.float32)
    K = np.random.randn(1, d_k).astype(np.float32)
    V = np.random.randn(1, d_k).astype(np.float32)
    
    Q_t, K_t, V_t = ch11.Tensor(Q), ch11.Tensor(K), ch11.Tensor(V)
    result_t = ch11.attention(Q_t, K_t, V_t)
    actual = ch11.forward(result_t)
    
    # With 1 token, attention weight is 1.0, output equals V
    np.testing.assert_allclose(actual, V, rtol=1e-4)
    print("✓ Single token test passed")
```

Each test targets specific behavior:
- **Identity**: Sparse, structured input
- **Uniform**: Symmetric attention weights
- **Single token**: Degenerate case (no choice in attention)

**Testing Individual Operations**. Before testing full attention, test components:

```python
def test_matmul():
    A = np.random.randn(3, 4).astype(np.float32)
    B = np.random.randn(4, 5).astype(np.float32)
    expected = A @ B
    
    A_t, B_t = ch11.Tensor(A), ch11.Tensor(B)
    result_t = ch11.matmul(A_t, B_t)
    actual = ch11.forward(result_t)
    
    np.testing.assert_allclose(actual, expected, rtol=1e-4)
    print("✓ Matmul test passed")

def test_transpose():
    A = np.random.randn(4, 6).astype(np.float32)
    expected = A.T
    
    A_t = ch11.Tensor(A)
    result_t = ch11.transpose(A_t)
    actual = ch11.forward(result_t)
    
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    print("✓ Transpose test passed")

def test_softmax():
    A = np.random.randn(3, 5).astype(np.float32)
    # NumPy softmax along last axis
    exp_a = np.exp(A - np.max(A, axis=1, keepdims=True))
    expected = exp_a / np.sum(exp_a, axis=1, keepdims=True)
    
    A_t = ch11.Tensor(A)
    result_t = ch11.softmax(A_t)
    actual = ch11.forward(result_t)
    
    np.testing.assert_allclose(actual, expected, rtol=1e-4)
    print("✓ Softmax test passed")
```

If these pass but attention fails, the bug is in composition logic (wrong operation order, incorrect dimensions). If these fail, the bug is in lowering patterns (wrong loop bounds, index swaps, uninitialized memory).

**Continuous Testing**. In production, tests run automatically:
- Every code change triggers test suite
- Pull requests require passing tests
- Regressions caught immediately

For this book chapter, tests validate the implementation actually works. Run [test_jit.py](ch.11.Attention/test_jit.py):

```bash
cd ch.11.Attention
python test_jit.py
```

Output:
```
✓ Matmul test passed
✓ Transpose test passed
✓ Softmax test passed
✓ Attention test passed
✓ Identity test passed
✓ Uniform test passed
✓ Single token test passed
All tests passed!
```

Each test validates a contract—if the contract changes (different tolerance, different expected behavior), update tests accordingly. The goal: high confidence in correctness, automated verification.

## 11.8 Debugging: When Attention Fails

Tests reveal bugs; debugging fixes them. This section discusses common attention implementation mistakes, diagnostic strategies, and tools for pinpointing errors. Attention is complex—many operations, many buffers, many indices—making debugging essential.

**Common Bug Categories**:

1. **Uninitialized Memory**: Forgetting to zero-initialize accumulation buffers (matmul, softmax sum)
2. **Index Errors**: Swapping indices (transpose), off-by-one (loop bounds), wrong dimension queries
3. **Shape Mismatches**: Allocating wrong-sized buffers, incompatible matmul dimensions
4. **Numerical Instability**: Not subtracting max before softmax (overflow/NaN), division by zero
5. **Compilation Failures**: Type mismatches, undefined symbols, pass pipeline errors

**Symptom 1: NaN Output**. Run test, get:

```
actual: [[nan, nan, ...], [nan, nan, ...]]
AssertionError: Arrays are not close
```

**Diagnostic Steps**:

1. **Isolate the Operation**: Test matmul, transpose, softmax individually. Which produces NaN?

   ```python
   # Test intermediate steps
   Q_t = ch11.Tensor(Q)
   K_t = ch11.Tensor(K)
   K_T = ch11.transpose(K_t)
   scores_t = ch11.matmul(Q_t, K_T)
   scores = ch11.forward(scores_t)
   print("Scores:", scores)  # Check for NaN
   ```

2. **Softmax Overflow**: If NaN appears in softmax, check max subtraction. Dump IR:

   ```python
   import ch11
   compiler = ch11.get_compiler()
   module = compiler.build_module(result_tensor.node)
   print(module)  # Inspect MLIR IR
   ```

   Look for the softmax lowering pattern. Is `math.maximumf` present? Is the max actually subtracted?

3. **Uninitialized Accumulators**: If matmul produces NaN, check zero initialization:

   ```mlir
   // WRONG: Missing zero initialization
   scf.for %i = %c0 to %rows step %c1 {
     scf.for %j = %c0 to %cols step %c1 {
       %sum = scf.for %k = %c0 to %dim step %c1 iter_args(%s = ???) -> (f32) {
         // %s is uninitialized!
         ...
       }
     }
   }
   
   // CORRECT: Zero init
   %zero = arith.constant 0.0 : f32
   %sum = scf.for %k = %c0 to %dim step %c1 iter_args(%s = %zero) -> (f32) {
     ...
   }
   ```

   Search the IR for `iter_args`. Is the initial value `arith.constant 0.0`?

4. **Division by Zero**: If softmax sums are zero, normalization divides by zero → NaN. Check input values—are all entries extremely negative? (exp(-1000) ≈ 0)

**Symptom 2: Wrong Output Values**. Tests pass sometimes, fail others. Or output is wrong but not NaN.

**Diagnostic Steps**:

1. **Compare Element-Wise**: Print both expected and actual, inspect differences:

   ```python
   print("Expected:\n", expected)
   print("Actual:\n", actual)
   print("Difference:\n", actual - expected)
   print("Max error:", np.max(np.abs(actual - expected)))
   ```

2. **Simplify Inputs**: Use trivial inputs where you can compute results manually:

   ```python
   # Identity matrices: Q @ K^T should be predictable
   Q = np.eye(3, 4, dtype=np.float32)
   K = np.eye(3, 4, dtype=np.float32)
   # Compute expected by hand...
   ```

3. **Test at Different Sizes**: Does seq_len=2 work but seq_len=4 fail? Suggests loop bound issues.

4. **Transpose Bugs**: The classic mistake (Section 11.3.2). Check generated IR:

   ```mlir
   // WRONG: Iterate input dimensions
   scf.for %i = %c0 to %c4 step %c1 {   // Input rows
     scf.for %j = %c0 to %c8 step %c1 { // Input cols
       %val = memref.load %input[%i, %j]
       memref.store %val, %output[%j, %i]  // OK so far
     }
   }
   // If input is 4x8, this works. If input is 8x4, out-of-bounds!
   
   // CORRECT: Iterate output dimensions
   scf.for %i = %c0 to %c8 step %c1 {   // Output rows (= input cols)
     scf.for %j = %c0 to %c4 step %c1 { // Output cols (= input rows)
       %val = memref.load %input[%j, %i]
       memref.store %val, %output[%i, %j]
     }
   }
   ```

   Verify loop bounds match output shape, not input shape.

**Symptom 3: Compilation Failure**. MLIR passes fail, or ExecutionEngine creation fails.

**Diagnostic Steps**:

1. **Check Pass Pipeline**: Ensure all required dialects are registered:

   ```cpp
   context_.loadDialect<TransformerDialect, func::FuncDialect,
                         arith::ArithDialect, memref::MemRefDialect,
                         scf::SCFDialect, math::MathDialect>();
   ```

2. **Verify Lowering Patterns**: Did you register all patterns in the pass?

   ```cpp
   void populateLowerTransformerToStandardPatterns(RewritePatternSet& patterns) {
     patterns.add<LowerMatmulOp>(patterns.getContext());
     patterns.add<LowerTransposeOp>(patterns.getContext());
     patterns.add<LowerSoftmaxOp>(patterns.getContext());
     patterns.add<LowerAddOp>(patterns.getContext());
     patterns.add<LowerMulOp>(patterns.getContext());
     // Did you forget one?
   }
   ```

3. **Dump IR Between Passes**: Insert `-mlir-print-ir-after-all` to see IR evolution:

   ```cpp
   auto pm = PassManager::on<ModuleOp>(&context_);
   pm.enableIRPrinting();
   ```

   Identify which pass fails, inspect IR before that pass.

4. **Type Mismatches**: Check function signatures match calling convention:

   ```cpp
   // Function signature
   func.func @foo(%arg0: memref<4x8xf32>) { ... }
   
   // Calling code
   %wrong_type = ... : memref<8x4xf32>  // Incompatible!
   func.call @foo(%wrong_type) : (memref<8x4xf32>) -> ()
   ```

   MLIR's type system catches these, but error messages can be cryptic. Read carefully.

**Debugging Tools**:

- **llvm::errs() in C++**: Print debug messages from lowering patterns:
  ```cpp
  llvm::errs() << "Lowering matmul with shape: " 
               << lhsType.getShape() << " x " << rhsType.getShape() << "\n";
  ```

- **module.dump() in Python bindings**: Expose MLIR IR to Python for inspection.

- **gdb/lldb**: Step through JIT-compiled code (requires debug symbols, non-trivial setup).

- **MLIR Passes**: Use `-mlir-print-op-generic` for fully-qualified IR (shows all attributes, types).

**Example Debugging Session**. Suppose transpose fails:

```
Expected: [[1, 4], [2, 5], [3, 6]]
Actual:   [[1, 2], [3, 4], [5, 6]]
```

Output shape is correct (3x2), but values are wrong. Hypothesis: indices are not swapped. Inspect IR:

```mlir
scf.for %i = %c0 to %c2 step %c1 {
  scf.for %j = %c0 to %c3 step %c1 {
    %val = memref.load %input[%i, %j]
    memref.store %val, %output[%i, %j]  // BUG: Should be [%j, %i]
  }
}
```

There it is: `memref.store ... [%i, %j]` instead of `[%j, %i]`. Fix the lowering pattern:

```cpp
builder.create<memref::StoreOp>(loc, val, output, ValueRange{j, i});  // Swapped
```

Recompile, retest—passes now.

Debugging is iterative: hypothesize, test, inspect, fix, repeat. MLIR's IR is readable—leverage that. When stuck, simplify: smaller inputs, fewer operations, more print statements. Attention is deterministic; bugs are reproducible.

## 11.9 Performance Characteristics

Implementation correct, how fast is it? This section analyzes attention's performance, identifying bottlenecks and comparing against optimized libraries. Understanding performance guides optimization efforts—know what's slow before trying to speed it up.

**Benchmarking Setup**. Timing attention with various input sizes:

```python
import ch11
import numpy as np
import time

def benchmark_attention(seq_len, d_k, num_runs=100):
    Q = np.random.randn(seq_len, d_k).astype(np.float32)
    K = np.random.randn(seq_len, d_k).astype(np.float32)
    V = np.random.randn(seq_len, d_k).astype(np.float32)
    
    Q_t = ch11.Tensor(Q)
    K_t = ch11.Tensor(K)
    V_t = ch11.Tensor(V)
    result_t = ch11.attention(Q_t, K_t, V_t)
    
    # Warmup (JIT compilation)
    _ = ch11.forward(result_t)
    
    # Timed runs
    start = time.time()
    for _ in range(num_runs):
        output = ch11.forward(result_t)
    elapsed = time.time() - start
    
    avg_time = elapsed / num_runs
    return avg_time

# Test different sizes
sizes = [(64, 64), (128, 64), (256, 64), (512, 64)]
for seq_len, d_k in sizes:
    t = benchmark_attention(seq_len, d_k)
    print(f"seq_len={seq_len}, d_k={d_k}: {t*1000:.2f} ms per forward pass")
```

Example output (on a modern CPU):
```
seq_len=64, d_k=64: 0.32 ms
seq_len=128, d_k=64: 1.15 ms
seq_len=256, d_k=64: 4.28 ms
seq_len=512, d_k=64: 16.85 ms
```

**Complexity Analysis**. From Section 11.1:
- Matmul (Q @ K^T): O(seq_len² × d_k)
- Scaling: O(seq_len²)
- Softmax: O(seq_len²)
- Matmul (weights @ V): O(seq_len² × d_k)
- **Total**: O(seq_len² × d_k)

Doubling seq_len quadruples compute. The 64→128 jump (2x seq_len) should be ~4x slower: 0.32 ms × 4 = 1.28 ms. Actual: 1.15 ms—close, suggesting compute-bound behavior.

**Bottleneck Breakdown**. Profile individual operations:

```python
def profile_attention(seq_len, d_k):
    Q = np.random.randn(seq_len, d_k).astype(np.float32)
    K = np.random.randn(seq_len, d_k).astype(np.float32)
    V = np.random.randn(seq_len, d_k).astype(np.float32)
    
    Q_t = ch11.Tensor(Q)
    K_t = ch11.Tensor(K)
    V_t = ch11.Tensor(V)
    
    # Time transpose
    start = time.time()
    K_T = ch11.transpose(K_t)
    _ = ch11.forward(K_T)
    t_transpose = time.time() - start
    
    # Time first matmul
    scores_t = ch11.matmul(Q_t, K_T)
    start = time.time()
    _ = ch11.forward(scores_t)
    t_matmul1 = time.time() - start
    
    # ... (time scaling, softmax, second matmul similarly)
    
    print(f"Transpose: {t_transpose*1000:.2f} ms")
    print(f"Matmul (Q@KT): {t_matmul1*1000:.2f} ms")
    # ...
```

Typical results (seq_len=256, d_k=64):
```
Transpose: 0.05 ms (1%)
Matmul (Q@KT): 2.80 ms (65%)
Scaling: 0.03 ms (<1%)
Softmax: 0.22 ms (5%)
Matmul (weights@V): 1.18 ms (28%)
Total: ~4.28 ms
```

**Key Observation**: The two matmuls dominate (93% of time). Softmax is 5%, transpose/scaling negligible. This matches theoretical expectations—matmuls are O(n³) (treating seq_len as n), while softmax is O(n²).

**Performance Context**. Our implementation is **educational**, demonstrating MLIR compilation concepts. The naive nested loops execute on a single CPU core without SIMD vectorization, fusion, or threading. Production attention implementations—whether in PyTorch (using Intel MKL or OpenBLAS), TensorFlow (XLA compilation), or specialized libraries (cuDNN, FlashAttention)—are significantly faster through:

1. **Optimized BLAS**: Hand-tuned assembly with SIMD instructions and cache blocking for matrix multiplication
2. **Kernel Fusion**: Combining operations (scale + softmax) to reduce memory bandwidth
3. **Multi-threading**: Parallelizing across CPU cores
4. **GPU Acceleration**: Exploiting thousands of parallel cores for data-parallel operations

The performance gap can be substantial (10-100× depending on problem size and hardware). For Chapter 11, the goal is correctness and understanding MLIR's compilation pipeline, not achieving peak performance. The infrastructure is in place; optimizations come in later chapters.

**Optimization Roadmap** (future chapters):

- **Chapter 10**: Apply vectorization, loop fusion, parallelization to generic operations
- **Chapter 14**: Production-grade optimizations including advanced tiling techniques
- **GPU**: Offload to accelerators where massive parallelism shines

For Chapter 11, focus is on correctness and clarity. The MLIR infrastructure is established; performance optimizations layer on top in subsequent chapters.

## 11.10 Conclusion

This chapter built a complete attention implementation in MLIR, from high-level operations to structured linear algebra operations to executable machine code. We defined a Transformer dialect with five operations, wrote lowering patterns converting those operations to linalg operations (which then lower to SCF loops and arithmetic), designed a Python API building computation graphs, and JIT-compiled graphs to native code via LLVM.

**Key Takeaways**:

1. **Domain-Specific Dialects**: Attention is naturally expressed as transformer.matmul, transformer.softmax, etc. Custom operations capture domain semantics, simplifying user code.

2. **Multi-Level IR**: The same computation exists at four levels—Python API (user-facing), transformer dialect (semantic), linalg operations (structured), SCF+arith (imperative). Each level serves a purpose: Python for ergonomics, transformer for domain semantics, linalg for optimization, SCF for execution.

3. **Linalg as Optimization Layer**: By lowering to linalg operations instead of raw loops, we enable MLIR's optimization passes (tiling, fusion, vectorization) to transform our code automatically. The transformer dialect becomes a thin wrapper providing ergonomic names while leveraging linalg's powerful infrastructure.

4. **Pattern Rewriting**: Lowering patterns are modular—each operation lowers independently. MLIR's rewrite infrastructure handles orchestration. Adding new operations means adding new patterns; existing patterns remain unchanged.

5. **Numerical Correctness**: Attention involves transcendental functions (exp), accumulation (matmul), and normalization (softmax). Numerical stability (max subtraction) and initialization (zero accumulators) are critical. Testing against references catches errors.

6. **Compilation Pipeline**: Computation graph → transformer dialect → linalg operations → SCF loops → LLVM IR → machine code → execution. Each stage is well-defined, with clear inputs/outputs. This modularity enables experimentation—swap lowering patterns, change targets, insert optimization passes.

**Limitations of Our Approach**:

- **Performance**: Naive loops without vectorization, fusion, or parallelization. Significantly slower than production implementations.
- **Memory**: Explicit intermediate buffers (scores, attn_weights) consume O(seq_len²) memory. Chapter 14's tiling reduces this.
- **Batch Size**: We handle single examples (seq_len, d_k). Production systems batch multiple examples, amortizing overhead.
- **Recompilation**: Every `forward()` call recompiles. Caching compiled modules is straightforward but omitted for clarity.

**Looking Ahead**:

- **Chapter 12** (Transformer Blocks): Compose attention with feed-forward networks, layer normalization, residual connections. Build full transformer layers.
- **Chapter 13** (GPT Architecture): Stack transformer blocks, add positional encodings, implement autoregressive generation.
- **Chapter 14** (Production Optimizations): Apply DRR for pattern matching, write custom interfaces, implement FlashAttention-inspired fusion, explore KV caching for inference.
- **Chapter 15** (GPU Concepts): Port attention to GPUs, leveraging thousands of parallel threads.

Attention is the heart of transformers. Mastering its implementation—understanding each operation, each lowering decision, each numerical consideration—provides the foundation for modern AI compilers. From here, we scale: more layers, more optimizations, more targets. MLIR gives us the tools; we provide the insights.