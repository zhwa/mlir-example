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

3. **Type System**: Use tensors for functional-style operations with automatic bufferization handling memory management. Shapes can be dynamic (runtime-determined sequence lengths).

4. **Composability**: Operations compose to build higher-level primitives. Attention is `matmul` + `transpose` + element-wise scaling (via `mul`) + `softmax` + `matmul` again.

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

**Core Operations**. We define five operations matching attention's needs: `matmul`, `add`, `mul`, `softmax`, and `transpose`. Note that scaling (dividing by √d_k) is implemented at the Python API level by creating a constant-filled memref and using `mul` for element-wise multiplication—not as a separate dialect operation.

### 11.2.1 Matrix Multiplication

```tablegen
def Transformer_MatmulOp : Transformer_Op<"matmul"> {
  let summary = "Matrix multiplication";
  let description = [{
    Performs matrix multiplication: C = A @ B

    Supports 2D and 3D tensors (batched matmul).
    For 2D: (M, K) @ (K, N) -> (M, N)
    For 3D: (B, M, K) @ (B, K, N) -> (B, M, N)
    
    Returns a new tensor with the result.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}
```

**Design Choices**:
- **Tensor operations**: Operations consume tensor arguments and return tensor results, following pure functional semantics.
- **AnyTensor**: Accepts tensors of any rank/type (typically `tensor<?x?xf32>` for dynamic 2D matrices).
- **Results instead of output parameters**: Modern MLIR uses result values rather than output parameters.
- **Bufferization handles memory**: Tensor operations remain pure; the bufferization pipeline converts to memref-based execution later.
- **Assembly format**: Custom syntax `%result = transformer.matmul %A, %B : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>` mirrors pure functional semantics.


### 11.2.2 Transpose

```tablegen
def Transformer_TransposeOp : Transformer_Op<"transpose"> {
  let summary = "Transpose last two dimensions";
  let description = [{
    Transposes the last two dimensions of a tensor.

    For 2D: (M, N) -> (N, M)
    For 3D: (B, M, N) -> (B, N, M)
    
    Example:
      %transposed = transformer.transpose %input : tensor<?x?xf32> -> tensor<?x?xf32>
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}
```

Attention requires transposing K to compute Q @ K^T. Note the specification: "last two dimensions." This handles batched attention (3D tensors) automatically—batch dimension remains unchanged, inner matrix transposes. The operation returns a tensor result rather than mutating an output parameter.

### 11.2.3 Softmax

```tablegen
def Transformer_SoftmaxOp : Transformer_Op<"softmax"> {
  let summary = "Softmax activation along last dimension";
  let description = [{
    Applies numerically stable softmax along the last dimension:
    
    softmax(x)[i] = exp(x[i] - max(x)) / sum_j(exp(x[j] - max(x)))
    
    Example:
      %weights = transformer.softmax %scaled_scores : tensor<?x?xf32> -> tensor<?x?xf32>
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}
```

The description specifies **numerically stable** softmax—max subtraction prevents overflow. This is crucial: without it, exp(large_number) produces infinity, breaking attention. The implementation (Section 11.3) shows the four-pass tensor-based algorithm.

### 11.2.4 Element-Wise Operations

```tablegen
def Transformer_AddOp : Transformer_Op<"add"> {
  let summary = "Element-wise addition";
  let description = [{
    result = lhs + rhs
    
    Example:
      %sum = transformer.add %lhs, %rhs : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  }];
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

def Transformer_MulOp : Transformer_Op<"mul"> {
  let summary = "Element-wise multiplication";
  let description = [{
    result = lhs * rhs
    
    Example:
      %scaled = transformer.mul %scores, %scale_factor : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  }];
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}
```

Addition handles residual connections (Chapter 12). Multiplication enables element-wise operations, including scaling. The Python API's `scale()` function uses `mul` to implement scaling by √d_k. Both operations are rank-generic: work on any-shaped tensors with compatible shapes.


## 11.3 Lowering Patterns

Operations defined, we now implement lowering—converting high-level `transformer.*` operations to MLIR's **Linalg dialect** using tensor operations. The transformer dialect provides domain-specific names (`transformer.matmul` is clearer than `linalg.matmul` in attention code) that immediately lower to linalg for optimization. The bufferization pipeline (Section 11.5) later converts these pure tensor operations to efficient memref-based execution code.

### 11.3.1 Matrix Multiplication Lowering

Matrix multiplication C = A @ B lowers to tensor-based linalg operations:

```mlir
%empty = tensor.empty(%M, %N) : tensor<?x?xf32>
%zero = arith.constant 0.0 : f32
%filled = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
%result = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                        outs(%filled : tensor<?x?xf32>) -> tensor<?x?xf32>
```

**Tensor Creation**: `tensor.empty` creates an uninitialized tensor (bufferization allocates memory later).

**Zero Initialization**: `linalg.fill` returns a new zero-filled tensor. This is a pure operation—no mutation.

**Structured Operation**: `linalg.matmul` computes matrix multiplication on tensors, returning the result tensor. MLIR knows its semantics and can optimize (tile, fuse, vectorize) before bufferization.

**The Lowering Pattern**. From [src/TransformerToStandard.cpp](ch.11.Attention/src/TransformerToStandard.cpp):

```cpp
struct MatmulOpLowering : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op,
                                  PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // Get result type from the operation
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());

    // Create empty tensor for output
    SmallVector<Value> dynamicDims;
    for (auto dim : resultType.getShape()) {
      if (ShapedType::isDynamic(dim)) {
        auto dimSize = rewriter.create<tensor::DimOp>(loc, lhs, 0);
        dynamicDims.push_back(dimSize);
      }
    }
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType, dynamicDims);

    // Zero-initialize with linalg.fill (returns new tensor)
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(0.0));
    Value filledTensor = rewriter.create<linalg::FillOp>(
        loc, zero, emptyTensor).getResult(0);

    // Matrix multiplication (tensor → tensor)
    Value result = rewriter.create<linalg::MatmulOp>(
        loc, ValueRange{lhs, rhs}, ValueRange{filledTensor}).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};
```

The pattern creates tensor operations (`tensor.empty`, `linalg.fill`, `linalg.matmul`) that return result values. The bufferization pipeline (Section 11.5) later converts these to efficient memref-based execution. This ~25-line implementation replaces what would require ~115 lines of nested loops with manual index management, while enabling optimizations that run before loop generation.

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

    rewriter.replaceOp(op, output);
    return success();
  }
};
```

This pattern constructs a permutation vector `[0, 1, ..., rank-3, rank-1, rank-2]` (identity except swapping the last two dimensions). Works for 2D (rank 2) and 3D (rank 3, batched attention) tensors uniformly.

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

**The Lowering Pattern** (abbreviated, focusing on the key operations):

```cpp
struct SoftmaxOpLowering : public OpRewritePattern<SoftmaxOp> {
  LogicalResult matchAndRewrite(SoftmaxOp op, PatternRewriter &rewriter) const override {
    // ... extract location, input, types, allocate temporaries ...
    
    // Step 1: Reduce max along last dimension
    rewriter.create<linalg::ReduceOp>(
        loc, ValueRange{input}, ValueRange{maxVals},
        ArrayRef<int64_t>{rank - 1},
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value max = b.create<arith::MaximumFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, max);
        });
    
    // Step 2: Subtract max (broadcasting) and compute exp
    Value expVals = rewriter.create<linalg::GenericOp>(
        loc, outputType, ValueRange{input, maxVals}, ValueRange{expTensor},
        broadcastMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value diff = b.create<arith::SubFOp>(loc, args[0], args[1]);
          Value exp = b.create<math::ExpOp>(loc, diff);
          b.create<linalg::YieldOp>(loc, exp);
        }).getResult(0);
    
    // Step 3: Reduce sum along last dimension
    rewriter.create<linalg::ReduceOp>(
        loc, ValueRange{expVals}, ValueRange{sumVals},
        ArrayRef<int64_t>{rank - 1},
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });
    
    // Step 4: Divide by sum (broadcasting)
    Value normalized = rewriter.create<linalg::GenericOp>(
        loc, outputType, ValueRange{expVals, sumVals}, ValueRange{outTensor},
        broadcastMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::DivFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, result);
        }).getResult(0);
    
    rewriter.replaceOp(op, normalized);
    return success();
  }
};
```

Linalg's structured form enables **fusion**: later passes can merge operations, eliminating intermediate buffers and improving performance.

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
    
    rewriter.replaceOp(op, result);
    return success();
  }
};
```

Multiplication follows identically, replacing `arith.AddFOp` with `arith.MulFOp`. Linalg's rank-polymorphic operations handle arbitrary tensor ranks naturally.

### 11.3.5 From Tensors to Executable Code: The Pass Pipeline

Our transformer operations lower to tensor-based linalg operations, which then go through bufferization and loop generation. The complete pass pipeline:

```cpp
pm.addPass(createLowerTransformerToStandardPass());  // transformer -> linalg (tensors)
// Optimization passes work on linalg tensor operations here
pm.addPass(createBufferizePass());                   // tensors -> memrefs
pm.addPass(createConvertLinalgToLoopsPass());        // linalg -> scf.for
pm.addPass(createSCFToControlFlowPass());            // scf -> control flow
// ... remaining passes to LLVM IR
```

**The Tensor-First Advantage**: Our lowering produces tensor-based linalg operations (`linalg.matmul` on tensors, `linalg.generic` with tensor inputs/outputs). This enables optimization passes that work on the tensor level:

- **Tiling**: Break operations into cache-friendly blocks
- **Fusion**: Merge producer-consumer operations, eliminating intermediate tensors
- **Vectorization**: Generate SIMD instructions (AVX, NEON)
- **Parallelization**: Distribute across threads

After optimizations, bufferization converts tensors to memrefs, and loop lowering generates actual control flow. This staged approach—high-level tensor operations → optimizations → bufferization → loops—gives us the best of both worlds: composable functional-style IR and efficient imperative execution.

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

## 11.4 JIT Compilation: From Graphs to Native Code

Computation graph built (using the tensor abstraction design from Chapter 10), we now compile it to executable code. This section traces the compilation pipeline: graph → MLIR IR → LLVM IR → native machine code → execution via libffi. Each stage transforms the representation, eventually producing code running on the CPU.

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

**Stage 1: Graph to MLIR**. The `buildGraphFunction()` traverses the graph, emitting tensor-based MLIR operations:

```cpp
void buildGraphFunction(OpBuilder& builder, ModuleOp module,
                         std::shared_ptr<GraphNode> outputNode) {
  auto loc = builder.getUnknownLoc();
  
  // Collect all input nodes
  std::vector<std::shared_ptr<GraphNode>> inputs;
  std::unordered_map<GraphNode*, Value> nodeToValue;
  collectInputs(outputNode, inputs);
  
  // Build function signature with TENSOR types
  SmallVector<Type> argTypes;
  for (auto& input : inputs) {
    // Dynamic tensor types (compatible with NumPy arrays)
    SmallVector<int64_t> dynamicShape(input->shape.size(), ShapedType::kDynamic);
    auto tensorType = RankedTensorType::get(dynamicShape, builder.getF32Type());
    argTypes.push_back(tensorType);
  }
  
  // Function returns a tensor (not mutating an output parameter)
  SmallVector<int64_t> outputShape(outputNode->shape.size(), ShapedType::kDynamic);
  auto outputType = RankedTensorType::get(outputShape, builder.getF32Type());
  
  // Create function with tensor signature
  auto funcType = builder.getFunctionType(argTypes, {outputType});
  auto func = builder.create<func::FuncOp>(loc, "graph_func", funcType);
  auto* entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  
  // Map input nodes to function arguments (tensors)
  for (size_t i = 0; i < inputs.size(); ++i) {
    nodeToValue[inputs[i].get()] = entryBlock->getArgument(i);
  }
  
  // Emit operations for each node (returning tensors)
  Value finalResult = emitNode(builder, outputNode, nodeToValue);
  
  builder.create<func::ReturnOp>(loc, finalResult);
  module.push_back(func);
}
```

**Emitting Operations**. The `emitNode()` function returns tensor values:

```cpp
Value emitNode(OpBuilder& builder, std::shared_ptr<GraphNode> node,
               std::unordered_map<GraphNode*, Value>& nodeToValue) {
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
      Value lhs = emitNode(builder, node->inputs[0], nodeToValue);
      Value rhs = emitNode(builder, node->inputs[1], nodeToValue);
      
      // Compute result type
      auto lhsType = mlir::cast<RankedTensorType>(lhs.getType());
      auto rhsType = mlir::cast<RankedTensorType>(rhs.getType());
      SmallVector<int64_t> resultShape = {lhsType.getShape()[0], rhsType.getShape()[1]};
      auto resultType = RankedTensorType::get(resultShape, builder.getF32Type());
      
      // Emit matmul operation (returns tensor result)
      Value result = builder.create<MatmulOp>(loc, resultType, lhs, rhs).getResult();
      
      nodeToValue[node.get()] = result;
      return result;
    }
    // ... other cases ...
  }
}
```

The pattern: emit dependencies recursively, compute result type, emit operation returning tensor result, cache result. No allocations—tensor operations are pure functions. The lowering patterns (Section 11.3) create `tensor.empty` operations, and bufferization (next) converts those to allocations.

**Generated MLIR**. For the attention graph, the emitted IR looks like:

```mlir
func.func @graph_func(%Q: tensor<?x?xf32>, %K: tensor<?x?xf32>,
                       %V: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // Transpose K
  %K_T = transformer.transpose %K : tensor<?x?xf32> -> tensor<?x?xf32>
  
  // Compute scores = Q @ K^T
  %scores = transformer.matmul %Q, %K_T : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  
  // Scale (implemented as mul with constant tensor)
  %scale_factor = <constant tensor creation>
  %scaled = transformer.mul %scores, %scale_factor : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  
  // Apply softmax
  %attn_weights = transformer.softmax %scaled : tensor<?x?xf32> -> tensor<?x?xf32>
  
  // Compute output = weights @ V
  %output = transformer.matmul %attn_weights, %V : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  
  return %output : tensor<?x?xf32>
}
```

Pure functional operations, tensor types throughout, no explicit allocations. This high-level tensor-based IR is ready for lowering and bufferization.

**Stage 2: Lowering to Linalg and Bufferization**. The compiler applies passes from Section 11.3, followed by a critical three-pass bufferization pipeline:

```cpp
bool TransformerCompiler::lowerToLLVM(ModuleOp module) {
  PassManager pm(&context_);

  // Step 1: Lower transformer dialect to tensor-based linalg
  pm.addNestedPass<func::FuncOp>(createLowerTransformerToStandardPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  
  // Step 2: Register bufferization interfaces for all relevant dialects
  DialectRegistry registry;
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  context_.appendDialectRegistry(registry);
  
  // Step 3: Three-pass bufferization (tensor → memref)
  bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizationOptions));
  pm.addPass(bufferization::createBufferResultsToOutParamsPass());
  pm.addPass(createConvertBufferizationToMemRefPass());
  pm.addPass(createCanonicalizerPass());
  
  // Step 4: Lower linalg to loops (now operating on memrefs)
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // Step 5: Lower standard dialects to LLVM
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertMathToLibmPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  return succeeded(pm.run(module));
}
```

**The Bufferization Interface Registration**: Before bufferization can work, we must register "bufferizable op interface" implementations for each dialect that uses tensors. This tells the bufferization pass how to convert operations from each dialect:

- `arith::registerBufferizableOpInterfaceExternalModels`: Handle arithmetic ops on tensors
- `linalg::registerBufferizableOpInterfaceExternalModels`: Handle linalg ops (matmul, generic, etc.)
- `tensor::registerBufferizableOpInterfaceExternalModels`: Handle tensor creation/manipulation (empty, extract_slice, etc.)
- `bufferization::func_ext::registerBufferizableOpInterfaceExternalModels`: Handle function signatures and calls

Without these registrations, the bufferization pass would not know how to convert tensor operations to memref operations.

**The Bufferization Pipeline**:

1. **OneShotBufferize**: Converts tensor operations to memref operations. With `bufferizeFunctionBoundaries = true`, it handles function boundaries correctly—tensors in function signatures also convert to memrefs.

After this pass:
- `%result = linalg.matmul ins(%a, %b : tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>` becomes
- `%alloc = memref.alloc() : memref<?x?xf32>` + `linalg.matmul ins(%a, %b) outs(%alloc)`

2. **BufferResultsToOutParams**: Transforms function results into output parameters, matching the calling convention needed for libffi execution:

- `func.func @graph_func(%Q: memref<?x?xf32>, %K: memref<?x?xf32>) -> memref<?x?xf32>` becomes
- `func.func @graph_func(%Q: memref<?x?xf32>, %K: memref<?x?xf32>, %arg2: memref<?x?xf32>)`

The final argument receives the output, enabling efficient in-place computation.

3. **ConvertBufferizationToMemRef**: Removes remaining tensor artifacts. Converts `tensor.empty` → `memref.alloc`, cleans up any bufferization-specific operations.

After these three passes, all tensor operations are gone—the IR is fully memref-based, ready for loop lowering.

**Why Three Passes?** Each handles a distinct aspect:
- **OneShotBufferize**: Core tensor → memref conversion
- **BufferResultsToOutParams**: ABI transformation (results → output parameters)
- **ConvertBufferizationToMemRef**: Cleanup (remove tensor remnants)

Trying to do all three in one pass would be extremely complex. The modular approach keeps each pass focused and maintainable.

First, transformer ops → tensor-based linalg (Section 11.3's patterns). Then, bufferization converts tensors → memrefs. Then, linalg → scf.for loops (via `createConvertLinalgToLoopsPass()`). Finally, loops → LLVM IR (standard MLIR passes). Canonicalization and CSE (common subexpression elimination) clean up between major transformations. After this pipeline, the module contains only LLVM dialect operations—pointers, branches, arithmetic instructions.

**Stage 3: LLVM IR to Native Code**. MLIR's ExecutionEngine compiles LLVM dialect to machine code. The process: register LLVM dialect translations, invoke `ExecutionEngine::create()` which runs LLVM's JIT compiler (OrcJIT) at optimization level 3, generate machine code for the target architecture, and look up the compiled function's entry point address—a raw pointer to executable code in memory.

**Stage 4: Execution via libffi**. With a function pointer to the compiled code, we need to call it. MLIR's memref calling convention (Chapter 7) passes memrefs as expanded arguments: pointer, offset, sizes, strides. For 2D memrefs, that's 7 arguments per memref. The attention function with three 2D inputs (Q, K, V) and one 2D output requires 28 arguments total. Manually marshaling this is tedious and error-prone.

**libffi** (Foreign Function Interface library) solves this by dynamically constructing function calls. The key steps:

```cpp
// Marshal each input memref
std::vector<void*> arg_values;
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
```

The `marshal_memref_2d()` function expands NumPy arrays into memref descriptor arguments, which has been discussed in previous chapters.

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
  
  // Step 3: Scale by 1/sqrt(d_k) - implemented via constant fill + mul
  %scale_constant = memref.alloc() : memref<4x4xf32>
  %scale_factor = arith.constant 0.353553 : f32  // 1/sqrt(8)
  // (nested scf.for loops fill %scale_constant with %scale_factor)
  %scale_buf = memref.alloc() : memref<4x4xf32>
  transformer.mul %scores, %scale_constant, %scale_buf : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>
  
  // Step 4: Apply softmax
  %attn_weights = memref.alloc() : memref<4x4xf32>
  transformer.softmax %scale_buf, %attn_weights : memref<4x4xf32>, memref<4x4xf32>
  
  // Step 5: Multiply attention weights by V
  transformer.matmul %attn_weights, %V, %output : memref<4x4xf32>, memref<4x8xf32>, memref<4x8xf32>
  
  func.return
}
```

Four transformer dialect operations (transpose, two matmuls, softmax) plus element-wise multiplication for scaling, with several intermediate buffers. This is the **before-lowering** IR—still high-level, still compositional. Note that the scale operation expands into constant buffer allocation, filling loops, and a `transformer.mul` operation. 

**Two-Stage Lowering**: Section 11.3's patterns lower transformer operations to **linalg** operations first, then MLIR's `createConvertLinalgToLoopsPass()` converts linalg to scf.for loops. The intermediate linalg representation enables optimization passes (tiling, fusion, vectorization) before final loop generation.

**IR Transformation Stages**. To understand what the compiler generates, consider the transformation pipeline for attention:

**Stage 1: Transformer Dialect Operations** → The high-level `transformer.attention` operation and its components (matmul, transpose, softmax, scale) exist at this level. These are domain-specific operations that capture semantic meaning.

**Stage 2: Linalg Operations** → After lowering, operations become structured linalg ops: `linalg.matmul` for matrix operations, `linalg.generic` for element-wise operations (scaling), `linalg.reduce` for reductions (max, sum in softmax), and `linalg.transpose` with explicit permutations. This representation preserves semantic meaning—MLIR's optimizer understands what these operations do and can apply transformations like tiling and fusion.

**Stage 3: Explicit Loops** → After `createConvertLinalgToLoopsPass()`, linalg operations become nested `scf.for` loops with explicit indices, bounds, and strides. For attention with seq_len=4 and d_k=8, this produces approximately 18 nested loops: 2 for transpose, 3 for each matmul (2 total = 6 loops), 2 for scaling, 6 for softmax's three passes (find max per row, compute exp, normalize), plus memory allocations for intermediate buffers (K_T, scores, scale_buf, attn_weights, max_buf, sum_buf). The generated code explicitly iterates through memory, performs arithmetic operations, and manages intermediate storage—typically expanding a few high-level operations into ~100 lines of loop-based IR.

**Memory Considerations**. For small inputs (seq_len=4, d_k=8), intermediate buffers total ~352 bytes. But at production scale (seq_len=512, d_k=64), attention requires ~3.3 MB per operation: transposed keys (128 KB), scores matrix (1 MB), scaled scores (1 MB), attention weights (1 MB), plus reduction buffers. Multiply by number of attention heads (8-12 typical) and layers (12-24 typical), and memory becomes significant. The explicit loop representation enables optimizations: loop fusion can eliminate intermediate buffers (scale directly into softmax's first pass), buffer reuse allows reusing memory for operations with disjoint lifetimes (scores buffer can become scale_buf), vectorization applies SIMD to element-wise operations, and parallelization distributes independent row computations across threads. These optimizations (explored in Chapters 10 and 14) leverage the IR structure—high-level operations preserve semantics for analysis, lowering patterns expose parallelism, and standard passes apply transformations automatically.

## 11.7 Numerical Validation: Testing Correctness

Attention implementation complete, how do we know it's correct? Numerical validation compares MLIR outputs against reference implementations, with careful attention to floating-point precision.

**Testing Strategy**. A NumPy reference implementation provides ground truth, using the same numerically stable softmax (subtract max before exponentiating). The test harness compares MLIR-compiled results against NumPy using `np.testing.assert_allclose()` with tolerances (rtol=1e-4, atol=1e-6) that account for harmless floating-point differences from operation reordering, hardware approximations (exp, sqrt), and compiler optimizations. These tolerances are small enough to catch bugs but large enough to tolerate implementation differences.

**Test Coverage**. Beyond random inputs, structured test cases target specific behaviors: identity matrices test sparse structured inputs, uniform values test symmetric attention weights, single-token sequences test degenerate cases. Testing individual operations (matmul, transpose, softmax) before testing full attention helps isolate bugs—if components pass but attention fails, the bug is in composition logic; if components fail, the bug is in lowering patterns.

**Common Debugging Scenarios**:

**NaN outputs** typically indicate numerical instability (missing max subtraction in softmax, causing overflow) or uninitialized memory (accumulation buffers not zeroed). Isolate which operation produces NaN by testing intermediate results. Inspect the generated IR to verify softmax includes `math.maximumf` for finding max and proper subtraction, and that loop accumulators initialize with `arith.constant 0.0`.

**Wrong values** suggest index errors (transpose with swapped indices), shape mismatches (wrong buffer allocations), or incorrect loop bounds. Compare expected vs actual element-wise, test with trivial inputs where results are manually computable, and try different sizes (does seq_len=2 work but seq_len=4 fail?). The classic transpose bug: iterating over input dimensions instead of output dimensions causes out-of-bounds access when shapes don't match.

**Compilation failures** point to missing dialect registrations, unregistered lowering patterns, or type mismatches between operations. Use `-mlir-print-ir-after-all` to see IR evolution through the pass pipeline and identify which pass fails. Check that all patterns are added to the RewritePatternSet and all required dialects are loaded.

**Debugging Tools**: Insert `llvm::errs()` print statements in lowering patterns to trace execution, expose MLIR IR to Python for inspection, and use `-mlir-print-op-generic` for fully-qualified IR showing all attributes and types. Debugging is iterative: hypothesize, test simple cases, inspect IR, fix, repeat. MLIR's readable IR is your primary diagnostic tool.

The test suite in [test_jit.py](ch.11.Attention/test_jit.py) validates all operations and attention mechanism, providing confidence in correctness through automated verification.

## 11.8 Conclusion

This chapter built a complete attention implementation in MLIR, from high-level operations to executable machine code. We defined a Transformer dialect with five operations (`matmul`, `add`, `mul`, `softmax`, `transpose`), wrote lowering patterns to tensor-based linalg operations, implemented a tensor abstraction for building computation graphs, and JIT-compiled graphs to native code via LLVM.

**Key insights**: Domain-specific dialects capture attention semantics naturally. Multi-level IR (Python API → transformer dialect → linalg → loops → LLVM IR) enables optimization at each abstraction level. Lowering to linalg rather than raw loops unlocks MLIR's optimization passes (tiling, fusion, vectorization). Pattern rewriting keeps transformations modular—adding operations means adding patterns without changing existing code. Numerical correctness requires careful attention to stability (max subtraction in softmax) and initialization (zeroed accumulators).

**Next steps**: Chapter 12 composes attention with feed-forward networks and layer normalization to build full transformer blocks. Chapter 13 stacks these blocks into complete GPT architecture. Chapter 14 applies production optimizations (DRR pattern matching, FlashAttention-inspired fusion, KV caching). Chapter 15 explores GPU implementations. Attention is the heart of transformers—mastering its implementation provides the foundation for modern AI compilers.