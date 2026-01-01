# Chapter 14: Production-Grade Optimizations

Chapters 1-13 built a complete journey: MLIR fundamentals (1-9), transformer architecture (10-12), and GPT implementation (13). Chapter 13's GPT uses Linalg dialect for structured operations, enabling basic compiler optimizations (elementwise fusion, LICM, CSE) that provide speedups. The implementation is correct and demonstrates key architectural patterns, but production systems require orders-of-magnitude greater performance.

Modern MLIR compilers achieve dramatic speedups through aggressive loop transformations and algorithmic improvements. **Transform Dialect** applies tiling to create cache-friendly memory access patterns, vectorization to exploit SIMD units, and advanced fusion to eliminate intermediate memory traffic—providing 3-5× speedups for production models. **KV caching** changes the generation algorithm itself, eliminating O(N²) redundant computation by caching attention keys/values across generation steps—providing 10-100× speedups. **Declarative Rewrite Rules (DRR)** express optimizations as pattern-match-rewrite specifications in TableGen. **OpInterface** enables polymorphic algorithms operating on operation categories rather than concrete types. These techniques represent the state-of-the-art in production compilers like IREE, Torch-MLIR, and NVIDIA's systems.

This chapter introduces all four techniques and implements Transform Dialect with KV caching in the codebase. DRR and OpInterface are presented as educational material—widely used in production but not implemented here to maintain focus on the optimization pipeline. Section 14.1 demonstrates DRR syntax and semantics. Section 14.2 covers canonicalization patterns. Section 14.3 shows OpInterface design. Section 14.4 implements Transform Dialect optimization pipeline. Section 14.5 implements KV caching for generation. Our nano GPT (d_model=64, 2 layers, ~50KB fitting entirely in L1 cache) cannot demonstrate the full performance gains visible in production models (GPT-2: 500MB, GPT-3: 350GB), but the implementations are architecturally identical and production-ready.

## 14.1 Declarative Rewrite Rules and Canonicalization

Chapter 9 introduced TableGen for defining operations but deferred **Declarative Rewrite Rules (DRR)**—a TableGen-based pattern matching system for expressing optimizations declaratively. DRR eliminates boilerplate C++ for common transformations. **Canonicalization** uses DRR to transform IR into canonical (standard) form—eliminating redundancies like `x + 0 → x`, `transpose(transpose(x)) → x`, and folding constants at compile time.

**The C++ Pattern Problem**. Chapter 9's lowering patterns required substantial C++ code (~30 lines per simple transformation):

```cpp
struct SimplifyDoubleTranspose : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op, PatternRewriter &rewriter) const override {
    auto innerTranspose = op.getInput().getDefiningOp<TransposeOp>();
    if (!innerTranspose) return failure();
    rewriter.replaceOp(op, innerTranspose.getInput());
    return success();
  }
};
```

**DRR Solution**. Express the same transformation in ~3 lines of TableGen:

```tablegen
def SimplifyDoubleTranspose : Pat<
  (TransposeOp (TransposeOp $x)),  // Match pattern
  (replaceWithValue $x)            // Replacement
>;
```

TableGen generates the C++ pattern matching code automatically. This is **declarative**: you specify what to match and what to replace it with, not how.

**DRR Pattern Structure**. A DRR pattern has three components:

1. **Match Pattern**: DAG (Directed Acyclic Graph) representing operations to match
2. **Constraints** (optional): Additional conditions (type constraints, attribute checks)
3. **Result Pattern**: DAG representing replacement operations

Example showing all three components:

```tablegen
def AddF32Identity : Pat<
  (AddOp:$result $x, (ConstantOp ConstantAttr<F32Attr, "0.0">)),  // Match pattern (DAG)
  (replaceWithValue $x),                                          // Result pattern
  [(F32Tensor $result)]                                           // Constraint
>;
```

**Common Canonicalization Patterns**. Transformer operations define standard simplifications:

```tablegen
// Identity elimination
def AddZero : Pat<(AddOp $x, (ConstantOp ConstantAttr<F32Attr, "0.0">)), (replaceWithValue $x)>;
def MulOne : Pat<(MulOp $x, (ConstantOp ConstantAttr<F32Attr, "1.0">)), (replaceWithValue $x)>;

// Algebraic simplifications
def TransposeInverse : Pat<(TransposeOp (TransposeOp $x)), (replaceWithValue $x)>;
def DoubleNegate : Pat<(NegateOp (NegateOp $x)), (replaceWithValue $x)>;

// Constant folding (requires C++ helper)
def FoldConstantAdd : Pat<
  (AddOp (ConstantOp $a), (ConstantOp $b)),
  (ConstantOp (AddConstants $a, $b)),
  [(AddConstants $a, $b)]  // C++ helper for compile-time computation
>;
```

**Type Constraints**. Patterns can specify type requirements:

```tablegen
def AddF32Identity : Pat<
  (AddOp:$result $x, (ConstantOp ConstantAttr<F32Attr, "0.0">)),
  (replaceWithValue $x),
  [(F32Tensor $result)]  // Only match float32 tensors
>;
```

**Integration into Operations**. Operations declare canonicalization support:

```tablegen
def Transformer_AddOp : Transformer_Op<"add", [Pure, Commutative]> {
  let summary = "Element-wise addition";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  let hasCanonicalizer = 1;  // Enable canonicalization
}
```

**Using Patterns in Passes**. Generated patterns integrate into pipelines:

```cpp
#include "TransformerOpsPatterns.inc"  // Auto-generated from TableGen

struct TransformerCanonicalizerPass : public PassWrapper<...> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);  // Add all DRR patterns

    // Apply patterns until fixpoint (no more changes)
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

// Or use MLIR's built-in canonicalizer
pm.addPass(mlir::createCanonicalizerPass());
```

The canonicalizer iterates until no more patterns apply (fixpoint), producing maximally simplified IR.

**Constant Folding with C++ Helpers**. Complex canonicalization requires computation:

```cpp
// src/TransformerOps.cpp
Attribute AddConstants(Attribute a, Attribute b) {
  auto aFloat = a.cast<FloatAttr>();
  auto bFloat = b.cast<FloatAttr>();
  double result = aFloat.getValueAsDouble() + bFloat.getValueAsDouble();
  return FloatAttr::get(aFloat.getType(), result);
}
```

This performs arithmetic at **compile time**, replacing runtime computation with constants.

**Why DRR and Canonicalization Matter**. Three key benefits:

1. **Conciseness**: 3 lines vs 30 lines for simple patterns
2. **Optimization Enablement**: Simplified IR exposes opportunities for fusion, constant propagation, and other transformations
3. **Maintainability**: Declarative syntax makes intent obvious, patterns follow consistent structure

**When to Use DRR vs C++ Patterns**: Use DRR for algebraic transformations, pattern-based rewrites, and canonicalization. Use C++ patterns for complex logic, runtime decisions, and lowering passes requiring control flow. DRR complements C++ patterns—use the right tool for each transformation.

## 14.2 Custom OpInterface

Chapter 9 briefly mentioned interfaces but focused on operation definition. **OpInterface** is MLIR's mechanism for polymorphism—defining generic algorithms that work across multiple operations without knowing their specific types. This section demonstrates defining and using custom interfaces.

**The Problem: Generic Algorithms**. Consider shape inference—computing output shapes from input shapes:

```cpp
// Without interfaces: specific code for each operation
if (auto matmul = dyn_cast<MatMulOp>(op)) {
  // MatMul: [M, K] @ [K, N] -> [M, N]
  int64_t M = matmul.getLhs().getShape()[0];
  int64_t N = matmul.getRhs().getShape()[1];
  resultShape = {M, N};
} else if (auto add = dyn_cast<AddOp>(op)) {
  // Add: broadcast shapes
  resultShape = broadcastShapes(add.getLhs().getShape(), add.getRhs().getShape());
} else if (auto relu = dyn_cast<ReLUOp>(op)) {
  // ReLU: shape unchanged
  resultShape = relu.getInput().getShape();
}
// ... hundreds of operations
```

This approach doesn't scale—every shape inference pass needs explicit knowledge of every operation.

**Interface Solution**. Define a `ShapeInferenceOpInterface`:

```tablegen
// inc/ShapeInferenceOpInterface.td
def ShapeInferenceOpInterface : OpInterface<"ShapeInferenceOpInterface"> {
  let description = [{
    Interface for operations that can infer their result shapes from input shapes.
  }];

  let methods = [
    InterfaceMethod<
      /*description=*/"Infer output shape from input shapes",
      /*returnType=*/"SmallVector<int64_t>",
      /*methodName=*/"inferOutputShape",
      /*arguments=*/(ins "ArrayRef<SmallVector<int64_t>>":$inputShapes),
      /*methodBody=*/[{}],  // Operations provide implementation
      /*defaultImplementation=*/[{
        // Default: return unknown shape
        return SmallVector<int64_t>{ShapedType::kDynamic};
      }]
    >
  ];
}
```

This defines an interface with one method: `inferOutputShape()`. Operations that implement this interface must provide the method.

**Implementing the Interface**. Operations declare they implement the interface:

```tablegen
// inc/TransformerOps.td
def Transformer_MatMulOp : Transformer_Op<"matmul", [
    Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>  // Implement interface
  ]> {
  let summary = "Matrix multiplication";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
}

def Transformer_AddOp : Transformer_Op<"add", [
    Pure, Commutative,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
  ]> {
  let summary = "Element-wise addition";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
}
```

`DeclareOpInterfaceMethods` tells TableGen: "This operation implements ShapeInferenceOpInterface."

**Providing Implementations**. Each operation implements its shape inference logic:

- **MatMul**: `[M, K] @ [K, N] → [M, N]` - extract dimensions from inputs, return `{lhsShape[0], rhsShape[1]}`
- **Add**: Broadcast shapes - if identical return either, otherwise return larger rank shape
- **ReLU**: Shape unchanged - return `inputShapes[0]`

```cpp
// MatMul shape inference
SmallVector<int64_t> MatMulOp::inferOutputShape(ArrayRef<SmallVector<int64_t>> inputShapes) {
  auto lhsShape = inputShapes[0];  // [M, K]
  auto rhsShape = inputShapes[1];  // [K, N]
  return {lhsShape[0], rhsShape[1]};  // [M, N]
}
```

**Using the Interface**. Generic algorithms operate polymorphically:

The shape inference pass iterates operations, checks if they implement `ShapeInferenceOpInterface` via `dyn_cast`, collects input shapes from operands, calls `inferOutputShape()`, and updates result types. This works for **any operation** implementing the interface—no knowledge of specific operation types needed.

```cpp
struct ShapeInferencePass : public PassWrapper<...> {
  void runOnOperation() override {
    getOperation().walk([](Operation *op) {
      if (auto shapeOp = dyn_cast<ShapeInferenceOpInterface>(op)) {
        // Collect input shapes from operands
        SmallVector<SmallVector<int64_t>> inputShapes = ...;

        // Polymorphic call - works for MatMul, Add, ReLU, etc.
        SmallVector<int64_t> outputShape = shapeOp.inferOutputShape(inputShapes);

        // Update result type with inferred shape
        op->getResult(0).setType(RankedTensorType::get(outputShape, ...));
      }
    });
  }
};
```

**Interface Benefits**: Extensibility (add operations without modifying passes), code reuse (write generic algorithms once), type safety (TableGen generates compile-time checking), and clear documentation (interface defines expected behavior). Interfaces solve the expression problem—adding new operations without modifying existing code.

## 14.3 Transform Dialect: Modern Optimization

Chapters 11-13 applied basic optimizations (fusion, LICM, CSE) using legacy pass infrastructure. These passes work like black boxes—you configure options and run them, but can't see or control their internal logic. Chapter 14 introduces **Transform Dialect**, MLIR's modern approach to optimization that treats transformations as first-class IR operations. This enables declarative specification, fine-grain control, and transparent reasoning about what transformations apply and when.

**The Legacy Pass Problem**. Traditional MLIR passes configure optimizations through opaque options. Consider a typical optimization pipeline that tiles matrix multiplications and then fuses elementwise operations. With the legacy approach, you create a pass manager, add tiling and fusion passes with specific options, and run them on your module. The problem is that these passes operate as black boxes—you can't see the transformation logic, can't inspect intermediate results between transformations, and can't control fine-grain details beyond what the pass options expose. If a transformation fails, diagnosing why requires diving into pass internals. If you want to apply transformations in a novel combination not anticipated by pass authors, you must write new passes. This inflexibility becomes a significant limitation when optimizing complex models where different operations benefit from different transformation strategies.

Traditional passes also struggle with composability. Each pass runs independently on the entire IR, applying its transformation wherever applicable. You can't easily express "tile this specific matmul with these parameters, then vectorize the result, then fuse it with this specific elementwise operation." The pass model is inherently whole-program: it processes all matching operations uniformly. This works well for many scenarios but becomes restrictive when you need surgical control over specific operations in specific contexts.

**Transform Dialect: Transformations as IR**. Transform dialect solves these problems by representing transformations as IR operations themselves. This creates two levels of IR operating simultaneously. The **payload IR** contains the actual computation being optimized—your linalg operations, function calls, and data flow. The **transform IR** contains transformation operations that manipulate the payload IR. Transform operations operate on handles, which are references to sets of payload IR operations or values. When the transform IR executes, each transform operation performs its transformation on the payload IR objects referenced by its handle operands.

This two-level architecture provides several key advantages. First, transform IR is readable—you can inspect exactly what transformations will apply and in what order, just by reading the IR. Second, transform IR is debuggable—you can insert operations that print handles, inspect payload IR state, or conditionally apply transformations. Third, transform IR is composable—you build complex transformation sequences by composing simple transform operations, with full control over ordering and operand flow. Fourth, transform IR is extensible—adding new transformations requires implementing new transform operations, not modifying the interpreter infrastructure.

**Handles: References to Payload IR**. At the heart of transform dialect lies the concept of handles. A handle is a transform IR value that references one or more payload IR operations or values. Think of a handle as a pointer or reference, but instead of pointing to a single object, it can point to a set of objects. When you execute `transform.structured.match ops{["linalg.matmul"]}`, you get back a handle that references all `linalg.matmul` operations in the payload IR. This handle can then be passed to other transform operations that will operate on those matmul operations.

Handles have types that describe properties of the referenced payload IR. A handle type might indicate "this handle references operations that implement the TileableOp interface" or "this handle references block arguments." These types serve as documentation and enable static verification—the transform IR verifier can check that you're only passing appropriate handles to transform operations. However, verification happens at transform execution time rather than when constructing the transform IR, providing flexibility while maintaining safety.

Handle semantics follow a resource management model similar to memory management. Transform operations that modify payload IR typically consume their input handles, making them invalid for further use. This prevents dangling references—once you've transformed or deleted payload operations, the old handles pointing to them become meaningless. Operations that only inspect payload IR (like pattern matching or shape inference) don't consume handles, allowing you to use the same handle multiple times. This consumption model forces explicit tracking of transformation effects and prevents accidental reuse of stale references.

**Transform Operations and Scripts**. Transform operations implement the actual transformations. The `transform.structured.match` operation finds payload operations matching specified criteria (operation name, interface implementation, custom predicates). The `transform.structured.tile_using_for` operation applies loop tiling to structured operations, breaking large iterations into smaller blocks. The `transform.structured.vectorize` operation converts scalar operations into vector operations exploiting SIMD parallelism. These operations take handles as inputs and produce new handles referencing the transformed payload IR.

A transform script assembles these operations into a transformation sequence. The `transform.sequence` operation provides a container that executes its body operations one by one. The sequence's entry block argument represents the root payload IR operation (typically a module or function). From this root, you build up your transformation pipeline by matching specific operations, applying transformations, and threading handles through the sequence. Consider a typical script that optimizes a matrix multiplication:

```mlir
transform.sequence failures(propagate) {
^bb0(%module: !transform.any_op):
  // Find all linalg.matmul operations in the payload
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %module
    : (!transform.any_op) -> !transform.any_op

  // Tile matmul into smaller blocks for better data locality
  %tiled, %loops = transform.structured.tile_using_for %matmul [32, 32, 32]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Vectorize the tiled operations to exploit SIMD parallelism
  transform.structured.vectorize %tiled : !transform.any_op

  // Clean up with canonicalization patterns
  %func = transform.structured.match ops{["func.func"]} in %module
    : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
}
```

This script explicitly orchestrates the optimization sequence. First, we locate all matmul operations. Then we tile them into 32×32×32 blocks, which returns handles to both the tiled operations and the newly generated loops. Next, we vectorize the tiled operations. Finally, we apply canonicalization patterns to clean up redundant operations introduced by earlier transformations. The entire sequence is visible and controllable—you can inspect it, modify it, or debug it by examining the transform IR.

**Execution Model and Failure Handling**. Transform IR executes through an interpreter that processes transform operations sequentially. The interpreter maintains a mapping between transform IR handles and their corresponding payload IR operations. When you call `transform::applyTransforms()` with a transform script and payload root, the interpreter associates the payload root with the script's entry block argument and begins executing transform operations in order.

Each transform operation's execution modifies this handle-to-payload mapping. Operations that create new payload IR (like tiling creating loops) allocate new handles and associate them with the created operations. Operations that modify payload IR consume their input handles—logically "freeing" them from the mapping—and allocate new handles for the modified operations. Operations that only inspect payload IR neither consume nor allocate handles, leaving the mapping unchanged.

Transform operations report three possible execution outcomes: success, recoverable failure, or irrecoverable failure. Success means the transformation completed as expected. Recoverable failure means the transformation couldn't proceed but hasn't modified the payload IR—for example, a tiling operation might fail because the input operation doesn't satisfy tiling preconditions. Irrecoverable failure means the transformation has partially modified the payload IR in an inconsistent state. Container operations like `transform.sequence` can intercept recoverable failures and perform recovery actions (like skipping the failed operation and continuing), but must propagate irrecoverable failures upward.

This failure model provides fine-grain error handling. A transform script can attempt optimistic transformations and gracefully degrade if they don't apply. The `failures(propagate)` attribute in transform sequences specifies failure handling policy—propagate means any failure stops the sequence and reports the error. Alternative policies allow silent failures or custom recovery logic. This flexibility enables robust transformation pipelines that adapt to varying input IR while maintaining correctness guarantees.

**Handle Invalidation**. When a transform operation consumes a handle, it invalidates not just that handle but potentially many related handles. The invalidation rules follow payload IR structure: consuming an operation handle invalidates all handles to nested operations and to values produced by those operations. Consuming a value handle invalidates handles to the operation producing that value and to nested operations. This transitive invalidation prevents dangling references—if you delete or modify an operation, any handles that might reference it (directly or indirectly) become invalid.

The transform infrastructure can optionally check for invalid handle usage before executing transform operations, but this checking is computationally expensive and typically only enabled during development. Production uses rely on careful handle management and the static analysis provided by `transform-dialect-check-uses` pass, which warns about potential use-after-consume errors without examining payload IR.

**Tiling, Fusion, and Vectorization**. The transform script shown earlier applies three key optimizations: tiling, fusion, and vectorization. Tiling decomposes large loop iterations into nested loops with smaller iteration counts. This improves data locality by working on smaller data chunks that fit better in memory hierarchies. Fusion merges producer-consumer operation pairs, eliminating intermediate buffers and enabling better compiler optimization of the merged computation. Vectorization converts scalar operations into vector operations that process multiple data elements simultaneously using SIMD instructions.

These transformations are widely applicable across different model architectures and problem sizes. Tiling helps whenever computation operates on data larger than available fast memory. Fusion helps whenever operations have producer-consumer relationships with intermediate results that don't need to materialize. Vectorization helps whenever operations perform data-parallel computation on regular data types. The transform dialect implementation in Chapter 14 applies these transformations to GPT's attention and feedforward computations, demonstrating the techniques on a realistic workload.

**Integration and Usage**. Transform scripts execute through the `transform::applyTransforms()` API. This function takes a payload root operation, an optional set of extra handle mappings, a transform operation (the script), and execution options. It creates an interpreter, associates the payload root with the transform script's entry argument, and executes the script. Upon completion, the payload IR has been modified according to the transformations, and the function returns success or failure status.

**Practical Implementation: Pass-Based Optimization**. While the transform dialect represents the modern direction for MLIR optimization, Chapter 14's implementation uses traditional pass infrastructure to achieve similar optimizations. This approach is more accessible and demonstrates the same conceptual transformations (fusion, canonicalization, optimization) through established APIs. The optimization pipeline in [bindings.cpp](ch.14.GPT-Optimized/src/bindings.cpp#L89-L130) constructs a pass manager that orchestrates tensor-level and buffer-level optimizations:

```cpp
bool TransformerCompiler::lowerToLLVM(ModuleOp module) {
  PassManager pm(&context_);
  
  // Lower custom dialect to Linalg operations (structured linear algebra)
  pm.addNestedPass<func::FuncOp>(createLowerTransformerToStandardPass());
  
  // Canonicalization and common subexpression elimination
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  
  // Linalg optimizations on tensor IR (before bufferization)
  pm.addPass(createLinalgGeneralizeNamedOpsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLinalgElementwiseOpFusionPass());  // Fusion
  pm.addPass(createCanonicalizerPass());
  
  // Bufferization: tensor → memref
  // [... bufferization setup ...]
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));
  
  // Lower Linalg operations to loops
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  
  // Loop optimizations
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createCanonicalizerPass());
  
  // [... LLVM conversion ...]
}
```

This pipeline demonstrates several key concepts from transform dialect mapped to passes. The `LinalgElementwiseOpFusionPass` performs producer-consumer fusion similar to transform dialect's fusion operations—it identifies element-wise operations that can be merged to eliminate intermediate allocations. The canonicalization passes (`createCanonicalizerPass()`) apply local simplifications similar to pattern matching, repeatedly simplifying the IR after each major transformation. Loop invariant code motion (`createLoopInvariantCodeMotionPass()`) performs optimization after lowering to loops, hoisting computations out of loops when they don't depend on loop variables.

The structured progression through Linalg → loops → LLVM mirrors transform dialect's philosophy: maintain high-level semantics as long as possible for optimization, then progressively lower. Operating on Linalg tensor operations (before bufferization) enables more aggressive transformations than would be possible on low-level buffer operations. The fusion pass, for example, can recognize that producer and consumer operations both work on tensor dimensions and merge their iteration spaces—something much harder to analyze after bufferization introduces explicit memory operations. This demonstrates that while transform dialect provides finer control and better composability for complex optimization scenarios, traditional passes can achieve similar optimization objectives for many workloads.

**Comparison to Alternatives**. Transform dialect occupies a middle ground between pattern rewriting and pass infrastructure. Pattern rewriting (DRR, C++ patterns) excels at local transformations—simple rewrites that pattern match a small IR fragment and replace it. Patterns can't orchestrate multi-step transformations or maintain state across rewrites. Passes excel at whole-program transformations—uniform transformations applied to all matching operations in the IR. Passes can orchestrate complex transformations but lack fine-grain control over specific operation instances.

Transform dialect provides fine-grain control with multi-step orchestration. You can target specific operation instances (not all instances of an operation), apply different transformations to different instances based on context, and explicitly sequence transformation steps with full visibility into intermediate results. This capability becomes crucial when optimizing models where different layers or operations require different optimization strategies. A typical optimization workflow might use patterns for local simplifications, transform dialect for controlled optimization of hot paths, and passes for uniform whole-program transformations, leveraging each tool's strengths.

## 14.4 KV Caching: Algorithmic Optimization

While compiler optimizations have limited impact on nano GPT, **KV caching** provides dramatic speedup at any scale—it's an algorithmic win, not a hardware-dependent optimization.

**The Redundancy Problem**. Autoregressive generation recomputes attention for all tokens every iteration:

```python
tokens = [prompt_tokens]

for _ in range(max_new_tokens):
    logits = gpt_forward(tokens)  # Recomputes keys/values for ALL tokens
    next_token = sample(logits[-1])
    tokens.append(next_token)
```

At token N, we compute keys/values for tokens 0..N-1. At token N+1, we **recompute** the same keys/values—but they haven't changed! This is O(N²) complexity.

**KV Cache Solution**. Cache computed keys and values:

```python
# Initialize caches: [num_layers, max_seq_len, d_model]
k_caches = [zeros((max_seq_len, d_model)) for _ in range(num_layers)]
v_caches = [zeros((max_seq_len, d_model)) for _ in range(num_layers)]

# Process prompt: fill caches
for pos, token in enumerate(prompt_tokens):
    for layer in range(num_layers):
        k_caches[layer][pos] = compute_key(token, layer)
        v_caches[layer][pos] = compute_value(token, layer)

# Generate new tokens (incremental)
for step in range(max_new_tokens):
    # Only compute Q/K/V for NEW token
    new_token_embedding = embedding_table[tokens[current_pos]]

    for layer in range(num_layers):
        q = compute_query(new_token_embedding, layer)
        k = compute_key(new_token_embedding, layer)
        v = compute_value(new_token_embedding, layer)

        # Cache K/V for future iterations
        k_caches[layer][current_pos] = k
        v_caches[layer][current_pos] = v

        # Attention using cached keys/values
        scores = q @ k_caches[layer][:current_pos+1].T
        weights = softmax(scores)
        output = weights @ v_caches[layer][:current_pos+1]

    current_pos += 1
```

Complexity: O(N²) → O(N). Each iteration computes only one new Q/K/V.

**What is KV Cache?** In transformer attention, each token computes three vectors: Query (Q), Key (K), and Value (V). During autoregressive generation, tokens attend to all previous tokens—but the K and V vectors for previous tokens don't change. KV caching stores these previously computed K/V tensors to avoid redundant recomputation. For a sequence of length N, without caching we recompute O(N) keys/values at each step; with caching we compute just one new K/V pair. This reduces generation complexity from O(N²) to O(N).

Chapter 14's compiler infrastructure includes API hooks for KV-cached forward passes (`gpt_forward_prefill` and `gpt_forward_decode`), demonstrating how MLIR can support stateful inference patterns. Chapter 16 uses these APIs to implement complete production serving with KV cache management, paged memory allocation, and prefix sharing.

## 14.8 Summary

Chapter 14 introduced production-grade optimization techniques spanning declarative transformations (DRR, Transform dialect), advanced dialect features (interfaces, canonicalization), and compiler support for stateful inference patterns. These techniques represent modern MLIR practice—the same approaches used in production compilers at Google, Meta, and NVIDIA.

**Looking Ahead**. Chapter 15 introduces GPU concepts: CUDA programming model, memory hierarchy (global, shared, registers), kernel programming, and MLIR's GPU dialect. Chapter 16 completes the book with production serving: KV cache management, continuous batching, radix cache for prefix sharing, and chunked prefill. These chapters build on Chapter 14's optimization foundation to achieve true production-scale performance.

Chapter 14 completed the optimization arc: from basic operations (Chapters 1-9) through transformer architecture (Chapters 10-13) to production-grade optimizations. You now understand modern MLIR compilation—declarative, composable, and extensible. The techniques are production-ready; the scale is educational. Apply these to real models, and you'll achieve the theoretical speedups documented here.