# Chapter 14: Production-Grade Optimizations

Chapters 1-13 built a complete journey: MLIR fundamentals (1-9), transformer architecture (10-12), and GPT implementation (13). Chapter 13's GPT uses Linalg dialect for structured operations, enabling basic compiler optimizations (elementwise fusion, LICM, CSE) that provide speedups. The implementation is correct and demonstrates key architectural patterns, but production systems require orders-of-magnitude greater performance.

Modern MLIR compilers achieve dramatic speedups through aggressive loop transformations and algorithmic improvements. **Transform Dialect** applies tiling to create cache-friendly memory access patterns, vectorization to exploit SIMD units, and advanced fusion to eliminate intermediate memory traffic—providing 3-5× speedups for production models. **KV caching** changes the generation algorithm itself, eliminating O(N²) redundant computation by caching attention keys/values across generation steps—providing 10-100× speedups. **Declarative Rewrite Rules (DRR)** express optimizations as pattern-match-rewrite specifications in TableGen. **OpInterface** enables polymorphic algorithms operating on operation categories rather than concrete types. These techniques represent the state-of-the-art in production compilers like IREE, Torch-MLIR, and NVIDIA's systems.

This chapter introduces all four techniques and implements Transform Dialect with KV caching in the codebase. DRR and OpInterface are presented as educational material—widely used in production but not implemented here to maintain focus on the optimization pipeline. Section 14.1 demonstrates DRR syntax and semantics. Section 14.2 shows OpInterface design. Section 14.3 implements Transform Dialect optimization pipeline. Section 14.4 introduces KV caching concepts for generation. Our nano GPT cannot demonstrate the full performance gains visible in production models, but the implementations are architecturally identical and production-ready.

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

This section demonstrates the **production pattern** for Transform Dialect usage: embedding transform scripts as string literals in C++ code, parsing them once, caching the result, and applying the cached transforms to compilation targets. This approach is used by major MLIR-based compilers and eliminates the complexity of external file dependencies while maintaining the expressiveness of Transform Dialect IR.

**The Legacy Pass Problem**. Traditional MLIR passes configure optimizations through opaque options. Consider a typical optimization pipeline that tiles matrix multiplications and then fuses elementwise operations. With the legacy approach, you create a pass manager, add tiling and fusion passes with specific options, and run them on your module:

```cpp
// Legacy pass approach - black box transformations
PassManager pm(&context);
pm.addPass(createCanonicalizerPass());  // What patterns? How many iterations?
pm.addPass(createCSEPass());            // What heuristics?
pm.run(module);                          // Did it work? What changed?
```

The problem is that these passes operate as black boxes—you can't see the transformation logic, can't inspect intermediate results between transformations, and can't control fine-grain details beyond what the pass options expose. If a transformation fails, diagnosing why requires diving into pass internals.

**Transform Dialect Solution**. Transform dialect solves this by making transformations explicit and composable. Instead of opaque passes, you write declarative transform scripts that clearly specify what transformations to apply:

```mlir
// Transform dialect - explicit, inspectable transformations
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // Apply canonicalization patterns to simplify IR
  transform.apply_patterns to %arg0 {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
}
```

This transform script offers several advantages over legacy passes. You can read it directly to understand exactly what optimizations will apply, making the transformation logic transparent rather than hidden inside pass implementations. Each transformation step is explicit, so you can inspect intermediate states and understand exactly how the IR evolves. The script composes transformations clearly—you can see how operations flow from one transformation to the next through handle threading. Finally, transform scripts are parameterizable and reusable across different compilation contexts, unlike monolithic pass implementations that hardcode their transformation strategies.

**Production Pattern: Embedded Transform Scripts**. Major MLIR compilers embed transform scripts directly in C++ code as string literals rather than loading them from external files. This approach, currently necessary in LLVM 21 due to limited C++ APIs for Transform Dialect operations, provides the expressiveness of Transform Dialect IR while avoiding deployment complexity. Future LLVM versions are expected to provide direct C++ APIs for constructing transform operations programmatically, eliminating the need for embedded strings. For now, the embedded string pattern used by production systems like Torch-MLIR and IREE works well—the script becomes part of the binary with no file paths to configure, no version mismatches between compiler and script files, and no runtime file I/O overhead.

The implementation in TransformDialectOptimization.cpp follows this pattern with three key components. First, the transform script is defined as a compile-time constant string literal using C++ raw string syntax. This embeds the MLIR transform IR directly in the source code where it's easy to read and modify:

```cpp
static constexpr const char *kOptimizeTransformIR = R"mlir(
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  transform.apply_patterns to %arg0 {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
}
)mlir";
```

Second, the implementation parses this string once and caches the result using a thread-safe initialization pattern. The `getCachedTransformModule` function uses `std::call_once` to ensure that even with concurrent compilations, the transform script is parsed exactly once. Subsequent calls return a clone of the cached module, which is fast and avoids reparsing overhead:

```cpp
static OwningOpRef<ModuleOp> getCachedTransformModule(MLIRContext *ctx) {
  static OwningOpRef<ModuleOp> cachedModule;
  static std::once_flag initFlag;
  
  std::call_once(initFlag, [ctx]() {
    ParserConfig config(ctx);
    cachedModule = parseSourceString<ModuleOp>(kOptimizeTransformIR, config);
  });
  
  return cachedModule.get() ? cachedModule.get().clone() : OwningOpRef<ModuleOp>();
}
```

Third, the public API retrieves the cached transform module, locates its top-level transform operation, and applies it to the compilation target using the Transform Dialect interpreter. Each compilation gets its own clone of the transform module, preventing interference between concurrent compilations while avoiding the cost of reparsing:

```cpp
LogicalResult applyTransformDialectOptimizations(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  
  auto transformModule = getCachedTransformModule(ctx);
  
  Operation *topLevelTransformOp = nullptr;
  for (Operation &op : transformModule->getBodyRegion().getOps()) {
    if (auto transformOp = dyn_cast<transform::TransformOpInterface>(&op)) {
      topLevelTransformOp = transformOp;
      break;
    }
  }
  
  transform::TransformOptions options;
  return transform::applyTransforms(
      module.getOperation(),
      cast<transform::TransformOpInterface>(topLevelTransformOp),
      {},
      options);
}
```

This pattern works well in production.

**Transform Dialect Fundamentals**. Transform dialect operates on two levels of IR simultaneously. The **payload IR** contains your actual program—operations like `linalg.matmul`, tensor values, and function definitions that represent the computation you're compiling. The **transform IR** contains operations that manipulate the payload IR—operations like `transform.structured.tile` or `transform.apply_patterns` that describe transformations to apply. Transform operations work with **handles**, which are transform IR values that reference sets of payload IR operations or values. When a transform operation executes, it operates on the payload IR objects referenced by its handle operands.

Consider a simple example that applies canonicalization patterns. The transform sequence takes a handle to the root payload operation as its entry block argument. The `transform.apply_patterns` operation then takes this handle as input and applies canonicalization patterns to all operations referenced by that handle. Since the handle references the entire module, canonicalization applies throughout the payload IR:

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  transform.apply_patterns to %arg0 {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
}
```

The `transform.apply_patterns` operation serves as the Transform Dialect's equivalent to legacy canonicalization and CSE passes. Instead of creating pass objects and running them through a pass manager, you declaratively specify pattern application in transform IR. The canonicalization patterns include constant folding, common subexpression elimination, dead code elimination, and various algebraic simplifications. This single transform operation replaces what would previously require separate calls to `createCanonicalizerPass()` and `createCSEPass()`, with the advantage that the pattern application is explicit and visible in the transform script:

```mlir
transform.apply_patterns to %target {
  transform.apply_patterns.canonicalization
} : !transform.any_op
```

Failure handling in transform scripts uses an explicit attribute on the sequence operation. The `failures(propagate)` attribute specifies that if any transformation fails—for example, if a tiling operation can't apply because its target operations don't satisfy tiling preconditions—the entire sequence should stop and report the error. Alternative policies like `failures(suppress)` allow transformations to fail silently, with the sequence continuing to execute subsequent operations. This explicit failure control enables robust transformation pipelines that can attempt optimistic transformations and gracefully degrade if they don't apply:

```mlir
transform.sequence failures(propagate) {
  // Any failure stops execution and reports an error
}

transform.sequence failures(suppress) {
  // Failures are ignored; execution continues
}
```

**Transform Script Composition**. More complex transform scripts demonstrate how transformations compose through handle threading. Consider a script that optimizes matrix multiplications by tiling them into smaller blocks, vectorizing the tiled computation, and cleaning up with canonicalization. The script starts by locating all `linalg.matmul` operations in the module using pattern matching. The `transform.structured.match` operation returns a handle referencing all matching operations. This handle then flows to the tiling operation, which tiles each referenced matmul and returns handles to both the tiled operations and the newly created loops. The vectorization operation takes the handle to tiled operations and vectorizes them to exploit SIMD parallelism. Finally, canonicalization cleans up any redundant operations introduced by earlier transformations:

```mlir
transform.sequence failures(propagate) {
^bb0(%module: !transform.any_op):
  // Simplify IR before optimization
  transform.apply_patterns to %module {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  
  // Find matrix multiplications to optimize
  %matmuls = transform.structured.match ops{["linalg.matmul"]} in %module
    : (!transform.any_op) -> !transform.any_op
  
  // Tile into 32×32×32 blocks for cache locality
  %tiled, %loops = transform.structured.tile_using_for %matmuls [32, 32, 32]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  
  // Vectorize for SIMD parallelism
  transform.structured.vectorize %tiled : !transform.any_op
  
  // Clean up redundant operations
  transform.apply_patterns to %module {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
}
```

This script demonstrates the transformation pipeline's structure. Initial canonicalization simplifies the IR, making subsequent optimizations more effective by eliminating redundant operations early. Pattern matching locates specific optimization targets—in this case matrix multiplications—without affecting other operations. Tiling improves data locality by breaking large computations into smaller blocks that fit better in cache hierarchies. Vectorization converts scalar operations into vector operations that process multiple data elements simultaneously. Final canonicalization cleans up any inefficiencies introduced by earlier stages, such as redundant address calculations or loop bounds. Each step is explicit and independently modifiable, unlike monolithic passes where transformation logic is hidden inside pass implementations.

**Integration with Compilation Pipeline**. Transform dialect optimizations integrate into the lowering pipeline early, operating on high-level tensor operations before bufferization converts them to low-level buffer operations. This placement is critical because tensor operations preserve semantic information that enables more aggressive optimization than would be possible after bufferization. After lowering custom dialect operations to Linalg (which represents structured operations over tensors), the compilation pipeline applies transform dialect optimizations. These optimizations happen before bufferization intentionally—tensor-level semantics make it easy to reason about operation fusion, tiling, and vectorization. After transform dialect optimizations complete, the pipeline continues with bufferization (converting tensors to buffers) and subsequent lowering stages:

```cpp
bool lowerToLLVM(ModuleOp module) {
  PassManager pm(&context_);
  
  // Lower custom dialect to Linalg structured operations
  pm.addNestedPass<func::FuncOp>(createLowerTransformerToStandardPass());
  
  // Apply Transform Dialect optimizations at the tensor level
  // This replaces what would previously be multiple passes:
  // canonicalization, CSE, fusion, etc.
  if (failed(applyTransformDialectOptimizations(module))) {
    return false;
  }
  
  // Continue with bufferization and further lowering
  pm.addPass(bufferization::createOneShotBufferizePass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  // ... rest of lowering pipeline
  
  return succeeded(pm.run(module));
}
```

Operating at the tensor level enables more powerful optimizations than would be possible on bufferized code. Tensor operations have clear producer-consumer relationships that make fusion straightforward—you can identify that one operation produces a tensor that another operation immediately consumes, and merge them into a single operation that eliminates the intermediate tensor. Buffer operations obscure these relationships with explicit memory allocations, loads, and stores, making producer-consumer analysis much more difficult. Similarly, tiling tensor operations involves manipulating iteration spaces over abstract tensors, while tiling buffer operations requires reasoning about pointer arithmetic and memory access patterns. Transform dialect leverages these high-level semantics by operating before bufferization, then letting subsequent lowering stages handle the mechanical translation to low-level code.

**Comparing Transform Dialect and Legacy Passes**. Transform dialect and legacy passes offer different tradeoffs for optimization. Legacy passes operate as black boxes—you invoke a pass and it applies its transformations, but you can't inspect or modify the transformation logic without changing pass internals. Transform dialect makes transformations explicit in transform IR that you can read, modify, and debug. Legacy passes provide coarse-grained control through options (like tile sizes or fusion heuristics), while transform dialect provides fine-grained control where you specify exactly which operations to transform and how. Debugging legacy passes requires understanding their internal implementation and often involves instrumentation or stepping through pass code, while debugging transform dialect means inspecting transform IR and observing how handles flow through transformations. Passes compose implicitly through pass manager ordering, where you add passes in sequence and hope they interact correctly. Transform dialect composes explicitly through handle threading, where you can see exactly how one transformation's outputs become another's inputs. When you need new optimization strategies, passes require writing new pass implementations, while transform dialect lets you write new scripts that compose existing transform operations in novel ways.

These tradeoffs inform when to use each approach. Transform dialect excels when you need fine-grained control over specific operations—for example, applying different tiling strategies to different matrix multiplications based on their context. It's ideal when you want transparent, inspectable transformations that you can debug by examining transform IR. Complex optimization pipelines with multiple interdependent stages benefit from transform dialect's explicit composition and handle threading. When you need to debug transformation sequences, transform dialect's visibility into intermediate states and explicit transformation steps simplifies diagnosis. Conversely, legacy passes work well for simple, uniform transformations that should apply uniformly throughout the IR. If existing passes already implement the optimizations you need, using them directly is straightforward. Well-established optimization patterns like canonicalization and CSE have mature pass implementations that work reliably. When coarse-grained control is sufficient—you just want to enable an optimization without fine-tuning its application—passes provide a simpler interface than writing transform scripts.

Chapter 14's implementation demonstrates transform dialect in production, showing how to embed scripts as string literals, parse them once and cache the result, and apply explicit transformations through the Transform Dialect interpreter. This pattern scales from the simple canonicalization shown here to complex multi-stage optimization pipelines with tiling, fusion, vectorization, and specialized transformations. Production compilers use this approach because it combines transform dialect's expressiveness with zero deployment complexity, making it suitable for both development experimentation and production deployment.

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

Chapter 14's codebase includes Python-level KV caching implementation in [generation.py](ch.14.GPT-Optimized/generation.py), with the `generate_cached` function demonstrating how to maintain key/value caches across generation steps. The compiler infrastructure also provides MLIR-level hooks (`gpt_forward_prefill` and `gpt_forward_decode`) for supporting stateful inference patterns, though the detailed implementation of these functions is beyond the scope of this educational chapter.

## 14.5 Summary

Chapter 14 introduced production-grade optimization techniques spanning declarative transformations (DRR, Transform dialect), advanced dialect features (interfaces, canonicalization), and compiler support for stateful inference patterns. These techniques represent modern MLIR practice—the same approaches used in production compilers at Google, Meta, and NVIDIA.

**Looking Ahead**. Chapter 15 introduces GPU concepts: CUDA programming model, memory hierarchy (global, shared, registers), kernel programming, and MLIR's GPU dialect. Chapter 16 completes the book with production serving: KV cache management, continuous batching, radix cache for prefix sharing, and chunked prefill. These chapters build on Chapter 14's optimization foundation to achieve true production-scale performance.

Chapter 14 completed the optimization arc: from basic operations (Chapters 1-9) through transformer architecture (Chapters 10-13) to production-grade optimizations. You now understand modern MLIR compilation—declarative, composable, and extensible. The techniques are production-ready; the scale is educational. Apply these to real models, and you'll achieve the theoretical speedups documented here.