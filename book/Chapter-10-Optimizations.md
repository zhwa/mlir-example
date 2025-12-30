# Chapter 10: Optimization Passes and Performance

Chapters 8 and 9 taught dialect design: building high-level operations (the `nn` dialect) using Python strings (pedagogical approach) and TableGen (production approach), then lowering them to standard MLIR dialects. We created operations like `nn.matmul`, implemented lowering patterns converting them to `linalg` operations, and compiled everything to executable code. That pipeline works correctly—operations compute the right results—but it's not optimized. The generated code performs redundant work, misses SIMD opportunities, and allocates unnecessary temporary buffers. Production AI compilers must deliver not just correctness but **performance**.

This chapter explores MLIR's **optimization infrastructure**: passes that transform IR to execute faster while preserving semantics. We'll examine Linalg fusion (merging adjacent operations to reduce memory traffic), loop-invariant code motion (hoisting computations out of loops), and vectorization (converting scalar loops to SIMD operations using the `vector` dialect). Unlike Chapters 8-9 where we built a dialect from scratch, Chapter 10 reuses the `nn` dialect unchanged—operations stay the same, Python API stays the same, but the compilation pipeline adds optimization passes between high-level lowering and final code generation. This separation of concerns mirrors production systems: stable user-facing APIs, evolving backend optimizations.

The chapter progresses from understanding why optimizations matter (performance gaps between naive and optimized code), through MLIR's pass infrastructure (how passes compose and interact), to specific optimization techniques (fusion algorithms, vectorization patterns). We'll see that MLIR's multi-level IR philosophy enables optimizations at different abstraction levels—Linalg fusion works on structured operations, loop optimizations work on SCF dialect, vectorization works on affine loops—each level providing optimization opportunities unavailable at others. By the end, you'll understand not just how to write optimized pipelines, but **why** MLIR's architecture makes sophisticated optimizations composable and maintainable.

## 10.1 Why Optimizations Matter: The Performance Gap

Before examining optimization techniques, let's quantify the problem. Consider a simple two-operation computation: matrix multiplication followed by element-wise ReLU. Chapter 9's unoptimized pipeline generates code that:

1. **Allocates output buffer** for matmul
2. **Computes matmul**, writing results to buffer
3. **Allocates another buffer** for ReLU output
4. **Reads matmul results**, applies ReLU, writes to second buffer

For a 128×128 matrix, this performs:
- **2 memory allocations** (overhead)
- **128×128 = 16384 writes** matmul output
- **16384 reads + 16384 writes** (ReLU pass)
- **Total**: 16384*3=49152 memory operations plus allocation overhead

Optimized code using fusion:

1. **Allocates single output buffer**
2. **Computes matmul and applies ReLU** in one pass, writing fused results

This performs:
- **1 memory allocation**
- **16384 writes** (fused results)
- **Total**: 16384 memory operations

The optimized version performs significantly less memory traffic. On modern CPUs where memory bandwidth often bottlenecks computation, reducing memory operations improves performance. Crucially, fusion provides **temporal locality**: the unoptimized version writes matmul results to DRAM then immediately reads them back, while the fused version keeps intermediate values in L1 cache or CPU registers. This cache residency makes the performance delta even larger than raw operation counts suggest. The optimized code also enables vectorization (processing multiple elements per instruction with SIMD) and reduces allocation overhead (one malloc instead of two). These optimizations combine to substantially improve execution efficiency.

**Why Manual Optimization Doesn't Scale**. You might think "just write fused kernels manually"—implement `matmul_relu_fused()` by hand. This works for small operator sets but fails at scale:

- **Combinatorial explosion**: N operations produce O(N²) pairwise fusions, O(N³) three-way fusions, etc. TensorFlow has 1000+ operations; manual fusion is infeasible.
- **Maintainability**: Hand-coded kernels become technical debt. When matmul gets updated (new algorithm, hardware support), every fusion involving matmul must be updated too.
- **Portability**: Manual SIMD code (AVX2, NEON, etc.) is architecture-specific. MLIR's vectorization works on portable IR, generating appropriate SIMD for each target.
- **Optimization interactions**: Fusion enables vectorization, vectorization changes register pressure, register pressure affects loop tiling decisions. Manual optimization struggles with these interdependencies.

MLIR's pass-based architecture solves these problems: each pass implements one optimization concern, passes compose automatically, and the framework handles interaction complexity. Write fusion once (works on any Linalg operations), write vectorization once (works on any loops), combine them in a pipeline. When operations evolve, passes adapt automatically because they work on generic IR patterns, not specific operation names.

**The Performance Mindset**. Throughout this chapter, remember: optimizations preserve semantics while changing performance characteristics. The correctness guarantee—optimized code produces identical numerical results to unoptimized code—comes from MLIR's verification infrastructure. Every pass must maintain IR validity; passes can't introduce bugs by construction (assuming the pass itself is correct). This separation lets compiler engineers focus on performance without debugging correctness regressions for every optimization. Trust the framework's verification, focus on algorithmic improvements.

## 10.2 MLIR's Pass Infrastructure Revisited

Chapter 3 introduced passes conceptually—transformations applied to IR in sequence. Chapter 10 requires deeper understanding: how passes interact, what guarantees they provide, and how to compose them effectively. Let's examine the PassManager infrastructure used throughout Chapter 10's optimization pipeline.

**Pass Types and Scopes**. MLIR distinguishes passes by the IR they operate on:

```cpp
// Module pass: operates on entire module
class MyModulePass : public OperationPass<ModuleOp> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Transform entire module
  }
};

// Function pass: operates on individual functions
class MyFuncPass : public OperationPass<func::FuncOp> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    // Transform this function only
  }
};
```

The scope matters for parallelization: function passes can run on multiple functions concurrently (MLIR's PassManager supports multi-threading), while module passes serialize execution. Chapter 10 uses mostly function-scoped passes (canonicalizer, fusion, vectorization) for better parallelism.

**Pass Dependencies and Ordering**. Passes declare dependencies implicitly through their effects:

```cpp
void MyPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<linalg::LinalgDialect, arith::ArithDialect>();
}
```

This tells the PassManager "I create linalg and arith operations," ensuring those dialects are loaded before the pass runs. Explicit ordering comes from pipeline construction:

```cpp
PassManager pm(&context);
pm.addPass(createLinalgGeneralizeNamedOpsPass());  // Must run first
pm.addPass(createCanonicalizerPass());             // Cleans up generalized ops
pm.addPass(createLinalgElementwiseOpFusionPass()); // Fuses cleaned ops
```

The order matters: fusion works better after canonicalization removes redundancy; generalization must precede fusion because fusion patterns match generic ops. Getting this order right requires understanding what each pass does and what invariants it expects. Chapter 10's pipeline evolved through experimentation and understanding MLIR's optimization design patterns.

**Verification and Invariants**. After each pass, MLIR verifies IR validity:

```cpp
if (failed(pm.run(module))) {
  llvm::errs() << "Pass pipeline failed\n";
  return failure();
}
```

Verification checks:
- **SSA dominance**: Values used after they're defined
- **Type consistency**: Operations match their type constraints
- **Region structure**: Blocks properly terminated, regions well-formed
- **Operation validity**: Each operation satisfies its verifier

If any check fails, the PassManager aborts with diagnostics pointing to the invalid IR and the pass that created it. This fast-fail approach prevents cascading errors: a buggy pass corrupts IR, next pass sees corruption, verification catches it before confusion spreads. In practice, verification overhead is negligible compared to transformation cost—a few percent slowdown for massive debugging benefits.

**The Canonicalizer: MLIR's Swiss Army Knife**. Notice how Chapter 10's pipeline intersperses `createCanonicalizerPass()` between almost every optimization:

```cpp
pm.addPass(createLinalgGeneralizeNamedOpsPass());
pm.addPass(createCanonicalizerPass());  // Clean up
pm.addPass(createLinalgElementwiseOpFusionPass());
pm.addPass(createCanonicalizerPass());  // Clean up again
```

The canonicalizer applies **canonicalization patterns**—simple, local optimizations that put IR in standard form:

- **Dead code elimination**: Remove unused operations
- **Constant folding**: Evaluate operations on constants at compile time
- **Algebraic simplification**: `x + 0 → x`, `x * 1 → x`
- **Redundancy elimination**: `transpose(transpose(x)) → x`

Each operation can register canonicalization patterns (Chapter 9 showed this for custom operations). The canonicalizer runs all registered patterns repeatedly until fixpoint—IR stops changing. Importantly, the canonicalizer is **greedy**: it applies all patterns without a cost model, assuming simplification is always beneficial. This distinguishes it from optimization passes like fusion, which should ideally evaluate whether transformations improve performance before applying them. This "cleanup between major transformations" pattern is ubiquitous in MLIR pipelines: big passes create messy IR (generalization creates verbose generic ops, fusion creates complicated indexing), canonicalizer simplifies, next pass works on cleaner input.

**Why Not One Big Pass?** You might wonder: why many small passes instead of one comprehensive optimizer? Several reasons:

1. **Composability**: Small passes combine in exponentially many ways. Fusion + vectorization, fusion + tiling, tiling + parallelization—combinatorial possibilities from linear components.

2. **Debuggability**: When optimization goes wrong, smaller passes narrow the problem. Add `-mlir-print-ir-after-all` and see exactly which pass introduced the issue.

3. **Maintainability**: 100-line pass files beat 10,000-line monoliths. Each pass has one job, making code easier to understand, test, and modify.

4. **Selective application**: Some optimizations help certain workloads, hurt others. Small passes let you customize pipelines per use case.

The cost is coordination complexity—ensuring passes don't interfere or assume inconsistent invariants. MLIR addresses this through verification (catches violations), documentation (each pass specifies preconditions), and community conventions (shared patterns across dialects). Chapter 10's pipeline follows these conventions, providing a template for your own optimization pipelines.

### 10.2.5 The Affine Dialect: Why This Book Uses Linalg Instead

Before diving into specific optimizations, we should address a dialect you might encounter in other MLIR resources: the **affine dialect**. Chapter 10—and indeed this entire book—**eschews the affine dialect in favor of the more flexible Linalg and SCF dialects**. This section explains what affine is, why it exists, and why we made this deliberate architectural choice.

**Important Distinction: AffineMap vs Affine Dialect**. You've already seen `AffineMap` extensively—it's the mathematical type used in Linalg's indexing maps:
```mlir
linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,  // AffineMap (just a type!)
    affine_map<(d0, d1) -> (d0, d1)>
  ],
  // ...
}
```

`AffineMap` is a **core MLIR type** representing linear transformations—it's used by Linalg, Vector, and other dialects. The **affine dialect**, by contrast, is a set of **operations** (`affine.for`, `affine.load`, `affine.store`) for polyhedral loop optimization. They're different concepts that happen to share the "affine" name.

**What Is the Affine Dialect?** The affine dialect provides loop operations with strict restrictions:
```mlir
// Affine loop: bounds must be affine expressions (linear functions)
affine.for %i = 0 to 128 {
  affine.for %j = %i to 256 {  // %j depends linearly on %i
    %v = affine.load %A[%i, %j] : memref<128x256xf32>
  }
}
```

Compared to SCF's flexibility, affine loops are **highly restrictive**:
- Loop bounds must be affine expressions (linear combinations: `%i + 5`, `2*%i + %j`, etc.)
- Memory accesses must use affine indices (`A[%i]`, `A[2*%i + %j]` ✓, but not `A[%i * %j]` ✗)
- No arbitrary control flow inside loops

These restrictions enable **polyhedral optimization**: automatic loop interchange, tiling, fusion, and parallelization. Compilers can prove safety (no data races) and profitability (better cache locality) mathematically.

**Why This Book Doesn't Use Affine**. We made a deliberate choice to use Linalg + SCF instead of affine dialect:

**Reason 1: Linalg is Higher-Level and More Expressive**
```mlir
// Linalg: declarative, operation-centric
linalg.matmul ins(%A, %B : memref<MxK>, memref<KxN>)
              outs(%C : memref<MxN>)
// Compiler generates optimal loop nest for target

// Affine: explicit loops (requires manual loop nest design)
affine.for %i = 0 to %M {
  affine.for %j = 0 to %N {
    affine.for %k = 0 to %K {
      // Must manually express multiply-accumulate
    }
  }
}
```

Linalg lets us focus on **what to compute** (matmul semantics), not **how to loop** (iteration strategy). Linalg's structured operations enable pattern matching for fusion—recognizing "matmul followed by ReLU" is easier than analyzing arbitrary loop nests.

**Reason 2: Linalg's Transformation Infrastructure is More Mature**

MLIR's Linalg dialect has extensive transformation libraries:
- **Fusion patterns** (Section 10.3): `createLinalgElementwiseOpFusionPass()`
- **Tiling strategies**: Tile sizes, interchange orders, parallelization hints
- **Code generation targets**: CPU, GPU, TPU-specific lowerings

Affine dialect optimizations require deeper polyhedral compilation knowledge and offer fewer pre-built passes. For pedagogical purposes, Linalg's high-level abstractions teach optimization concepts more directly.

**Reason 3: Production AI compilers Use Linalg**

The AI compiler ecosystem converged on Linalg:
- **Torch-MLIR**: PyTorch → Torch dialect → Linalg → SCF/Affine → LLVM
- **IREE**: TensorFlow/JAX → Linalg-based optimizations → HAL
- **StableHLO**: JAX's dialect, lowers to Linalg for optimization

While some compilers use affine for specific kernels (XLA's polyhedral scheduler), the mainstream path is Linalg-first. Learning Linalg prepares you for real-world AI compiler work.

**Reason 4: Dynamic Shapes**

Chapter 2 introduced dynamic shapes (`memref<?x?xf32>`). Affine dialect struggles with dynamic bounds—polyhedral analysis assumes static iteration spaces. Linalg handles dynamic shapes naturally:
```mlir
linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
              outs(%C : memref<?x?xf32>)  // Works with dynamic sizes
```

SCF loops handle dynamic bounds trivially (`scf.for %i = 0 to %N`). Affine requires escaping to SCF for dynamic cases, complicating the pipeline.

**When Affine IS Useful**. Despite our choice to use Linalg, the affine dialect has legitimate use cases:

1. **Static, Compute-Bound Kernels**: When you have fixed-size loops with complex iteration dependencies, affine's polyhedral analysis can find optimal loop orders and tiling strategies automatically. Useful for DSP, image processing, and some HPC workloads.

2. **Research and Education in Polyhedral Compilation**: If you're studying loop optimization theory, affine dialect provides a clean implementation of polyhedral models. Academic projects like Polygeist use it to teach compiler transformations.

3. **Integration with Existing Polyhedral Tools**: If you have code generated by tools like Pluto or ISL (polyhedral schedulers), affine dialect provides a natural MLIR representation.

4. **Specific Production Use Cases**:
   - **XLA's polyhedral scheduler**: Generates affine loops for GPU kernel fusion
   - **IREE's embedded targets**: Uses affine tiling for fixed-size tensor operations on microcontrollers
   - **Custom accelerators**: When targeting specialized hardware with predictable iteration spaces

**The Typical Pipeline** (when affine is used):
```
High-level Dialect → Linalg → Affine (polyhedral optimization) → SCF → CF → LLVM
```

Affine is an **intermediate optimization IR**, not a starting point. Even compilers using affine start from higher-level representations (Linalg, Tensor) and lower to affine for specific optimization passes.

**Our Pipeline** (this book):
```
NN Dialect → Linalg → SCF → CF → LLVM
```

We skip the affine layer entirely. Linalg's fusion patterns and SCF's loop optimizations (LICM, vectorization) provide sufficient optimization for pedagogical examples and most ML workloads.

## 10.3 Linalg Fusion: Reducing Memory Traffic

Fusion merges adjacent operations to reduce memory operations—the most impactful optimization for memory-bound workloads (which ML often is). MLIR's Linalg dialect enables sophisticated fusion through its structured operation abstraction. Let's understand how fusion works, why it requires careful analysis, and what MLIR automates.

**The Fusion Opportunity**. Consider this IR fragment from Chapter 9's pipeline:

```mlir
%0 = linalg.matmul ins(%A, %B : memref<128x256xf32>, memref<256x128xf32>)
                    outs(%C : memref<128x128xf32>)
%1 = linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                   affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]
} ins(%C : memref<128x128xf32>)
  outs(%D : memref<128x128xf32>) {
^bb0(%arg0: f32, %arg1: f32):
  %max = arith.maximumf %arg0, %cst : f32  // ReLU
  linalg.yield %max : f32
}
```

The matmul writes output `%C`, the generic reads `%C`. There's a **producer-consumer relationship**: matmul produces, ReLU consumes. Fusion recognizes this pattern and merges the operations:

```mlir
%fused = linalg.generic {
  indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,    // A
                   affine_map<(d0, d1, d2) -> (d2, d1)>,    // B
                   affine_map<(d0, d1, d2) -> (d0, d1)>],   // output
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B : memref<128x256xf32>, memref<256x128xf32>)
  outs(%D : memref<128x128xf32>) {
^bb0(%a: f32, %b: f32, %acc: f32):
  %prod = arith.mulf %a, %b : f32
  %sum = arith.addf %acc, %prod : f32
  %relu = arith.maximumf %sum, %cst : f32  // Fused!
  linalg.yield %relu : f32
}
```

The fused operation combines matmul's computation (multiply-accumulate) with ReLU's max operation in a single loop nest. The intermediate buffer `%C` becomes dead code—values flow directly from matmul to ReLU within registers. This eliminates:

- 16384 writes (matmul results to `%C`)
- 16384 reads (ReLU inputs from `%C`)

The unused buffer allocation is subsequently removed by MLIR's dead-buffer-elimination during bufferization cleanup—fusion makes the buffer unused, separate passes remove it. On modern CPUs, memory access costs ~100× more than arithmetic. Eliminating 32768 memory operations is huge.

**Why Fusion Is Hard**. Naive fusion breaks correctness. Consider:

```mlir
%0 = linalg.fill ins(%cst) outs(%buf)      // Initialize buffer
%1 = linalg.matmul ins(%A, %B) outs(%buf)  // Accumulate into buffer
%2 = linalg.generic ins(%buf) outs(%out)   // Use buffer
```

Can we fuse matmul and generic? **Not safely!** Matmul accumulates—it reads existing buffer values. Fusing would read uninitialized data if fill doesn't execute first. The fusion pass must check three dependency types:

- **Read-after-write (RAW)**: Consumer reads what producer writes → safe to fuse (producer's output becomes consumer's input)
- **Write-after-read (WAR)**: Producer writes to a location the consumer already read → unsafe if they execute simultaneously
- **Write-after-write (WAW)**: Both write to the same buffer location → unsafe unless they write to different indices

**Why Linalg Makes Dependency Analysis Practical**. MLIR's Linalg fusion analyzes these dependencies automatically because Linalg operations declare their access patterns explicitly through indexing maps. Consider `linalg.matmul`: the operation tells us "I read A[i,k], read B[k,j], accumulate into C[i,j]" via affine expressions. The compiler can algebraically compare these access patterns to determine if operations conflict.

Contrast with arbitrary loop code (like hand-written C loops with pointer arithmetic): the compiler must solve "does `ptr1 + f(i,j)` alias `ptr2 + g(k)`?" for arbitrary functions f and g. This pointer aliasing problem is undecidable in general. Linalg's structured format restricts access patterns to affine expressions (linear combinations of loop indices), making the problem solvable through well-known algorithms from polyhedral compilation. This is why Linalg fusion works reliably where general loop fusion often fails or requires programmer annotations.

**Generalization: The Prerequisite for Fusion**. Chapter 10's pipeline includes an unexpected pass before fusion:

```cpp
pm.addPass(createLinalgGeneralizeNamedOpsPass());
```

Why generalize? Linalg has **named operations** (`linalg.matmul`, `linalg.conv_2d`) with hardcoded semantics and **generic operations** (`linalg.generic`) with programmable bodies. Fusion patterns match generic operations because they need to inspect and merge loop bodies. Named operations hide their bodies (they're implicit), preventing fusion analysis.

Generalization converts:

```mlir
%0 = linalg.matmul ins(%A, %B) outs(%C)
```

Into:

```mlir
%0 = linalg.generic {
  indexing_maps = [...],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B) outs(%C) {
^bb0(%a: f32, %b: f32, %acc: f32):
  %prod = arith.mulf %a, %b : f32
  %sum = arith.addf %acc, %prod : f32
  linalg.yield %sum : f32
}
```

Now the fusion pass sees the loop body explicitly and can analyze dependencies between operations. This generalization → fusion → lower-to-loops pattern is standard in MLIR Linalg pipelines.

**The Fusion Algorithm (Simplified)**. MLIR's elementwise fusion pass:

1. **Find producer-consumer pairs**: Walk IR finding operations where one writes to a buffer another reads
2. **Check fusion legality**: Analyze indexing maps and iterator types to verify no dependency violations
3. **Merge loop nests**: Combine iteration spaces, fuse loop bodies, eliminate intermediate buffer
4. **Update uses**: Redirect buffer uses to the fused operation's result

While the core idea is straightforward, MLIR's implementation is sophisticated enough to handle **tiled fusion**—where operations are fused while simultaneously being broken into cache-friendly chunks. It also handles multi-consumer fusion (one producer, multiple consumers) and structured control flow. What makes MLIR's version powerful is its generality: the same fusion logic works on matmul+ReLU, conv+batchnorm, any producer-consumer pair with compatible iteration spaces.

**When Fusion Helps (and When It Doesn't)**. Fusion benefits memory-bound workloads where memory traffic dominates arithmetic cost. For compute-bound operations (large matmuls where memory traffic is small compared to O(n³) arithmetic), fusion matters less. Fusion can even hurt performance by:

- **Increasing register pressure**: Fused operations need more live values, potentially spilling to memory
- **Reducing parallelism**: Separate operations can execute concurrently (different cores, different functional units); fusion serializes them
- **Hurting cache behavior**: Very large fused operations might not fit in cache where separate operations would

Production compilers use cost models to decide when to fuse. Chapter 10's simple pipeline always fuses elementwise operations with their consumers—a good heuristic for typical ML workloads but not universally optimal. Chapter 14 applies similar techniques at larger scale using Linalg operations and the Transform dialect for declarative optimization specification.

## 10.4 Loop Invariant Code Motion: Hoisting Computations

After fusion reduces memory traffic, the next optimization targets redundant computation: **loop-invariant code motion (LICM)** hoists calculations out of loops when their inputs don't change across iterations. This classic compiler optimization works on MLIR's SCF dialect (structured control flow), complementing Linalg fusion which works at a higher abstraction level.

**The LICM Opportunity**. Consider code after Linalg lowering to SCF loops:

```mlir
scf.for %i = %c0 to %c128 step %c1 {
  scf.for %j = %c0 to %c128 step %c1 {
    %scale = arith.divf %c1, %size : f32  // size is loop-invariant
    %val = memref.load %input[%i, %j] : memref<128x128xf32>
    %normalized = arith.mulf %val, %scale : f32
    memref.store %normalized, %output[%i, %j] : memref<128x128xf32>
  }
}
```

The `scale = 1.0 / size` computation occurs inside nested loops, executing 128×128 = 16384 times. But `size` doesn't depend on loop iterators `%i` or `%j`—the computation is **loop-invariant**. LICM hoists it outside:

```mlir
%scale = arith.divf %c1, %size : f32  // Hoisted!
scf.for %i = %c0 to %c128 step %c1 {
  scf.for %j = %c0 to %c128 step %c1 {
    %val = memref.load %input[%i, %j] : memref<128x128xf32>
    %normalized = arith.mulf %val, %scale : f32
    memref.store %normalized, %output[%i, %j] : memref<128x128xf32>
  }
}
```

Now the division executes once instead of 16384 times. For expensive operations (division, transcendentals like `exp`, `log`), hoisting eliminates redundant computation.

**LICM's Safety Requirements**. Like fusion, naive hoisting breaks correctness. Consider:

```mlir
scf.for %i = %c0 to %c128 step %c1 {
  %cond = arith.cmpi sgt, %i, %threshold : i32
  scf.if %cond {
    %val = arith.divf %numerator, %denominator : f32  // Might divide by zero
    memref.store %val, %output[%i] : memref<128xf32>
  }
}
```

Can we hoist the division outside the loop? **Not safely!** The division only executes conditionally; hoisting makes it unconditional. If `%denominator` is zero and `%threshold` is large, the original code might never divide by zero (condition never true), but hoisted code always divides by zero. LICM must respect control flow.

MLIR's LICM pass checks safety conditions before hoisting:

1. **Definedness**: Operation's operands dominate the target hoist location (values available before the loop)
2. **Control independence**: Operation not guarded by conditional control flow inside the loop
3. **Side-effect freedom**: Operation doesn't read/write memory or have other observable effects (pure computation)
4. **Exception safety**: Operation can't fault (no division by zero, no out-of-bounds access, etc.)

For arithmetic operations on SSA values, these checks are straightforward. For memory operations, LICM is conservative: it doesn't hoist loads/stores because **alias analysis** is complex. If the compiler can't prove that a `memref.store` inside the loop doesn't affect a `memref.load` you want to hoist, the load must stay inside the loop. Wrong hoisting could reorder memory effects incorrectly, breaking correctness.

**How LICM Works**. The algorithm:

1. **Identify loops**: Walk IR finding `scf.for` and `scf.while` operations
2. **Analyze loop bodies**: For each operation inside the loop, check if all operands are:
   - Loop-invariant (defined outside the loop), or
   - Constants (defined nowhere, available everywhere)
3. **Check legality**: Verify safety conditions (definedness, control independence, side-effect freedom)
4. **Hoist operations**: Move legal loop-invariant operations to the loop's parent region
5. **Iterate**: Repeat until no more hoisting is possible (some operations become hoistable after others are hoisted)

The iteration is crucial. Consider:

```mlir
scf.for %i = ... {
  %a = arith.addf %x, %y : f32  // %x, %y defined outside loop
  %b = arith.mulf %a, %z : f32  // %z defined outside loop
}
```

First iteration hoists `%a` (operands %x, %y are loop-invariant). After hoisting `%a`, its result becomes loop-invariant too, enabling `%b` to be hoisted in the next iteration. Multi-pass execution finds all hoisting opportunities.

**LICM in Chapter 10's Pipeline**. Chapter 10 applies LICM after lowering to SCF:

```cpp
pm.addPass(createConvertLinalgToLoopsPass());  // Lower to SCF
pm.addPass(createLoopInvariantCodeMotionPass());  // Hoist invariants
pm.addPass(createCanonicalizerPass());  // Clean up
```

Why this order? Linalg operations don't have visible loop bodies—LICM works on explicit loops. We must lower to SCF first, exposing loop structure and invariant computations. Then LICM optimizes the explicit loops, and canonicalizer cleans up any redundancy created.

This pattern—lower abstraction, optimize, clean up—appears throughout MLIR pipelines. High-level dialects (Linalg) provide optimization opportunities (fusion), mid-level dialects (SCF) provide different opportunities (LICM), low-level dialects (LLVM) provide yet others (instruction scheduling). Each level has its place.

**LICM's Interaction with Fusion**. Fusion and LICM are complementary but not independent:

- **Fusion creates LICM opportunities**: Fused operations might have loop-invariant subexpressions not visible before fusion
- **LICM prepares for vectorization**: Hoisting reduces loop body size, making vectorization more effective
- **Both reduce memory pressure**: LICM by eliminating recomputation, fusion by eliminating intermediate buffers

Chapter 10's pipeline exploits these interactions by ordering passes to maximize benefit: fusion first (reduces memory operations), lower to loops (exposes loop structure), LICM (reduces computation), vectorization (accelerates remaining computation). Changing the order would miss optimization opportunities—a lesson learned through experimentation and performance measurement.

## 10.5 Vectorization: Exploiting SIMD Hardware

Modern CPUs provide **Single Instruction, Multiple Data (SIMD)** capabilities: instructions that operate on vectors of data (4, 8, 16 elements simultaneously) instead of scalars. AVX2 processes 8 floats per instruction; AVX-512 processes 16. Utilizing SIMD can significantly accelerate data-parallel computations—which ML workloads overwhelmingly are. MLIR's `vector` dialect provides portable SIMD abstraction, generating architecture-specific instructions automatically.

**The Vectorization Opportunity**. After LICM optimization, our normalization example contains:

```mlir
%scale = arith.divf %c1, %size : f32
scf.for %i = %c0 to %c128 step %c1 {
  scf.for %j = %c0 to %c128 step %c1 {
    %val = memref.load %input[%i, %j] : memref<128x128xf32>
    %normalized = arith.mulf %val, %scale : f32
    memref.store %normalized, %output[%i, %j] : memref<128x128xf32>
  }
}
```

The inner loop processes one element per iteration—**scalar execution**. With AVX2, we can process 8 elements simultaneously:

```mlir
%scale_vec = vector.broadcast %scale : f32 to vector<8xf32>
scf.for %i = %c0 to %c128 step %c1 {
  scf.for %j = %c0 to %c128 step %c8 {  // Step by 8 now
    %vals = vector.load %input[%i, %j] : memref<128x128xf32>, vector<8xf32>
    %normalized_vec = arith.mulf %vals, %scale_vec : vector<8xf32>
    vector.store %normalized_vec, %output[%i, %j] : memref<128x128xf32>, vector<8xf32>
  }
}
```

The loop now steps by 8, loads 8 elements at once into a vector register, multiplies the entire vector by the broadcasted scale, and stores 8 results. This reduces iterations from 128 to 16, with each iteration processing 8 elements. The SIMD instructions enable substantial performance improvements by exploiting data parallelism, though actual gains depend on memory bandwidth and other system factors.

**The Vector Dialect**. MLIR's `vector` dialect provides hardware-independent SIMD operations:

```mlir
// Vector types
vector<4xf32>      // 4 floats (SSE)
vector<8xf32>      // 8 floats (AVX2)
vector<16xf32>     // 16 floats (AVX-512)
vector<4x4xf32>    // 2D vectors (for matrix operations)

// Vector operations
%v = vector.load %mem[%idx] : memref<?xf32>, vector<8xf32>
vector.store %v, %mem[%idx] : memref<?xf32>, vector<8xf32>
%sum = vector.reduction <add>, %v : vector<8xf32>  // Result: f32
%bcast = vector.broadcast %scalar : vector<8xf32>   // %scalar: f32, result: vector<8xf32>
%shuffled = vector.shuffle %v1, %v2 [0, 2, 4, 6, 1, 3, 5, 7] : vector<8xf32>, vector<8xf32>
```

The dialect abstracts hardware details: `vector<8xf32>` compiles to `vmovups` (AVX load), `vmulps` (AVX multiply), `vmovups` (AVX store) on x86, but different instructions on ARM or other architectures. This portability is crucial—write vectorization once, run on any SIMD hardware.

**When Vectorization Works**. Not all loops vectorize. Requirements:

1. **Data parallelism**: Iterations must be independent (no loop-carried dependencies)
2. **Contiguous memory**: Vector loads/stores require adjacent elements (stride-1 access)
3. **Aligned access**: Many SIMD instructions require memory alignment (16-byte, 32-byte)
4. **Uniform control flow**: Branches inside loops complicate vectorization (predication helps but adds overhead)

Consider an invalid example:

```mlir
scf.for %i = %c0 to %c128 step %c1 {
  %prev = memref.load %data[%i] : memref<128xf32>
  %next = memref.load %data[%i + 1] : memref<128xf32>
  %avg = arith.addf %prev, %next : f32
  memref.store %avg, %data[%i] : memref<128xf32>
}
```

This loop has **loop-carried dependencies**: iteration `i` writes `data[i]`, iteration `i+1` reads `data[i]`. Vectorizing would process multiple iterations simultaneously, reading data before it's written. MLIR's vectorization pass detects this through dependency analysis and refuses to vectorize.

**Explicit vs. Auto-Vectorization**. MLIR supports two vectorization approaches:

1. **Auto-vectorization**: Analyze scalar loops, automatically convert to vector operations (LLVM's loop vectorizer does this at the LLVM IR level)
2. **Explicit vectorization**: Use high-level operations (Linalg) that express parallelism, lower to vector dialect before lowering to LLVM

Chapter 10 uses **auto-vectorization** via LLVM's backend:

```cpp
// Lower to loops (exposes loop structure)
pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());

// Optimize loops
pm.addPass(createLoopInvariantCodeMotionPass());

// Lower to LLVM (LLVM's auto-vectorizer kicks in with -O3)
pm.addPass(createSCFToControlFlowPass());
// ... rest of LLVM lowering
```

Why auto-vectorization? **Simplicity and maturity**. LLVM's loop vectorizer has decades of development and handles edge cases (remainder loops, alignment, cost models) robustly. While MLIR's vector dialect provides more control for advanced use cases (explicit tiling strategies, target-specific SIMD patterns), LLVM's auto-vectorizer suffices for typical ML workloads and requires no additional passes in the pipeline.

**LLVM 21 VPlan Enhancement**. This book uses LLVM 21, which includes major improvements to the loop vectorizer through **VPlan (Vectorization Plan)**—a hierarchical control flow graph that models multiple vectorization strategies before code generation. Key improvements:

- **Early Exit Vectorization**: Loops with `break` or `return` statements (e.g., `std::find`) can now be vectorized—previously these were rejected entirely
- **Better Register Pressure Modeling**: Per-plan cost estimation prevents over-vectorization that causes register spilling (5-10% performance improvement for data-heavy loops)
- **Uncountable Loop Support**: Loops where trip count isn't known at compile time benefit from VPlan's adaptive strategy selection

For MLIR users, this means the auto-vectorizer produces better code "for free"—your loops lower to LLVM IR, and LLVM 21's VPlan automatically applies more aggressive optimizations than LLVM 20 could. The loop invariant code motion and fusion optimizations in this chapter prepare IR that VPlan can exploit effectively.

**Vectorization of Reduction Loops**. Parallelizable loops (embarrassingly parallel computations like element-wise operations) vectorize straightforwardly. Reduction loops (sum, max, matrix multiplication) require more sophistication. Consider:

```mlir
%sum = scf.for %i = %c0 to %c1024 step %c1 iter_args(%acc = %c0) -> (f32) {
  %val = memref.load %data[%i] : memref<1024xf32>
  %new_acc = arith.addf %acc, %val : f32
  scf.yield %new_acc : f32
}
```

Naive vectorization fails: each iteration depends on the previous iteration's accumulator. Solution: **vector accumulation** followed by **horizontal reduction**:

```mlir
%init_vec = vector.splat %c0 : vector<8xf32>
%partial_sums = scf.for %i = %c0 to %c1024 step %c8 iter_args(%acc_vec = %init_vec) -> (vector<8xf32>) {
  %vals = vector.load %data[%i] : memref<1024xf32>, vector<8xf32>
  %new_acc_vec = arith.addf %acc_vec, %vals : vector<8xf32>
  scf.yield %new_acc_vec : vector<8xf32>
}
%sum = vector.reduction <add>, %partial_sums : vector<8xf32> to f32
```

The `iter_args(%acc_vec = %init_vec)` syntax specifies **loop-carried variables**: values that persist across iterations. The `%acc_vec` parameter receives `%init_vec` initially, then receives the yielded value from each iteration. The `-> (vector<8xf32>)` declares the loop's result type, which matches the yield type. This pattern is MLIR's equivalent of functional programming's fold/reduce with explicit accumulator threading.

The loop now accumulates into a vector of 8 partial sums (each element accumulates every 8th input element), then performs a final horizontal reduction (`vector.reduction`) to combine them into a scalar. This parallelizes the reduction while maintaining numerical correctness.

**Handling Remainder Loops**. Vector lengths rarely divide evenly into data sizes. For 128 elements with vector length 8, the main vectorized loop processes 120 elements (15 iterations), leaving 8 elements. MLIR's `vector-to-scf` conversion generates **remainder loops**:

```mlir
// Vectorized main loop
scf.for %i = %c0 to %c120 step %c8 {
  // Vector operations
}

// Scalar remainder loop
scf.for %i = %c120 to %c128 step %c1 {
  // Scalar operations
}
```

The pass calculates trip counts (`120 = (128 / 8) * 8`), generates a vectorized loop for the bulk, and a scalar loop for leftovers. This ensures correctness for any data size. More sophisticated approaches use **vector masking** (process full vector width with masks disabling unused lanes), but remainder loops are simpler and often sufficient.

## 10.6 The Complete Optimization Pipeline

Previous sections examined individual optimizations in isolation. Production pipelines compose them carefully, exploiting interactions and respecting dependencies. Let's analyze Chapter 10's complete pipeline, understanding why each pass appears where it does and what invariants each stage maintains.

**The Pipeline Code**. From `ch.10.Optimizations/src/bindings.cpp`:

```cpp
void NNCompiler::lowerToLLVM(mlir::ModuleOp module) {
  mlir::PassManager pm(&context);
  
  // Stage 1: Lower NN dialect to standard dialects (tensor-based)
  pm.addNestedPass<mlir::func::FuncOp>(nn::createConvertNNToStandardPass());
  pm.addPass(mlir::createCanonicalizerPass());
  
  // Stage 2: Linalg optimizations (on tensors)
  pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgElementwiseOpFusionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  
  // Stage 3: Bufferization (tensor → memref conversion)
  DialectRegistry registry;
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
  
  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(options));
  pm.addPass(bufferization::createBufferResultsToOutParamsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  
  // Stage 4: Lower to explicit loops (memrefs → SCF)
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  
  // Stage 5: Loop optimizations
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  
  // Stage 6: Handle vector operations (if any exist)
  pm.addPass(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createCanonicalizerPass());
  
  // Stage 7: Lower to LLVM
  pm.addPass(createConvertVectorToLLVMPass());  // Vector → LLVM first
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertVectorToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to LLVM failed\n";
  }
}
```

Seven stages, each with specific goals and invariants. Let's walk through them.

**Stage 1: High-Level Lowering** (NN → Standard Dialects). Chapter 9's NN dialect operations (`nn.matmul`, `nn.linear`, `nn.relu`) lower to standard MLIR dialects (Linalg, Arith, Tensor):

```mlir
// Before: NN dialect
%result = nn.matmul(%A, %B) : (tensor<128x256xf32>, tensor<256x128xf32>) -> tensor<128x128xf32>

// After: Linalg dialect (tensor-based)
%empty = tensor.empty() : tensor<128x128xf32>
%result = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x128xf32>)
                         outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
```

This stage translates domain-specific operations (NN) to mathematical operations (Linalg structured ops). **Crucially, operations remain tensor-based**—they return SSA values representing tensors, not write to pre-allocated buffers. The NN dialect disappears, and all subsequent passes work on standard dialects. Canonicalization afterward cleans up redundant tensor allocations and simplifies arithmetic.

**Stage 2: Linalg Optimizations** (Fusion). With operations in Linalg, we apply structured-operation-level optimizations:

1. **Generalize named ops**: Convert `linalg.matmul` → `linalg.generic` to expose loop bodies
2. **Canonicalize**: Simplify the verbose generic ops
3. **Fuse elementwise operations**: Merge producer-consumer pairs to eliminate intermediate tensors
4. **Canonicalize again**: Clean up fused operations

After this stage, operations are still **tensor-based Linalg** but with fewer operations and less memory traffic. The IR represents optimized computation graphs at the mathematical level, not yet converted to explicit memory operations or loops.

**Stage 3: Bufferization and Loop Lowering** (Tensor → MemRef → Loops). This critical stage performs two transformations:

**3a. Bufferization** (Tensor → MemRef):
```mlir
// Before: Tensor-based (SSA values)
%result = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x128xf32>)
                         outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>

// After: MemRef-based (in-place updates)
%C = memref.alloc() : memref<128x128xf32>
linalg.matmul ins(%A, %B : memref<128x256xf32>, memref<256x128xf32>)
              outs(%C : memref<128x128xf32>)
```

One-Shot Bufferize converts tensor-based operations (functional, immutable) to memref-based operations (imperative, in-place). Tensors become allocated buffers, operations that returned tensors now write to output buffers. This transformation is **semantics-preserving**—tensor semantics guarantee no aliasing, so bufferization can safely allocate buffers and reuse them.

**3b. Lower to Explicit Loops** (Linalg → SCF):
```mlir
// After bufferization + loop lowering: SCF loops on memrefs
scf.for %i = %c0 to %c128 step %c1 {
  scf.for %j = %c0 to %c128 step %c1 {
    scf.for %k = %c0 to %c256 step %c1 {
      %a = memref.load %A[%i, %k] : memref<128x256xf32>
      %b = memref.load %B[%k, %j] : memref<256x128xf32>
      %acc = memref.load %C[%i, %j] : memref<128x128xf32>
      %prod = arith.mulf %a, %b : f32
      %sum = arith.addf %acc, %prod : f32
      memref.store %sum, %C[%i, %j] : memref<128x128xf32>
    }
  }
}
```

Linalg's implicit iteration spaces become explicit `scf.for` loops. This is a **large abstraction drop**—we've gone from "compute matmul on tensors" to "three nested loops with explicit memory loads, arithmetic, and stores." But this explicitness enables loop-level optimizations (LICM, later auto-vectorization in LLVM).

**Why Bufferize After Fusion?** This ordering is critical:
- **Tensor-based fusion** works on mathematical operations without worrying about memory layout or aliasing
- Bufferization happens **after** fusion has eliminated intermediate tensors
- The resulting memref code has fewer allocations than pre-fusion code would have

If we bufferized before fusion, we'd need complex alias analysis to prove fusion safety. Tensor semantics make fusion straightforward.

**Stage 5: Loop Optimizations** (LICM). With explicit loops, apply loop-invariant code motion:

```mlir
// Before LICM
scf.for %i = ... {
  %scale = arith.divf %c1, %size : f32  // Loop-invariant
  scf.for %j = ... {
    %val = memref.load %input[%i, %j]
    %normalized = arith.mulf %val, %scale : f32
    memref.store %normalized, %output[%i, %j]
  }
}

// After LICM
%scale = arith.divf %c1, %size : f32  // Hoisted
scf.for %i = ... {
  scf.for %j = ... {
    %val = memref.load %input[%i, %j]
    %normalized = arith.mulf %val, %scale : f32
    memref.store %normalized, %output[%i, %j]
  }
}
```

This stage reduces redundant computation, preparing loops for vectorization (smaller loop bodies vectorize more efficiently).

**Stage 6: Vectorization** (Defer to LLVM). Chapter 10's pipeline prepares loops for LLVM's auto-vectorizer but doesn't explicitly generate vector dialect operations:

```mlir
// After Stage 5: Clean scalar loops ready for LLVM
scf.for %i = %c0 to %c128 step %c1 {
  %val = memref.load %input[%i] : memref<128xf32>
  %squared = arith.mulf %val, %val : f32
  memref.store %squared, %output[%i] : memref<128xf32>
}

// Stage 6: Convert any existing vector ops to SCF (cleanup)
pm.addPass(mlir::createConvertVectorToSCFPass());
pm.addPass(mlir::createCanonicalizerPass());

// After Stage 7: LLVM's auto-vectorizer (with -O3) converts to SIMD
// This happens in the ExecutionEngine's LLVM optimization pipeline
```

The `createConvertVectorToSCFPass()` handles any vector operations that might exist (from earlier stages or manual insertion), but the main vectorization happens through LLVM's auto-vectorizer when the ExecutionEngine compiles with optimization level 3 (`makeOptimizingTransformer(3, 0, nullptr)`). LLVM analyzes the scalar loops and automatically generates AVX2/AVX-512 instructions where profitable.

**Stage 7: Lower to LLVM** (Standard Dialects → LLVM). The final stage converts all high-level dialects to LLVM IR:

- **Vector → LLVM**: Convert any vector operations to LLVM vector intrinsics (first, before other conversions)
- **SCF → CF**: Convert structured control flow (`scf.for`, `scf.if`) to unstructured control flow graph (basic blocks, branches)
- **MemRef → LLVM**: Lower memory references to LLVM pointer operations
- **Func → LLVM**: Convert function signatures and calls to LLVM conventions
- **Arith → LLVM**: Map arithmetic operations to LLVM instructions
- **Reconcile casts**: Clean up any remaining type conversions

After this stage, the ModuleOp contains only LLVM dialect operations—ready for execution through MLIR's ExecutionEngine, which applies LLVM's optimization pipeline (including auto-vectorization with `-O3`).

**Why This Order?** The pipeline's ordering is deliberate:

1. **High-level optimizations first**: Fusion works better on structured ops (Linalg on tensors) than loops; lower too early and you miss fusion opportunities
2. **Tensor-based optimization**: Tensors provide functional semantics (no aliasing) that make fusion analysis simple and safe
3. **Bufferize after fusion**: Convert tensors to memrefs only after eliminating intermediate tensors through fusion
4. **Lower gradually**: Multi-stage lowering (NN → Linalg → MemRef → SCF → CF → LLVM) exposes different optimization opportunities at each level
5. **Canonicalize frequently**: Each major transformation creates cleanup opportunities; running canonicalizer prevents IR bloat
6. **Loop optimizations on explicit loops**: LICM requires visible loop structure (SCF dialect)
7. **Vectorization via LLVM**: Rather than explicit vector dialect IR, rely on LLVM's mature auto-vectorizer (enabled by `-O3` in ExecutionEngine)

Swapping stages breaks optimizations. Examples:
- **Bufferize before fusion**: Would require complex alias analysis to prove fusion safety; tensor semantics make this trivial
- **Lower to loops before fusion**: Fusion patterns match Linalg ops, not SCF loops
- **Run LICM before loop lowering**: LICM needs explicit loop structure (won't find Linalg operations)
- **Vectorize before LICM**: Might miss hoisting opportunities that would enable better vectorization

The pipeline reflects deep understanding of optimization interactions and MLIR's dialect abstractions. The tensor-first approach (Chapters 5-10) is the modern best practice.

**Pipeline Invariants and Debugging**. Each stage maintains invariants:

- After Stage 1: All NN operations converted to standard dialects (tensor-based)
- After Stage 2: Producer-consumer pairs fused where legal (still tensor-based)
- After Stage 3: All tensors converted to memrefs, function signatures use output parameters
- After Stage 4: All Linalg operations lowered to explicit loops (on memrefs)
- After Stage 5: Loop-invariant computations hoisted
- After Stage 6: Any vector operations converted to SCF
- After Stage 7: Only LLVM dialect remains (auto-vectorization happens during LLVM optimization in ExecutionEngine)

When debugging, verify invariants at each stage using `-mlir-print-ir-after-all`:

```bash
./test_jit -mlir-print-ir-after-all 2>&1 | less
```

This prints IR after every pass. If Stage 2 should eliminate intermediate buffers but doesn't, inspect IR after `LinalgElementwiseOpFusionPass` to see what fusion missed and why. Systematic debugging through pipeline stages beats guessing.

## 10.7 Topological Traversal: Ordering Computation Graphs

Before we can execute or compile a computation graph, we must solve a fundamental problem: **determining execution order**. Operations have dependencies—an addition operation that consumes the output of a matrix multiplication cannot execute until the multiplication completes. This dependency structure forms a **directed acyclic graph (DAG)**, and executing it correctly requires **topological sorting**—ordering operations so dependencies execute before dependents. This section explains why topological traversal is essential for compilers, how the algorithm works, and how to implement it efficiently.

**The Ordering Problem**. Consider a simple computation graph:

```python
# Python pseudo-code
a = input("a")
b = input("b")
c = matmul(a, b)      # c depends on a, b
d = add(c, a)         # d depends on c, a
e = relu(d)          # e depends on d
result = softmax(e)  # result depends on e
```

This defines a DAG:

```
  a ----→ c ----→ d ----→ e ----→ result
  |       ↑
  b ------┘
```

Multiple execution orders satisfy dependencies:
- **Valid**: `[a, b, c, d, e, result]` or `[b, a, c, d, e, result]`
- **Invalid**: `[c, a, b, ...]` (c before its inputs), `[a, d, b, ...]` (d before c)

Compilers must find a valid order. Without it, operations would try reading uncomputed values, producing garbage results or crashing. This isn't theoretical—Chapter 9's `forward()` function (Section 9.8) assumes `graph.nodes` is already topologically sorted. If the graph arrives in arbitrary order, we must sort it first.

**Why Topological Sorting Matters**. Beyond correctness, topological order affects performance:

1. **Memory Efficiency**: Executing dependencies before dependents allows earlier freeing of intermediate buffers. Out-of-order execution forces keeping more values live simultaneously.

2. **Optimization Opportunities**: Many compiler optimizations (fusion, common subexpression elimination) rely on knowing data flow. Topological order makes data dependencies explicit.

3. **Parallelization**: Operations with no path between them in the DAG can execute concurrently. Topological sorting identifies these independent sets, enabling parallel execution.

4. **JIT Compilation**: When compiling Python computation graphs (PyTorch, TensorFlow, JAX), users build graphs in arbitrary order. The compiler must sort before code generation.

Every MLIR-based compiler performs topological sorting somewhere—either when building IR from high-level operations (Chapter 9, 11) or when scheduling passes (MLIR's PassManager does this internally for pass dependencies).

**Kahn's Algorithm**. The standard topological sorting algorithm:

```
Algorithm: Kahn's Topological Sort
Input: DAG with nodes N and edges E
Output: Topologically sorted node list

1. Compute in-degree for each node (number of incoming edges)
2. Initialize queue Q with all nodes having in-degree 0 (no dependencies)
3. Initialize empty result list L
4. While Q is not empty:
   a. Remove node n from Q
   b. Append n to L
   c. For each node m that n points to (n → m):
      - Decrement m's in-degree
      - If m's in-degree reaches 0, add m to Q
5. If L contains all nodes, return L
   Else, graph has a cycle (error)
```

The algorithm works by repeatedly selecting nodes with no remaining dependencies, adding them to the result, and "removing" them from the graph (decrementing dependents' in-degrees). This continues until all nodes are processed or a cycle is detected.

**C++ Implementation**. Here's a practical implementation for computation graphs:

```cpp
struct GraphNode {
  int id;
  std::string op_type;
  std::vector<int> inputs;   // IDs of input nodes
  std::vector<int64_t> shape;
};

struct ComputationGraph {
  std::vector<GraphNode> nodes;
  std::unordered_set<int> input_ids;  // Graph inputs (no dependencies)
};

// Returns nodes in topological order, or empty vector if cycle detected
std::vector<GraphNode> topologicalSort(const ComputationGraph& graph) {
  // Build adjacency list and compute in-degrees
  std::unordered_map<int, std::vector<int>> adj;  // node_id → dependent_ids
  std::unordered_map<int, int> in_degree;         // node_id → in_degree
  std::unordered_map<int, const GraphNode*> node_map;  // id → node
  
  // Initialize
  for (const auto& node : graph.nodes) {
    node_map[node.id] = &node;
    in_degree[node.id] = 0;
  }
  
  // Build adjacency list and count in-degrees
  for (const auto& node : graph.nodes) {
    for (int input_id : node.inputs) {
      adj[input_id].push_back(node.id);
      in_degree[node.id]++;
    }
  }
  
  // Find all nodes with in-degree 0 (inputs or constants)
  std::queue<int> ready;
  for (const auto& [id, degree] : in_degree) {
    if (degree == 0 || graph.input_ids.count(id)) {
      ready.push(id);
    }
  }
  
  // Kahn's algorithm
  std::vector<GraphNode> sorted;
  while (!ready.empty()) {
    int node_id = ready.front();
    ready.pop();
    
    sorted.push_back(*node_map[node_id]);
    
    // Process dependents
    for (int dependent_id : adj[node_id]) {
      in_degree[dependent_id]--;
      if (in_degree[dependent_id] == 0) {
        ready.push(dependent_id);
      }
    }
  }
  
  // Check for cycles
  if (sorted.size() != graph.nodes.size()) {
    // Cycle detected: some nodes still have in-degree > 0
    return {};  // Empty vector indicates error
  }
  
  return sorted;
}
```

**Using Topological Sort in MLIR**. Chapter 9's `forward()` function can now handle arbitrary graph ordering:

```cpp
py::array_t<float> forward(const Tensor& output_tensor) {
  // ... context setup ...
  
  // Sort graph topologically
  auto sorted_nodes = topologicalSort(graph);
  if (sorted_nodes.empty()) {
    throw std::runtime_error("Computation graph has a cycle!");
  }
  
  // Build MLIR operations in topological order
  std::map<int, Value> valueMap;
  for (const auto& node : sorted_nodes) {
    if (graph.input_ids.count(node.id)) {
      // Input node: map to function argument
      valueMap[node.id] = /* ... function argument ... */;
    } else {
      // Operation node: inputs are guaranteed to be in valueMap
      Value lhs = valueMap[node.inputs[0]];
      Value rhs = valueMap[node.inputs[1]];
      Value output = /* ... create operation ... */;
      valueMap[node.id] = output;
    }
  }
  
  // ... compilation and execution ...
}
```

The key insight: after topological sorting, `valueMap[node.inputs[i]]` is guaranteed to exist because input nodes were processed earlier in the sorted order.

**Complexity Analysis**. Kahn's algorithm is efficient:

- **Time**: O(V + E) where V = number of nodes, E = number of edges. Each node and edge is visited exactly once.
- **Space**: O(V) for in-degree map and result list.

For typical computation graphs (hundreds to thousands of operations), this is negligible compared to compilation and execution time.

**Alternative: Depth-First Search**. Another topological sorting approach uses post-order DFS:

```cpp
void dfsVisit(int node_id, 
              const std::unordered_map<int, std::vector<int>>& adj,
              std::unordered_set<int>& visited,
              std::unordered_set<int>& in_stack,
              std::vector<int>& post_order) {
  if (in_stack.count(node_id)) {
    throw std::runtime_error("Cycle detected");
  }
  if (visited.count(node_id)) return;
  
  visited.insert(node_id);
  in_stack.insert(node_id);
  
  for (int dependent_id : adj[node_id]) {
    dfsVisit(dependent_id, adj, visited, in_stack, post_order);
  }
  
  in_stack.erase(node_id);
  post_order.push_back(node_id);  // Add after all dependents
}

std::vector<int> topologicalSortDFS(const ComputationGraph& graph) {
  std::unordered_map<int, std::vector<int>> adj;
  for (const auto& node : graph.nodes) {
    for (int input_id : node.inputs) {
      adj[input_id].push_back(node.id);
    }
  }
  
  std::unordered_set<int> visited, in_stack;
  std::vector<int> post_order;
  
  for (const auto& node : graph.nodes) {
    if (!visited.count(node.id)) {
      dfsVisit(node.id, adj, visited, in_stack, post_order);
    }
  }
  
  // Post-order traversal is reverse topological order
  std::reverse(post_order.begin(), post_order.end());
  return post_order;
}
```

DFS-based sorting is slightly harder to understand but has the same O(V + E) complexity. Kahn's algorithm is more intuitive ("repeatedly process nodes with no dependencies") and naturally detects cycles, making it preferable for beginners.

**Practical Considerations**. In real compilers:

1. **Caching**: If the graph structure doesn't change, cache the sorted order and reuse across compilations.

2. **Incremental Updates**: When adding new operations to a graph, you can update topological order incrementally rather than re-sorting from scratch.

3. **Multiple Valid Orders**: Kahn's algorithm finds one topological order, but many may exist. Some orders may be better for performance (e.g., grouping operations that can fuse). Advanced compilers use heuristics to select good orders.

4. **MLIR's Ordering**: MLIR operations within a block are inherently in topological order (SSA form ensures definitions precede uses). The PassManager uses topological sorting for pass dependencies, not operation scheduling.

Topological sorting is a fundamental algorithm every compiler engineer must understand. It bridges the gap between user-written computation graphs (arbitrary order) and executable IR (dependencies-respecting order), enabling everything else—optimization, parallelization, execution—to work correctly.

## 10.8 Summary

This chapter explored MLIR's optimization infrastructure through three core techniques—Linalg fusion (reduces memory traffic), loop-invariant code motion (reduces redundant computation), and vectorization (exploits SIMD hardware). We also examined topological sorting, the fundamental algorithm enabling correct computation graph execution.

Key insights:

- **Multi-level optimization**: MLIR's dialect hierarchy enables optimizations at different abstraction levels (structured ops, loops, vectors), each providing unique opportunities
- **Compositional passes**: Small, focused passes compose into sophisticated pipelines; the framework handles complexity through verification and canonicalization
- **Correct execution order**: Topological sorting transforms arbitrary computation graphs into executable sequences, respecting data dependencies
- **Measurement-driven**: Optimization effectiveness must be measured through profiling and benchmarking, not assumed

Chapter 10 established that optimizations preserve user-facing APIs—the Python interface stayed identical between Chapters 9 and 10, demonstrating how backend improvements remain transparent to users.