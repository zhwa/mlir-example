# Chapter 14: Production-Grade Optimizations

Chapter 13 built a complete GPT architecture—token embeddings, RoPE, attention, feedforward networks, and autoregressive generation. The implementation is correct but naive: full forward passes every generation step, no cache locality optimization, scalar operations instead of SIMD vectors, separate operations that could fuse. Production systems require orders-of-magnitude speedup for real-time serving. This chapter transforms Chapter 13's educational GPT into production-grade code using modern compiler techniques and advanced dialect features deferred from Chapter 9.

The optimization journey has four pillars: **high-level IR** (Linalg operations enable pattern recognition), **declarative transformations** (Transform dialect and DRR provide composable optimization), **advanced dialect features** (interfaces, canonicalization patterns), and **algorithmic improvements** (KV caching eliminates redundant computation). Modern MLIR optimization goes beyond legacy passes to embrace declarative, transparent, and composable transformations—the approach used in production compilers at Google (IREE), Meta (Torch-MLIR), and NVIDIA.

**A Note on Performance Numbers**. Throughout this chapter, we discuss theoretical speedups (3-5× for compiler optimizations, 10-100× for KV caching). These numbers represent what's achievable for **production-scale models** (GPT-2 with d_model=768, GPT-3 with d_model=12288). Our nano GPT implementation (d_model=64, 2 layers, vocab=256) is too small to demonstrate these gains—compiler optimizations become effective when operation sizes exceed cache capacities and memory bandwidth becomes the bottleneck. The techniques are correct and production-ready; the scale is educational. Think of this as learning race car driving with a go-kart: the principles are identical, but you won't hit 200 mph.

Chapter 14 completes the book's optimization arc and fulfills promises from Chapter 9. Chapters 1-9 built foundations (MLIR fundamentals, dialects, operations, TableGen). Chapters 10-13 constructed transformers (attention, blocks, GPT architecture). Chapter 14 optimizes for production serving and introduces advanced dialect features—the final step before deployment.

## 14.1 The Performance Problem

Chapter 13's GPT implementation prioritizes clarity over speed. Matrix multiplications use nested loops with scalar operations. Generation recomputes attention for all tokens every iteration. Element-wise operations (GELU, layer norm) don't exploit data parallelism. These choices simplify understanding but yield poor performance for production-scale models.

**Performance Characteristics**. Our nano GPT (d_model=64, 2 layers, seq_len=32, vocab=256) is memory-light and compute-light—optimizations have minimal impact because everything fits in L1 cache and computation completes before memory bandwidth saturates. For production models, the story differs dramatically:

```
Nano GPT (d_model=64, 2 layers):
  - Model size: ~50 KB (fits L1 cache)
  - Forward pass: dominated by function call overhead
  - Optimizations: minimal gain (data already cache-resident)

GPT-2 Small (d_model=768, 12 layers):
  - Model size: ~500 MB (exceeds L3 cache)
  - Forward pass: memory bandwidth bottleneck
  - Optimizations: 3-5× speedup (tiling, fusion, vectorization critical)

GPT-3 (d_model=12288, 96 layers):
  - Model size: ~350 GB (exceeds DRAM for single GPU)
  - Forward pass: severe memory bottleneck
  - Optimizations: 10-20× speedup (aggressive fusion, tensor parallelism)
```

The optimization techniques in this chapter are production-tested but require production-scale problems to demonstrate their value. We implement them for completeness and to prepare for real deployments.

**Problem 1: Untapped Optimization Opportunities**. Chapters 11-13 already use Linalg operations for structured computation—matrix multiplications lower to `linalg.matmul`, element-wise operations to `linalg.generic`. This high-level IR provides semantic richness that enables optimization, but we haven't yet applied aggressive transformations:

```cpp
// Chapters 11-13: Linalg operations (high-level, structured)
auto matmulOp = rewriter.create<linalg::MatmulOp>(
  loc, 
  ValueRange{lhs, rhs},        // Inputs
  ValueRange{output}           // Output (accumulated into)
);

// After createConvertLinalgToLoopsPass():
// Linalg → naive SCF loops (no tiling, no vectorization, no fusion)
auto iLoop = rewriter.create<scf::ForOp>(loc, zero, M, one);
  auto jLoop = rewriter.create<scf::ForOp>(loc, zero, N, one);
    auto kLoop = rewriter.create<scf::ForOp>(loc, zero, K, one);
      // Scalar operations...
```

The Linalg operations carry semantic information ("this is matrix multiplication"), but the default lowering to loops is naive—it doesn't apply transformations. Consequently:

- **No tiling**: Default lowering generates monolithic loops without cache-friendly blocking
- **No vectorization**: Loops lower to scalar operations (one float32 per cycle instead of 8-16 with SIMD)
- **No fusion**: Each Linalg operation lowers independently, preventing producer-consumer fusion

**Problem 2: Sequential Operations Waste Memory Bandwidth**. Operations execute independently:

```python
# Attention: separate operations
Q = query_proj(x)      # [seq_len, d_model]
K = key_proj(x)        # [seq_len, d_model]
V = value_proj(x)      # [seq_len, d_model]
scores = Q @ K.T       # [seq_len, seq_len]
masked = mask(scores)  # [seq_len, seq_len]
weights = softmax(masked)  # [seq_len, seq_len]
output = weights @ V   # [seq_len, d_model]
```

Each operation writes results to memory, then the next operation reads from memory. For production models, memory bandwidth (100-500 GB/s DRAM, 1-5 TB/s L1 cache) becomes the bottleneck—not compute (1-20 TFLOPS). Fusing operations eliminates intermediate materialization: compute query projections, immediately use in attention score computation, never write intermediate tensors to memory.

**Problem 3: Redundant Computation in Generation**. Autoregressive generation reprocesses the entire sequence every iteration:

```python
tokens = [prompt_tokens]

for _ in range(max_new_tokens):
    logits = gpt_forward(tokens)  # Recomputes attention for ALL tokens
    next_token = sample(logits[-1])
    tokens.append(next_token)
```

At token 20, the model recomputes attention for tokens 0-19 despite their keys and values being unchanged. This is O(N²) complexity: token N recomputes N previous tokens. The cost grows quadratically—generating 100 tokens requires 100 + 99 + ... + 1 = 5,050 forward passes worth of computation. This redundancy affects all model scales equally.

**Hardware Underutilization**. Modern CPUs have powerful SIMD units:

- AVX2 (2013+): 256-bit registers, 8× float32 operations per cycle
- AVX-512 (2017+): 512-bit registers, 16× float32 operations per cycle

Scalar code uses ~6-12% of CPU compute capability. The gap is wasted potential—especially for production-scale models where compute matters.

## 14.2 High-Level IR with Linalg

The first optimization isn't faster code—it's better IR. MLIR's **Linalg dialect** provides high-level operations for linear algebra (matrix multiplication, convolution, element-wise operations). Linalg operations are semantic: they encode what computation happens, not how. This semantic richness enables aggressive optimization through pattern recognition.

**The Abstraction Ladder**. MLIR's dialects form an abstraction hierarchy:

```
High Level (Semantic)
  ↓
Transformer dialect: transformer.attention, transformer.ffn
  ↓
Linalg dialect: linalg.matmul, linalg.generic
  ↓
SCF dialect: scf.for, scf.if
  ↓
Arith dialect: arith.addf, arith.mulf
  ↓
LLVM dialect: llvm.fadd, llvm.fmul
  ↓
Low Level (Mechanical)
```

Chapters 11-13 already use the Linalg dialect as an intermediate representation: Transformer → Linalg → SCF → Arith → LLVM. However, the default `createConvertLinalgToLoopsPass()` performs naive lowering without optimization. Chapter 14 applies advanced transformations before lowering: Transformer → Linalg → **(Transform Dialect optimizations)** → SCF → Arith → LLVM.

**Linalg Matrix Multiplication**. Replace 40+ lines of nested loops with semantic operation:

```cpp
// src/TransformerToLinalg.cpp
struct MatMulOpLowering : public OpRewritePattern<transformer::MatMulOp> {
  using OpRewritePattern::OpRewritePattern;
  
  LogicalResult matchAndRewrite(transformer::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();  // [M, K]
    Value rhs = op.getRhs();  // [K, N]
    
    auto lhsType = lhs.getType().cast<RankedTensorType>();
    auto rhsType = rhs.getType().cast<RankedTensorType>();
    
    int64_t M = lhsType.getShape()[0];
    int64_t K = lhsType.getShape()[1];
    int64_t N = rhsType.getShape()[1];
    
    // Create output tensor [M, N]
    auto outputType = RankedTensorType::get({M, N}, lhsType.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getFloatAttr(lhsType.getElementType(), 0.0)
    );
    Value output = rewriter.create<tensor::EmptyOp>(
      loc, outputType.getShape(), outputType.getElementType()
    );
    Value outputFilled = rewriter.create<linalg::FillOp>(
      loc, zero, output
    ).result();
    
    // Create Linalg matmul operation
    auto matmulOp = rewriter.create<linalg::MatmulOp>(
      loc, 
      ValueRange{lhs, rhs},        // Inputs
      ValueRange{outputFilled}     // Output (accumulated into)
    );
    
    rewriter.replaceOp(op, matmulOp.getResult(0));
    return success();
  }
};
```

The `linalg.matmul` operation is **structured**: it has well-defined semantics (C += A @ B), known iteration space (M×N outer, K reduction), and predictable data access patterns. Optimizers leverage this structure.

**Linalg Generic for Element-Wise Operations**. Not all operations have dedicated Linalg ops. `linalg.generic` handles arbitrary element-wise computations:

```cpp
// GELU activation: y = x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
struct GELUOpLowering : public OpRewritePattern<transformer::GELUOp> {
  using OpRewritePattern::OpRewritePattern;
  
  LogicalResult matchAndRewrite(transformer::GELUOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    auto inputType = input.getType().cast<RankedTensorType>();
    
    // Create output tensor
    Value output = rewriter.create<tensor::EmptyOp>(
      loc, inputType.getShape(), inputType.getElementType()
    );
    
    // Constants
    Value half = createF32Constant(rewriter, loc, 0.5);
    Value one = createF32Constant(rewriter, loc, 1.0);
    Value coeff = createF32Constant(rewriter, loc, 0.044715);
    Value sqrtTwoPi = createF32Constant(rewriter, loc, 0.7978845608);  // √(2/π)
    
    // Create generic operation
    SmallVector<AffineMap> indexingMaps = {
      AffineMap::getMultiDimIdentityMap(inputType.getRank(), rewriter.getContext()),
      AffineMap::getMultiDimIdentityMap(inputType.getRank(), rewriter.getContext())
    };
    
    SmallVector<utils::IteratorType> iteratorTypes(
      inputType.getRank(), utils::IteratorType::parallel
    );
    
    auto geluOp = rewriter.create<linalg::GenericOp>(
      loc, input.getType(), ValueRange{input}, ValueRange{output},
      indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value x = args[0];
        
        // Compute GELU: x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        Value x2 = b.create<arith::MulFOp>(loc, x, x);
        Value x3 = b.create<arith::MulFOp>(loc, x2, x);
        Value cx3 = b.create<arith::MulFOp>(loc, coeff, x3);
        Value inner = b.create<arith::AddFOp>(loc, x, cx3);
        Value scaled = b.create<arith::MulFOp>(loc, sqrtTwoPi, inner);
        Value tanhVal = b.create<math::TanhOp>(loc, scaled);
        Value onePlusTanh = b.create<arith::AddFOp>(loc, one, tanhVal);
        Value halfTerm = b.create<arith::MulFOp>(loc, half, onePlusTanh);
        Value result = b.create<arith::MulFOp>(loc, x, halfTerm);
        
        b.create<linalg::YieldOp>(loc, result);
      }
    );
    
    rewriter.replaceOp(op, geluOp.getResult(0));
    return success();
  }
};
```

The `iteratorTypes` attribute specifies `parallel`—all iterations are independent, enabling vectorization. The compiler knows: "This operation applies the same computation to every element independently."

**Why Linalg?** Three key benefits:

1. **Pattern Recognition**: Optimizers match Linalg operations by name (`linalg.matmul`, `linalg.generic`) rather than analyzing complex loop nests

2. **Indexing Maps**: Affine maps describe data access patterns, enabling fusion analysis (producer output indexing matches consumer input indexing → fusible)

3. **Iterator Types**: `parallel` vs `reduction` semantics enable aggressive transformations (parallel iterations vectorize, reorder freely)

Optimized Linalg IR (with tiling, vectorization, fusion) is **semantically equivalent** to Chapters 11-13's naive loop lowering—same computation, but with transformations applied before conversion to SCF. The educational value is learning these production-grade optimization techniques, not measuring speedup on nano GPT (which is too small to show significant gains).

## 14.3 Declarative Rewrite Rules (DRR)

Chapter 9 introduced TableGen for defining operations but deferred **Declarative Rewrite Rules (DRR)**—a TableGen-based pattern matching system for expressing optimizations declaratively. DRR eliminates boilerplate C++ for common transformations, making optimization passes more readable and maintainable.

**The C++ Pattern Problem**. Chapter 9's lowering patterns required substantial C++ code:

```cpp
// C++ pattern: ~30 lines per transformation
struct SimplifyDoubleTranspose : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  
  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    // Match: transpose(transpose(x))
    auto innerTranspose = op.getInput().getDefiningOp<TransposeOp>();
    if (!innerTranspose)
      return failure();
    
    // Rewrite: replace with x
    rewriter.replaceOp(op, innerTranspose.getInput());
    return success();
  }
};
```

For simple algebraic simplifications (transpose(transpose(x)) → x, add(x, 0) → x), writing 30 lines of C++ is tedious and error-prone.

**DRR Solution: TableGen Patterns**. DRR expresses the same transformation in ~3 lines of TableGen:

```tablegen
// DRR pattern: 3 lines, no C++
def SimplifyDoubleTranspose : Pat<
  (TransposeOp (TransposeOp $x)),  // Match pattern
  (replaceWithValue $x)            // Replacement
>;
```

TableGen generates the C++ pattern matching code automatically. This is **declarative**: you specify what to match and what to replace it with, not how to implement the matching logic.

**DRR Pattern Structure**. A DRR pattern has three components:

1. **Match Pattern**: DAG (Directed Acyclic Graph) representing operations to match
2. **Constraints** (optional): Additional conditions (type constraints, attribute checks)
3. **Result Pattern**: DAG representing replacement operations

Example patterns for transformer operations:

```tablegen
// DRR patterns for optimization
#ifndef TRANSFORMER_OPS_PATTERNS
#define TRANSFORMER_OPS_PATTERNS

include "TransformerOps.td"

// Pattern 1: Eliminate double negation
// -(-x) → x
def SimplifyDoubleNegate : Pat<
  (NegateOp (NegateOp $x)),
  (replaceWithValue $x)
>;

// Pattern 2: Add identity elimination
// x + 0 → x
def AddZeroElim : Pat<
  (AddOp $x, (ConstantOp ConstantAttr<F32Attr, "0.0">)),
  (replaceWithValue $x)
>;

// Pattern 3: Multiply by one elimination
// x * 1 → x
def MulOneElim : Pat<
  (MulOp $x, (ConstantOp ConstantAttr<F32Attr, "1.0">)),
  (replaceWithValue $x)
>;

// Pattern 4: Transpose simplification
// transpose(transpose(x)) → x
def TransposeSimplify : Pat<
  (TransposeOp (TransposeOp $x)),
  (replaceWithValue $x)
>;

// Pattern 5: Constant folding for transpose
// Transpose of constant matrix can be computed at compile time
def FoldConstantTranspose : Pat<
  (TransposeOp (ConstantOp $value)),
  (ConstantOp (TransposeConstant $value)),
  [(TransposeConstant $value)]  // C++ helper for constant transposition
>;

#endif // TRANSFORMER_OPS_PATTERNS
```

**Advanced DRR: Type Constraints**. Patterns can specify type constraints:

```tablegen
// Only match float32 tensors
def AddF32Identity : Pat<
  (AddOp:$result $x, (ConstantOp ConstantAttr<F32Attr, "0.0">)),
  (replaceWithValue $x),
  [(F32Tensor $result)]  // Type constraint
>;

// Match any shaped type
def TransposeAnyShape : Pat<
  (TransposeOp:$result (TransposeOp $x)),
  (replaceWithValue $x),
  [(AnyRankedTensor $result)]
>;
```

**Using DRR in Passes**. Generated patterns integrate into pass pipelines:

```cpp
// src/TransformerPasses.cpp
#include "TransformerOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// Include generated pattern definitions
#include "TransformerOpsPatterns.inc"

struct TransformerCanonicalizerPass
    : public PassWrapper<TransformerCanonicalizerPass, OperationPass<func::FuncOp>> {
  
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    
    // Populate with DRR-generated patterns
    populateWithGenerated(patterns);
    
    // Apply patterns greedily
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
```

The `populateWithGenerated()` function is auto-generated from TableGen definitions, adding all DRR patterns to the pattern set.

**Why DRR?** Five key advantages:

1. **Conciseness**: 3 lines vs 30 lines for simple patterns
2. **Readability**: Declarative syntax makes intent obvious
3. **Maintainability**: Changing pattern requires editing TableGen, not C++
4. **Type Safety**: TableGen validates pattern structure at generation time
5. **Consistency**: All patterns follow same structure, reducing bugs

**When to Use DRR vs C++ Patterns**:

- **DRR**: Simple algebraic transformations, pattern-based rewrites, canonicalization rules
- **C++ Patterns**: Complex logic, runtime decisions, lowering passes with control flow

DRR complements C++ patterns—use the right tool for each transformation.

## 14.4 Canonicalization Patterns

**Canonicalization** is the process of transforming IR into a canonical (standard) form, eliminating redundancies and simplifying operations. Every well-designed dialect should define canonicalization patterns—simplifications that preserve semantics while reducing IR complexity.

**What is Canonicalization?** Consider these equivalent expressions:

```
x + 0 ≡ x
x * 1 ≡ x
transpose(transpose(x)) ≡ x
-(-x) ≡ x
max(x, x) ≡ x
```

Canonicalization rewrites the left side to the right side, producing simpler IR that's easier to optimize and analyze. MLIR runs canonicalization between major transformation passes to clean up IR.

**Defining Canonicalization in TableGen**. Operations declare their canonicalization patterns:

```tablegen
// inc/TransformerOps.td
def Transformer_AddOp : Transformer_Op<"add", [Pure, Commutative]> {
  let summary = "Element-wise addition";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  
  // Canonicalization patterns (DRR)
  let hasCanonicalizer = 1;
}

// Canonicalization patterns defined separately
def AddZeroCanon : Pat<
  (Transformer_AddOp $x, (Transformer_ConstantOp ConstantAttr<F32Attr, "0.0">)),
  (replaceWithValue $x)
>;

def AddCommute : Pat<
  (Transformer_AddOp (Transformer_ConstantOp $c), $x),
  (Transformer_AddOp $x, (Transformer_ConstantOp $c)),
  [(Transformer_IsNotConstant $x)]  // Only if $x is not constant
>;
```

**Canonicalizer Pass**. MLIR provides a built-in canonicalizer pass that applies all registered canonicalization patterns:

```cpp
// Run canonicalizer in pipeline
mlir::PassManager pm(context);
pm.addPass(mlir::createCanonicalizerPass());
pm.run(module);
```

The canonicalizer iterates until no more patterns apply (fixpoint), producing maximally simplified IR.

**Complex Canonicalization: Constant Folding**. Some canonicalization requires computation:

```tablegen
// Fold constant arithmetic at compile time
def FoldConstantAdd : Pat<
  (Transformer_AddOp 
    (Transformer_ConstantOp $a), 
    (Transformer_ConstantOp $b)),
  (Transformer_ConstantOp (AddConstants $a, $b)),
  [(AddConstants $a, $b)]  // C++ helper
>;
```

The `AddConstants` helper is defined in C++:

```cpp
// src/TransformerOps.cpp
Attribute AddConstants(Attribute a, Attribute b) {
  auto aFloat = a.cast<FloatAttr>();
  auto bFloat = b.cast<FloatAttr>();
  
  double result = aFloat.getValueAsDouble() + bFloat.getValueAsDouble();
  return FloatAttr::get(aFloat.getType(), result);
}
```

This performs arithmetic at **compile time**, producing a single constant instead of runtime computation.

**Transformer Dialect Canonicalization**. Complete example for transformer operations:

```tablegen
// Arithmetic identities
def AddZero : Pat<(AddOp $x, (ConstantOp Zero)), (replaceWithValue $x)>;
def MulOne : Pat<(MulOp $x, (ConstantOp One)), (replaceWithValue $x)>;
def MulZero : Pat<(MulOp $x, (ConstantOp Zero)), (ConstantOp Zero)>;

// Transpose simplifications
def TransposeInverse : Pat<
  (TransposeOp (TransposeOp $x)),
  (replaceWithValue $x)
>;

// Softmax simplifications
def SoftmaxConstant : Pat<
  (SoftmaxOp (ConstantOp $c)),
  (ConstantOp (SoftmaxConstantFold $c))
>;

// ReLU simplifications
def ReLUConstant : Pat<
  (ReLUOp (ConstantOp $c)),
  (ConstantOp (ReLUConstantFold $c))
>;
```

These patterns eliminate redundant operations, simplify IR, and enable subsequent optimizations to be more effective.

**Why Canonicalization Matters**. Three key benefits:

1. **Optimization Enablement**: Simplified IR exposes optimization opportunities (fusion, constant propagation)
2. **IR Quality**: Canonical form makes IR easier to analyze and transform
3. **Code Size**: Fewer operations reduce memory footprint and compilation time

Canonicalization is **cheap** (pattern matching) and **high-value** (enables expensive optimizations).

## 14.5 Custom OpInterface

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

**Providing Implementations**. Implement methods in C++:

```cpp
// src/TransformerOps.cpp
#include "ShapeInferenceOpInterface.h"

// MatMul: [M, K] @ [K, N] -> [M, N]
SmallVector<int64_t> MatMulOp::inferOutputShape(
    ArrayRef<SmallVector<int64_t>> inputShapes) {
  assert(inputShapes.size() == 2 && "MatMul expects 2 inputs");
  
  auto lhsShape = inputShapes[0];  // [M, K]
  auto rhsShape = inputShapes[1];  // [K, N]
  
  assert(lhsShape.size() == 2 && rhsShape.size() == 2 && "MatMul expects 2D tensors");
  assert(lhsShape[1] == rhsShape[0] && "MatMul dimension mismatch");
  
  return {lhsShape[0], rhsShape[1]};  // [M, N]
}

// Add: broadcast shapes
SmallVector<int64_t> AddOp::inferOutputShape(
    ArrayRef<SmallVector<int64_t>> inputShapes) {
  assert(inputShapes.size() == 2 && "Add expects 2 inputs");
  
  // Simple broadcasting: assume same shape or one is scalar
  auto lhsShape = inputShapes[0];
  auto rhsShape = inputShapes[1];
  
  if (lhsShape == rhsShape)
    return lhsShape;
  
  // Handle broadcasting (simplified)
  return lhsShape.size() >= rhsShape.size() ? lhsShape : rhsShape;
}

// ReLU: shape unchanged
SmallVector<int64_t> ReLUOp::inferOutputShape(
    ArrayRef<SmallVector<int64_t>> inputShapes) {
  assert(inputShapes.size() == 1 && "ReLU expects 1 input");
  return inputShapes[0];
}
```

Each operation provides its specific shape inference logic.

**Using the Interface**. Generic algorithms use the interface:

```cpp
// Shape inference pass (works for ANY operation implementing the interface)
struct ShapeInferencePass : public PassWrapper<ShapeInferencePass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    getOperation().walk([](Operation *op) {
      // Check if operation implements ShapeInferenceOpInterface
      if (auto shapeOp = dyn_cast<ShapeInferenceOpInterface>(op)) {
        // Collect input shapes
        SmallVector<SmallVector<int64_t>> inputShapes;
        for (Value operand : op->getOperands()) {
          auto tensorType = operand.getType().cast<RankedTensorType>();
          inputShapes.push_back(llvm::to_vector(tensorType.getShape()));
        }
        
        // Call interface method (polymorphic!)
        SmallVector<int64_t> outputShape = shapeOp.inferOutputShape(inputShapes);
        
        // Update result type
        auto resultType = op->getResult(0).getType().cast<RankedTensorType>();
        auto newType = RankedTensorType::get(outputShape, resultType.getElementType());
        op->getResult(0).setType(newType);
      }
    });
  }
};
```

This pass works for **any operation** that implements `ShapeInferenceOpInterface`—no need to know specific operation types. This is true polymorphism in MLIR.

**Interface Benefits**:

1. **Extensibility**: Add new operations without modifying passes
2. **Code Reuse**: Write generic algorithms once, use for all implementing operations
3. **Type Safety**: TableGen generates interfaces with compile-time checking
4. **Documentation**: Interface definition documents expected behavior

Interfaces are MLIR's answer to the expression problem—adding new operations without modifying existing code.

## 14.6 Transform Dialect: Modern Optimization

With operations expressed as Linalg and patterns defined declaratively, we can optimize using Transform dialect. Transform dialect is MLIR's modern approach to optimization—declarative, transparent, and composable.

**Legacy Passes vs Transform Dialect**. Early MLIR relied on imperative passes:

```cpp
// Old approach (inflexible)
mlir::PassManager pm(context);
pm.addNestedPass<func::FuncOp>(mlir::createLinalgTilingPass(
  LinalgTilingOptions().setTileSizes({32, 32, 32})
));
pm.addNestedPass<func::FuncOp>(mlir::createLinalgElementwiseOpFusionPass());
pm.run(module);
```

Problems: black-box, inflexible options, hard to debug, not composable.

**Transform Dialect Approach**. Express optimizations as transformation scripts:

```mlir
// Modern approach: declarative, transparent
transform.sequence failures(propagate) {
^bb0(%module: !transform.any_op):
  // Match operations
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %module
    : (!transform.any_op) -> !transform.any_op
  
  // Tile for cache locality (L1: 32 KB)
  %tiled, %loops = transform.structured.tile_using_for %matmul [32, 32, 32]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  
  // Vectorize (AVX2: 8-wide float32)
  transform.structured.vectorize %tiled : !transform.any_op
  
  // Cleanup
  %func = transform.structured.match ops{["func.func"]} in %module
    : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
}
```

Advantages: declarative (what not how), transparent (readable logic), composable (reorder/add/remove), debuggable (inspect IR between transformations).

**Tiling for Cache Locality**. Modern CPUs have cache hierarchies:

- L1: 32-64 KB, ~1 cycle latency
- L2: 256-512 KB, ~10 cycles latency
- L3: 8-32 MB, ~40 cycles latency
- DRAM: 16-128 GB, ~200 cycles latency

Tiling blocks computation into cache-fitting chunks (32×32×32 tiles = 12 KB, fits L1). For production models where data exceeds cache capacity, tiling provides 1.2-1.5× speedup through reduced memory latency.

**Fusion and Vectorization**. Fusion merges producer-consumer operations, eliminating intermediate buffers. Vectorization exploits SIMD (AVX2: 8× float32, AVX-512: 16× float32). For production models, fusion yields 1.3-1.7× speedup, vectorization 2-4× speedup.

**Theoretical Speedup for Production Models**:

| Optimization | Speedup (Production) | Why Nano GPT Doesn't Show This |
|--------------|---------------------|--------------------------------|
| Tiling | 1.2-1.5× | Data already fits L1 cache |
| Fusion | 1.3-1.7× | Bandwidth not bottleneck |
| Vectorization | 2.0-4.0× | Compute-light operations |
| **Combined** | **3-5×** | **All data cache-resident** |

These techniques are correct and production-ready—the educational GPT scale just doesn't stress the optimization targets.

## 14.7 KV Caching: Algorithmic Optimization

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

**Implementation in MLIR**:

```cpp
// inc/TransformerOps.td
def Transformer_AttentionCachedOp : Transformer_Op<"attention_cached", [Pure]> {
  let summary = "Attention with KV caching";
  let arguments = (ins
    AnyRankedTensor:$query,         // [1, d_model]
    AnyRankedTensor:$key_cache,     // [max_seq_len, d_model]
    AnyRankedTensor:$value_cache,   // [max_seq_len, d_model]
    I32:$pos,                       // Current position
    AnyRankedTensor:$wq, AnyRankedTensor:$wk,
    AnyRankedTensor:$wv, AnyRankedTensor:$wo
  );
  let results = (outs
    AnyRankedTensor:$output,
    AnyRankedTensor:$updated_key_cache,
    AnyRankedTensor:$updated_value_cache
  );
}
```

**Performance Impact**. For 20-token generation:

- Naive: O(20²) = 400 attention computations
- Cached: O(20) = 20 attention computations
- **Theoretical speedup: 20×**

This speedup applies to nano GPT and production models equally—it's pure algorithmic optimization, independent of hardware.

**Memory Cost**:

```
Per layer: 2 × max_seq_len × d_model × 4 bytes

Nano GPT (2 layers, d_model=64, max_seq_len=512):
  2 × 2 × 512 × 64 × 4 = 524,288 bytes ≈ 512 KB

GPT-2 Small (12 layers, d_model=768, max_seq_len=1024):
  2 × 12 × 1024 × 768 × 4 = 75,497,472 bytes ≈ 72 MB

GPT-3 (96 layers, d_model=12288, max_seq_len=2048):
  2 × 96 × 2048 × 12288 × 4 = 19,327,352,832 bytes ≈ 18 GB
```

For GPT-3, KV cache dominates memory usage. Production systems carefully manage cache memory across batches.

## 14.8 Lessons and Production Patterns

This chapter introduced advanced techniques—some show immediate benefit (KV caching, DRR, interfaces), others demonstrate production patterns even if nano GPT doesn't stress them (Transform dialect, tiling, fusion).

**Key Lessons**:

1. **Algorithmic Wins are Universal**: KV caching provides 10-100× speedup regardless of model scale
2. **Compiler Optimizations Scale-Dependent**: Tiling, fusion, vectorization matter for production models where memory bandwidth is bottleneck
3. **Declarative Beats Imperative**: DRR and Transform dialect are more maintainable than C++ passes
4. **Interfaces Enable Extensibility**: Generic algorithms work across operations without knowing types
5. **Measurement Validates Theory**: Theoretical speedups require appropriate problem scale

**Production Patterns**:

- **Progressive Lowering**: High-level IR → Linalg → Optimizations → SCF → LLVM
- **Declarative Transformations**: Use Transform dialect for complex optimization pipelines
- **Algorithmic Before Compiler**: Fix O(N²) algorithms before micro-optimizations
- **Interface-Based Passes**: Write generic passes using interfaces, not type-specific logic

**When to Apply Each Technique**:

| Technique | Nano GPT | Production GPT | Reason |
|-----------|----------|----------------|--------|
| KV Caching | ✅ Yes | ✅ Yes | Algorithmic, scale-independent |
| DRR Patterns | ✅ Yes | ✅ Yes | Code quality, maintainability |
| Interfaces | ✅ Yes | ✅ Yes | Extensibility, generic algorithms |
| Transform Dialect | ⚠️ Learn | ✅ Yes | Production standard, educational value |
| Tiling/Fusion | ❌ No gain | ✅ Yes | Requires memory bandwidth bottleneck |
| Vectorization | ❌ Minimal | ✅ Yes | Benefits compute-bound operations |

## 14.9 Summary

Chapter 14 introduced production-grade optimization techniques spanning declarative transformations (DRR, Transform dialect), advanced dialect features (interfaces, canonicalization), and algorithmic improvements (KV caching). These techniques represent modern MLIR practice—the same approaches used in production compilers at Google, Meta, and NVIDIA.

**Key Insights**:

- **DRR Simplifies Patterns**: TableGen-based patterns eliminate boilerplate C++ for common transformations
- **Canonicalization is Foundation**: Simplifying IR enables subsequent optimizations
- **Interfaces Enable Polymorphism**: Generic algorithms work across operations without type-specific logic
- **Transform Dialect is Production Standard**: Declarative transformations are more maintainable than imperative passes
- **KV Caching Dominates**: Algorithmic optimization (O(N²) → O(N)) provides 10-100× speedup at any scale

**Looking Ahead**. Chapter 15 introduces GPU concepts: CUDA programming model, memory hierarchy (global, shared, registers), kernel programming, and MLIR's GPU dialect. Chapter 16 completes the book with production serving: batching, dynamic batching, model parallelism, and multi-GPU inference. These chapters build on Chapter 14's optimization foundation to achieve true production-scale performance.

Chapter 14 completed the optimization arc: from basic operations (Chapters 1-9) through transformer architecture (Chapters 10-13) to production-grade optimizations. You now understand modern MLIR compilation—declarative, composable, and extensible. The techniques are production-ready; the scale is educational. Apply these to real models, and you'll achieve the theoretical speedups documented here.