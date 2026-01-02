# Chapter 7: Neural Network Operations and Computation Graphs

In Chapter 6, we implemented softmax—our first complete machine learning operation—using explicit loop constructions with SCF dialect and mathematical functions from the Math dialect. We built the operation from scratch, carefully managing numerical stability and loop-carried variables. This bottom-up approach taught us fundamental MLIR patterns, but modern ML frameworks don't ask developers to write loops for every operation. Instead, they provide high-level operations (matrix multiplication, activations, normalizations) that compose into larger computations. This chapter introduces that compositional approach, showing how to build complex neural network computations by combining simpler operations through a **computation graph**.

The computation graph is a foundational abstraction in modern machine learning systems. PyTorch, TensorFlow, JAX, and production serving frameworks all use graph representations internally, though they may hide them behind different APIs. A computation graph treats operations as nodes in a directed acyclic graph (DAG) where edges represent data dependencies—each operation consumes outputs from previous operations and produces outputs for subsequent ones. This structure enables powerful optimizations: fusion of adjacent operations, elimination of redundant computations, and whole-program analysis that wouldn't be possible with operation-at-a-time execution.

This chapter builds a computation graph API in C++ that tracks operations symbolically during graph construction, then generates complete MLIR modules when requested. We'll implement several core neural network operations—element-wise arithmetic (add, mul), matrix multiplication (the workhorse of neural networks), and activation functions (ReLU and softmax). Each operation will be built using MLIR's C++ APIs, demonstrating patterns for different operation types: element-wise operations that map naturally to SCF loops, matmul using the Linalg dialect's structured operations, and reductions using the loop-carried variable patterns from Chapter 6. By the end, we'll compose these operations into a complete two-layer neural network, showing how the same principles scale from single operations to full models.

## 7.1 From Eager to Deferred Execution

The operations we've built in previous chapters used **eager execution**: we generated MLIR, compiled it, and ran it immediately, all within a single function call. When Python code called `softmax(x)`, it received computed results directly—no intermediate representation, no opportunity to inspect or optimize the computation before execution. This immediate feedback makes development easy and intuitive, which is why NumPy and PyTorch's default mode work this way. However, eager execution has limitations for optimization: each operation executes independently, making it impossible to analyze relationships between operations or apply cross-operation optimizations.

**Deferred execution** (also called lazy evaluation or graph mode) takes a different approach: operations build a symbolic representation of the computation without executing anything. When you call `add(x, y)` in deferred mode, it doesn't compute the sum—instead, it records "there's an addition operation with these inputs" in a graph data structure. Only when you explicitly compile and execute the graph do the actual computations occur. This separation of graph construction from execution enables whole-program optimization, as the compiler sees the entire computation before generating any machine code.

TensorFlow 1.x used deferred execution exclusively, requiring developers to build complete graphs before running session.run(). TensorFlow 2.x defaults to eager execution but offers tf.function for deferred mode. JAX's jit decorator compiles functions to XLA graphs before execution. PyTorch introduced torch.jit.script and torch.jit.trace to capture graphs from eager code. These frameworks recognize that both modes have value: eager execution for development and debugging, deferred execution for production performance. The key insight is that deferred execution enables optimizations impossible in eager mode, at the cost of more complex mental models and debugging.

Our computation graph API follows the deferred execution pattern. Users build graphs by calling methods like `add()`, `mul()`, and `matmul()`, each of which returns an operation ID—just an integer identifying that operation in the graph. These IDs represent symbolic values, not computed arrays. Only when calling `compile()` do we generate MLIR from the graph structure, apply lowering passes, and JIT compile to machine code. This design mirrors TensorFlow 1.x and JAX's philosophy: separate "what to compute" (graph structure) from "how to compute it" (compilation) and "when to compute it" (execution with actual data).

The benefits of deferred execution become clear when building larger models. With eager execution, each operation's implementation is opaque to subsequent operations—matrix multiply doesn't know its result will immediately feed into ReLU, so it can't avoid materializing intermediate results. With a graph, we see that pattern explicitly and could fuse the operations, computing ReLU during the matrix multiply's final write loop. For transformers with dozens of operations per layer and hundreds of layers, such optimizations accumulate significantly. Chapter 10 will explore these optimizations in depth; for now, we focus on the graph structure that makes them possible.

## 7.2 Why Graphs? From Sequential Code to DAG Representation

Before diving into our computation graph implementation, we need to understand why graphs are necessary at all. When you write sequential Python or C++ code, operations execute in the order you write them. This linear execution model is intuitive but fundamentally limits what compilers can do: they can only optimize within basic blocks (straight-line code) and must assume operations have side effects that prevent reordering. For machine learning computations with dozens or hundreds of operations, this sequential view leaves enormous optimization opportunities on the table.

A **graph representation** captures the essential structure of a computation: which operations depend on which others. If operation B uses operation A's result, there's an edge from A to B. If operations C and D don't depend on each other, the graph shows they're independent—the compiler could execute them in parallel, reorder them, or fuse them with adjacent operations. This dependency information, implicit in sequential code, becomes explicit and analyzable in a graph. Every modern ML framework (TensorFlow, PyTorch, JAX, XLA) uses graphs internally precisely for this reason: they enable whole-program analysis and transformation.

Graphs also solve a practical API problem. When building neural networks, operations compose naturally: the output of one layer feeds into the next, activations follow linear transformations, losses depend on predictions. If we required users to manually allocate buffers and pass them between operations, the API would be cumbersome and error-prone. With a graph, users just say "relu of matmul of input" and the system tracks dependencies automatically, allocating buffers and generating efficient code as needed. The graph is both an optimization enabler and an ergonomic abstraction.

Our computation graph is a **directed acyclic graph (DAG)**—directed because data flows in one direction (from inputs through operations to outputs), acyclic because feedback loops would require recurrence mechanisms we don't support yet. In graph theory terms, variables are source nodes (no incoming edges), operations are internal nodes (incoming and outgoing edges), and the final output we compile is a sink node (no outgoing edges). The DAG property guarantees we can process operations in topological order: no operation is generated before its dependencies. (Topological ordering and graph-level optimizations are covered in detail in Chapter 10; for Chapters 7-9, we assume operations are added in correct dependency order.)

## 7.3 Graph Data Structure: Vectors and Integer IDs

Implementing a graph requires choosing a data structure that supports efficient construction (adding nodes/edges), traversal (following dependencies), and lookup (finding nodes by identifier). Our implementation uses a simple but effective approach: a **vector of operations** where each operation is identified by its position (index) in the vector. This indexed representation trades off some flexibility for simplicity and performance—it's the same strategy used in many production compilers.

The `GraphOperation` struct represents a single node in our graph:

```cpp
struct GraphOperation {
    enum class OpType {
        Input,      // Variable/placeholder (no dependencies)
        Add,        // Element-wise addition
        Mul,        // Element-wise multiplication
        MatMul,     // Matrix multiplication
        ReLU,       // Activation function
        Softmax     // Probability normalization
    };

    OpType type;                    // What operation this is
    std::vector<int> inputs;        // IDs of operations this depends on
    std::vector<int64_t> shape;     // Output shape
    int id;                         // Unique identifier (= vector index)
};
```

The `type` field identifies what computation to perform. The `inputs` vector contains integer IDs of operations this one depends on—these are the incoming edges in our graph. For a binary operation like Add, `inputs` has two elements; for unary operations like ReLU, it has one; for variables (graph leaves), it's empty. The `shape` describes the output tensor's dimensions, essential for allocating buffers during IR generation. The `id` is just the operation's position in the vector, providing a stable handle users can reference.

The `ComputationGraph` class manages this vector:

```cpp
class ComputationGraph {
public:
    explicit ComputationGraph(MLIRContext* ctx);

    int addVariable(const std::vector<int64_t>& shape);
    int add(int lhs, int rhs);
    int matmul(int lhs, int rhs);
    // ... other operations

    ModuleOp generateMLIR(int outputId, const std::string& funcName);

private:
    MLIRContext* context;
    std::vector<GraphOperation> operations;  // The graph!
    int nextId;
    // ... helper methods
};
```

The `operations` vector is our graph. When adding an operation, we append it to the vector and return its index:

```cpp
int ComputationGraph::add(int lhs, int rhs) {
    const auto& lhsShape = operations[lhs].shape;
    int id = nextId++;
    operations.emplace_back(OpType::Add, 
                           std::vector<int>{lhs, rhs},  // Dependencies
                           lhsShape,                     // Output shape
                           id);
    return id;
}
```

The caller receives an integer (the operation's ID) which they can pass to subsequent operations. For example, `z = graph.add(x, y)` returns an integer, and `w = graph.mul(z, v)` uses that integer to reference the addition's result. The graph structure is implicit in these integer references—when we later traverse the graph, we follow the IDs to find dependencies.

**Why a vector instead of a traditional graph structure with pointers?** Several reasons make this indexed approach attractive. First, **cache locality**: vector elements are contiguous in memory, so traversing operations has good cache behavior. Second, **stability**: operation IDs never change, even if we later add graph transformation passes that modify operations (stable IDs simplify debugging and error reporting). Third, **simplicity**: no need for graph node allocation/deallocation or pointer management—the vector handles memory automatically. Fourth, **serialization**: if we wanted to save/load graphs, integers serialize trivially while pointer-based structures require complex serialization logic.

The cost is inflexibility: we can't efficiently remove operations from the middle of the graph (would invalidate all subsequent IDs), and we can't insert operations between existing ones. For our use case—build graph once, generate IR, discard graph—these limitations don't matter. Production systems might use more sophisticated representations (hash tables, node pools, etc.) but the vector approach suffices for learning and prototyping.

## 7.4 Computation Graph Architecture

The architecture of our computation graph system follows a clean separation of concerns: the `ComputationGraph` class manages graph construction and IR generation, while the Python wrapper (via pybind11) provides the user-facing API. This separation allows us to keep C++ code focused on MLIR manipulation while Python handles user interaction, NumPy integration, and ergonomics.

When users build a graph, they create a `Graph` object and call methods like `variable()`, `add()`, `matmul()`, etc. Each call appends a `GraphOperation` to the internal vector and returns an integer ID. These IDs become handles for subsequent operations—instead of passing tensors around (eager execution), we pass IDs around (symbolic execution). The graph accumulates operations without executing anything; it's just recording the computation's structure.

When adding an operation, we record its type, input operation IDs, output shape, and assign it the next available ID. For example, if we have variables `x` (ID 0) and `y` (ID 1), then `add(0, 1)` creates operation ID 2 of type Add with inputs [0, 1] and shape matching the operands. Subsequent operations refer to operation 2 if they need the sum. This indexed representation is simple and efficient—lookups are O(1) array accesses, and the topology is implicit in the input references.

The key method is `generateMLIR()`, which takes an output operation ID and function name, then generates a complete MLIR module implementing that computation. The generation process involves several steps: first, identify all variable operations (the leaves of our DAG) which become function parameters. Second, construct a function signature taking one memref parameter per variable, plus an output memref. Third, recursively build MLIR operations in depth-first order, memoizing results to avoid regenerating shared subexpressions. Fourth, copy the final result to the output buffer. The result is a complete `func.func` ready for lowering and JIT compilation.

Recursive generation with memoization is critical for efficiency. If operation 5 depends on operation 3, and operation 6 also depends on operation 3, we want to generate operation 3's MLIR exactly once, not twice. We maintain a `valueMap` mapping operation IDs to MLIR `Value` objects representing their results. Before building an operation, we check if it's already in the map; if so, return the cached value. If not, recursively build its inputs (which populates the map with their results), then build the operation itself using those input values, cache the result, and return it. This ensures each operation is built exactly once, regardless of how many downstream operations consume it.

The separation between graph construction (recording operations) and IR generation (building MLIR) is philosophically important. During construction, we're working at a high semantic level—"add these tensors", "multiply these matrices"—without committing to implementation details. During generation, we make concrete decisions about memory layout, loop structure, and operation fusion. This separation allows the same graph to target different backends (CPU vs GPU), different optimization levels, or different MLIR dialect combinations, all from a single high-level description.

## 7.5 Element-Wise Operations: Add and Multiply

Element-wise operations apply the same scalar operation to corresponding elements of input tensors: addition, multiplication, maximum, and so forth. These operations are embarrassingly parallel—each output element can be computed independently—and our implementation uses **linalg.generic**, a high-level operation that expresses element-wise computations through affine maps and region-based computation.

**Linalg.generic** is MLIR's Swiss Army knife for structured array operations. It takes a set of input and output tensors, affine maps describing how to access elements, iterator types (parallel or reduction), and a region containing the scalar computation. For element-wise operations, all iterators are parallel (no reductions), the affine maps are all identity maps (element i,j in output corresponds to element i,j in inputs), and the region contains the scalar arithmetic operation.

For addition of two tensors, we create a `linalg.generic` operation with:
- Two input tensors and one output tensor (initialized with `tensor.empty`)
- Identity affine maps for all operands: `(d0, d1) -> (d0, d1)`
- All parallel iterators: `["parallel", "parallel"]` for 2D
- A region that adds the two scalar elements: `%sum = arith.addf %lhs, %rhs`

This high-level specification generates the same efficient loops as explicit SCF code but provides more optimization opportunities—the Linalg dialect knows this is element-wise and can apply fusion, tiling, and vectorization transformations automatically.

## 7.6 Matrix Multiplication with Linalg Dialect

Matrix multiplication is the computational heart of neural networks—every linear layer, every attention mechanism, every learned transformation performs matrix multiplication. Unlike element-wise operations where output position (i,j) depends only on inputs at (i,j), matrix multiply has complex dependencies: output[i,j] depends on all elements in row i of the left matrix and column j of the right matrix. The **Linalg dialect** provides `linalg.matmul`, a structured operation that encapsulates this computation at a high semantic level.

The Linalg dialect (short for "linear algebra") provides structured operations for common array computations. Operations like `linalg.matmul`, `linalg.conv_2d`, and `linalg.reduce` encapsulate common access patterns at a high semantic level, avoiding manual loop writing while preserving optimization opportunities. `linalg.matmul` specifically represents C = A × B as a single operation, leaving the actual loop implementation to lowering passes. This high-level representation enables optimizations that would be difficult to recognize from explicit loops: tiling for cache locality, fusion with adjacent operations, or replacement with vendor-optimized BLAS libraries.

Using `linalg.matmul` involves two steps: **initialize the output** and **perform the multiply-accumulate**. First, we create an output tensor using `tensor.empty` with the appropriate shape (if A is M×K and B is K×N, result is M×N). The output needs initialization to zero since matmul accumulates results. We use `linalg.fill` to create a zero-initialized tensor:

```mlir
%c0 = arith.constant 0.0 : f32
%empty = tensor.empty() : tensor<2x4xf32>
%zero_init = linalg.fill ins(%c0 : f32) outs(%empty : tensor<2x4xf32>) -> tensor<2x4xf32>
```

Then we create the matmul operation:

```mlir
%result = linalg.matmul ins(%lhs, %rhs : tensor<2x3xf32>, tensor<3x4xf32>)
                        outs(%zero_init : tensor<2x4xf32>) -> tensor<2x4xf32>
```

The operation semantics are: for each output element C[i,j], compute the sum over k of A[i,k] × B[k,j] and add it to the output accumulator. Since we initialized the output to zero, we get the standard matrix product. The generated IR works directly with tensors—no explicit memory allocation, no manual buffer management, just high-level operations that will be lowered to efficient code later.

When `linalg.matmul` lowers to loops (via the `convert-linalg-to-loops` pass), it expands into three nested SCF loops: outer over rows, middle over columns, inner over the reduction dimension with loop-carried variables for accumulation. The bufferization passes then convert tensors to memrefs with proper memory allocation. This layered approach separates concerns: at the graph level, we think about mathematical operations on tensors; at the lowering level, we handle memory and loops.

## 7.7 Activation Functions: ReLU and Softmax

Activation functions introduce non-linearity into neural networks. Without them, stacking multiple linear layers would just compute another linear function—we'd gain nothing from depth. ReLU (Rectified Linear Unit) and softmax are two of the most common activations, appearing in nearly every modern architecture. Both are implemented using high-level linalg operations that will be efficiently lowered during compilation.

**ReLU** is computationally trivial but critically important: output max(0, input) element-wise. We implement it using `linalg.generic` with a scalar computation region that compares with zero and selects the maximum. The operation takes one input tensor and produces one output tensor, with identity affine maps and parallel iterators.

Inside the linalg.generic region, we have scalar operations:
```mlir
^bb0(%in: f32, %out: f32):
  %c0 = arith.constant 0.0 : f32
  %max = arith.maximumf %in, %c0 : f32
  linalg.yield %max : f32
```

The `arith.maximumf` operation computes the maximum of two floating-point values, handling NaN values according to IEEE 754 semantics (NaN propagates). This maps to efficient instructions on modern hardware—typically a single MAXSS/MAXPS instruction on x86 or FMAX on ARM, with no branching required.

**Softmax** in the computation graph reuses the multi-pass algorithm from Chapter 6, but now implemented with linalg operations. The four-pass algorithm remains the same:
1. **Find maximum**: Use `linalg.reduce` with `arith.maximumf` to find max value
2. **Subtract and exponentiate**: Use `linalg.generic` with `math.exp` to compute exp(x - max)
3. **Sum exponentials**: Use `linalg.reduce` with `arith.addf` to compute the normalizing sum
4. **Normalize**: Use `linalg.generic` with `arith.divf` to divide each exp by the sum

Each pass is a structured linalg operation. The reductions use `linalg.reduce` which specifies:
- Input and output tensors
- The dimensions to reduce over (dimension 0 for 1D)
- A combiner region with the reduction operation (max or add)

For the element-wise passes (exp and normalize), we use `linalg.generic` as in element-wise operations, but the scalar computation is more complex—calling `math.exp` or `arith.divf`.

This multi-level approach provides excellent separation of concerns: at the graph level, we compose four linalg operations; at the lowering level, each linalg operation becomes efficient loops; at the LLVM level, the math operations become vectorized or library calls. The algorithm structure is clear at the high level, but performance comes from systematic lowering.

## 7.8 Recursive IR Generation from Graph Structure

The `generateMLIR()` method transforms our symbolic graph into concrete MLIR operations through depth-first traversal with memoization. The generated IR uses tensor operations, with bufferization handled by later passes.

**Step 1: Identify Variables**

We scan all operations in the graph, collecting those with type `Input`—these are variables, placeholders for actual data that will be provided at execution time. Each variable becomes a function parameter. If we have three variables with shapes [4], [4], and [2,3], our function signature will take three tensor parameters: `tensor<4xf32>`, `tensor<4xf32>`, and `tensor<2x3xf32>`.

**Step 2: Create Function Signature**

Using the collected variables, we construct a `FunctionType` taking tensor inputs and returning a single tensor output:

```mlir
func.func @compute(%x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32>
```

This functional style (explicit return value rather than out-parameters) is cleaner and more composable. The conversion to out-parameters happens during bufferization via the `buffer-results-to-out-params` pass.

**Step 3: Recursive Operation Building**

The `buildOperation()` method recursively generates tensor operations. Given an operation ID:

1. Check if that operation's result is already in `valueMap` (memoization)
2. If it's a variable, return the corresponding function parameter
3. Otherwise, recursively build all input operations to get their tensor results
4. Call the appropriate builder (`buildElementWiseOp`, `buildMatMul`, etc.)
5. Cache the resulting tensor value and return it

Each builder returns a tensor value—the result of the linalg operation. These values flow through subsequent operations without any explicit memory management at this level.

**Step 4: Return Result**

After building the output operation, we have a tensor value representing the final result. We simply return it from the function:

```cpp
builder.create<func::ReturnOp>(loc, ValueRange{outputValue});
```

No explicit copy loops, no buffer management—just return the tensor. The bufferization passes later will:
- Allocate buffers for intermediate results
- Convert return value to output parameter
- Insert necessary buffer copies
- Manage buffer lifetimes and deallocations

This separation of concerns is powerful: graph construction focuses on mathematical semantics (what to compute), while bufferization handles implementation details (where to store data, when to copy, etc.).

**Generated IR Structure**

The complete module contains a function with tensor operations:

```mlir
func.func @compute(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<2x4xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4xf32>) -> tensor<2x4xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>)
                      outs(%1 : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %2 : tensor<2x4xf32>
}
```

Clean, high-level, optimizable. The bufferization and lowering passes transform this into efficient imperative code with explicit loops and memory management.

## 7.9 Building Element-Wise Operations: C++ Implementation

Let's examine the C++ implementation of element-wise operations using `linalg.generic`, showing how to construct affine maps and scalar computation regions. This implementation pattern is fundamental to many tensor operations and demonstrates MLIR's composability—complex operations built from simple primitives.

**Creating Affine Maps**

Element-wise operations access corresponding elements from inputs and output, expressed via **identity affine maps**. For 2D tensors:

```cpp
Value buildElementWiseOp(OpBuilder& builder, Location loc, 
                         Value lhs, Value rhs, OpType opType) {
    auto lhsType = lhs.getType().cast<RankedTensorType>();
    auto rank = lhsType.getRank();

    // Create identity affine maps: (d0, d1, ...) -> (d0, d1, ...)
    SmallVector<AffineMap> indexingMaps;
    auto identityMap = builder.getMultiDimIdentityMap(rank);
    indexingMaps.push_back(identityMap);  // lhs
    indexingMaps.push_back(identityMap);  // rhs
    indexingMaps.push_back(identityMap);  // output
```

The identity map means: for position (i,j) in the output, read from position (i,j) in both inputs. All maps are identical because element-wise operations have 1:1 correspondence across all tensors.

**Defining Iterator Types**

Element-wise operations have all-parallel iteration—no reductions, every output element is independent. We use `utils::IteratorType::parallel` to specify this:

```cpp
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
```

For 2D, this creates `[parallel, parallel]`. Parallel iterators tell the compiler these can execute in any order or concurrently, enabling optimizations like parallelization and vectorization.

**Creating Output Tensor**

We need an output tensor with the same shape as inputs. Since we support dynamic shapes (where dimensions might be unknown at compile time), we must construct the shape at runtime using `tensor.dim` operations:

```cpp
    // Create empty tensor for result
    SmallVector<OpFoldResult> dynSizes;
    for (int i = 0; i < rank; ++i) {
        dynSizes.push_back(builder.create<tensor::DimOp>(loc, lhs, i).getResult());
    }
    Value emptyTensor = builder.create<tensor::EmptyOp>(loc, dynSizes, lhsType.getElementType());
```

`tensor.empty` declares a tensor without allocating or initializing—it's a placeholder that bufferization will turn into actual memory allocation later. By using dynamic sizes, our code works for both static and dynamic input shapes.

**Building linalg.generic with Region**

The core operation specifies inputs, outputs, indexing, iterators, and the scalar computation:

```cpp
    auto genericOp = builder.create<linalg::GenericOp>(
        loc,
        lhsType, // Result type
        ValueRange{lhs, rhs}, // Inputs
        ValueRange{emptyTensor}, // Outputs
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder& b, Location loc, ValueRange args) {
            // args[0] = lhs element, args[1] = rhs element, args[2] = output element
            Value result;
            if (opType == OpType::Add) {
                result = b.create<arith::AddFOp>(loc, args[0], args[1]);
            } else {  // Mul
                result = b.create<arith::MulFOp>(loc, args[0], args[1]);
            }
            b.create<linalg::YieldOp>(loc, result);
        });

    return genericOp.getResult(0);
}
```

The lambda defines the scalar computation: given elements from lhs and rhs, compute the result and yield it. The linalg infrastructure will generate loops that:
1. Iterate over all indices in parallel
2. Load elements from lhs and rhs at each index
3. Call the scalar computation
4. Store the result to the output

This high-level specification is much shorter than explicit loop construction, yet produces equally efficient (or better) code after lowering and optimization.

**Key Advantages**

Compared to manual SCF loops, `linalg.generic`:
- Expresses intent directly (element-wise operation with this scalar function)
- Enables high-level optimization (fusion, tiling, distribution)
- Handles arbitrary ranks automatically (same code for 1D, 2D, 3D, etc.)
- Simplifies type inference (tensor types flow through naturally)
- Separates iteration structure from computation (easier to understand and modify)

## 7.10 Matrix Multiplication with C++ APIs

Matrix multiplication demonstrates using Linalg dialect operations from C++, showing how high-level structured operations work with tensor types.

**Function Signature**

```cpp
Value buildMatMul(OpBuilder& builder, Location loc, Value lhs, Value rhs) {
    auto lhsType = lhs.getType().cast<RankedTensorType>();
    auto rhsType = rhs.getType().cast<RankedTensorType>();
    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();
```

We work with `RankedTensorType` instead of memrefs—tensors represent immutable values, perfect for high-level IR before bufferization converts them to mutable memory.

**Determine Result Shape**

For matrix multiply, if left is M×K and right is K×N, result is M×N. We handle both static and dynamic dimensions by checking if the input dimensions are dynamic:

```cpp
    // Determine result type shape - use static if both inputs are static
    SmallVector<int64_t> resultShape;
    if (lhsType.isDynamicDim(0)) {
        resultShape.push_back(ShapedType::kDynamic);
    } else {
        resultShape.push_back(lhsType.getDimSize(0));
    }
    // ... same for second dimension ...

    auto resultType = RankedTensorType::get(resultShape, builder.getF32Type());
```

**Initialize Output Tensor**

Matrix multiply accumulates, so we need zero initialization. We use `tensor.empty` (using runtime dimensions if needed) and `linalg.fill`:

```cpp
    // Create empty output tensor with potentially static sizes
    SmallVector<OpFoldResult> outputSizes;
    outputSizes.push_back(lhsType.isDynamicDim(0) ? 
        OpFoldResult(m.getDefiningOp()->getResult(0)) :
        builder.getIndexAttr(lhsType.getDimSize(0)));
    // ... same for second dimension ...

    Value empty = builder.create<tensor::EmptyOp>(loc, outputSizes, builder.getF32Type());

    // Initialize to zero
    Value zero = builder.create<arith::ConstantOp>(loc, builder.getF32Type(),
                                                   builder.getF32FloatAttr(0.0));
    Value init = builder.create<linalg::FillOp>(loc, zero, empty).getResult(0);
```

`linalg.fill` takes a scalar value and a destination tensor, returning a new tensor with all elements set to that scalar. This is pure (functional) style—no mutation, just value transformations.

**Create linalg.matmul Operation**

With inputs and zero-initialized output ready:

```cpp
    // Perform matrix multiplication using linalg.matmul
    auto matmulOp = builder.create<linalg::MatmulOp>(
        loc, resultType,
        ValueRange{lhs, rhs},
        ValueRange{init}
    );

    return matmulOp.getResult(0);
}
```

The operation computes C = A × B as: for each output element C[i,j], sum over k of A[i,k] × B[k,j]. The `outputs` parameter is the accumulator—the operation conceptually does `output += lhs @ rhs`. Since we passed zeros, we get pure multiplication.

**Type Inference and Static Shapes**

A subtle but important detail: when creating the result shape, we use **static dimensions** when possible. If the input shapes are known at compile time (e.g., `tensor<2x3xf32>` and `tensor<3x4xf32>`), the result shape is `tensor<2x4xf32>` with static dimensions. This enables better optimization—the compiler can unroll loops, vectorize more aggressively, and eliminate dynamic size queries.

If inputs have dynamic dimensions (e.g., `tensor<?x3xf32>`), we propagate that: the result is `tensor<?x4xf32>`. The lowering infrastructure handles both cases correctly, but static shapes enable more optimization. In our implementation, we preserve static dimensions from inputs whenever possible.

**Lowering Path**

After graph generation, the IR goes through bufferization:

1. **OneShotBufferize**: Converts tensors to memrefs, inserting allocations and copies as needed
2. **BufferResultsToOutParams**: Transforms return values into output parameters
3. **ConvertLinalgToLoops**: Expands linalg.matmul into three nested SCF loops
4. Further lowering to LLVM

At each level, the representation becomes more concrete: tensors → buffers → loops → machine code. But at the graph construction level, we think purely in terms of mathematical operations on immutable tensors, which is both simpler and more optimizable.

## 7.11 Composing Operations: Building a Neural Network

With individual operations implemented, we can compose them into larger computations. Let's build a two-layer neural network (also called a multi-layer perceptron or MLP) to demonstrate how operations compose naturally through the graph API. A two-layer network computes:

```
h = ReLU(x × W1)
y = h × W2
```

where `x` is the input, `W1` and `W2` are weight matrices, and `×` denotes matrix multiplication. This simple architecture appears throughout machine learning: as the building block of feed-forward networks, as the "MLP" component in transformers (after attention), and as a general-purpose function approximator.

**Building the Graph**

```cpp
// Create graph
ComputationGraph g(context);

// Define variables (placeholders for data)
int x_id = g.addVariable({2, 3});    // Input: 2 samples, 3 features each
int W1_id = g.addVariable({3, 4});   // Layer 1 weights
int W2_id = g.addVariable({4, 2});   // Layer 2 weights

// Layer 1: x × W1
int h_id = g.matmul(x_id, W1_id);    // → [2, 4]

// Activation: ReLU(h)
int h_relu_id = g.relu(h_id);

// Layer 2: h_relu × W2
int y_id = g.matmul(h_relu_id, W2_id);  // → [2, 2]

// Generate MLIR for the entire computation
ModuleOp module = g.generateMLIR(y_id, "mlp");
```

Each line returns an operation ID, which subsequent operations reference. The graph structure is implicit in these references: operation `h_id` depends on `x_id` and `W1_id`, `h_relu_id` depends on `h_id`, and `y_id` depends on `h_relu_id` and `W2_id`. When we call `generateMLIR(y_id, "mlp")`, the recursive builder walks this dependency graph, generating all necessary operations in topological order.

**Generated Function Signature**

The generated MLIR function signature follows the functional style:

```mlir
func.func @mlp(%arg0: tensor<2x3xf32>, 
               %arg1: tensor<3x4xf32>, 
               %arg2: tensor<4x2xf32>) -> tensor<2x2xf32>
```

Three input tensors (x, W1, W2) and one tensor return value. The function body contains high-level linalg operations that express the computation at a semantic level. Bufferization later converts this to imperative code with explicit memory management.

**Execution Flow**

From Python, we call:

```python
g = ch7.Graph()
x = g.variable([2, 3])
W1 = g.variable([3, 4])
W2 = g.variable([4, 2])
h = g.matmul(x, W1)
h_relu = g.relu(h)
y = g.matmul(h_relu, W2)

# Compile once
fn = g.compile(y, "mlp")

# Execute with actual data
x_data = np.random.randn(2, 3).astype(np.float32)
W1_data = np.random.randn(3, 4).astype(np.float32)
W2_data = np.random.randn(4, 2).astype(np.float32)
result = ch7.execute_generic(fn, [x_data, W1_data, W2_data], (2, 2))
```

The graph construction happens once, producing a JIT-compiled function. We can then execute that function repeatedly with different input data, amortizing compilation cost over many executions. This is the performance model of JAX, TensorFlow, and PyTorch's JIT modes: pay compilation cost once, then execute efficiently many times.

**Optimization Opportunities**

Even this simple example has optimization potential. The matmul producing `h` writes to memory, then ReLU reads from memory to write `h_relu`. An optimizer could fuse these operations: during the matmul's final write loop, apply ReLU immediately and write the activated value to `h_relu` directly, eliminating the intermediate `h` buffer entirely. This "operation fusion" is one of the most important optimizations in deep learning compilers, and it's only possible when the compiler sees both operations together—precisely what our graph provides. Chapter 10 will explore such optimizations.

## 7.12 Python API and Generic Execution

The Python API provides a clean interface to graph construction and execution, hiding C++ details while exposing full functionality. Let's examine the key components and how they enable flexible, type-safe execution.

**Graph Construction API**

The `Graph` class wraps `ComputationGraph` in Python:

```python
import ch7_neural_ops as ch7

g = ch7.Graph()
x = g.variable([4])           # Returns operation ID (int)
y = g.variable([4])
z = g.add(x, y)               # Returns operation ID for the sum
fn = g.compile(z, "add_func") # JIT compiles to executable function
```

Each method (`variable`, `add`, `mul`, `matmul`, `relu`, `softmax`) returns an integer operation ID. These IDs are opaque to Python—just handles identifying graph nodes. Only `compile()` actually generates MLIR and produces executable code. The `get_mlir()` method allows inspection of generated IR before compilation, useful for debugging and learning.

## 7.13 The Binding Explosion Problem and libffi

Before we can discuss execution, we must confront a fundamental challenge that arises from MLIR's memref calling convention. This problem is subtle enough that many MLIR tutorials skip it entirely, but any practical ML framework built on MLIR will hit it immediately. Understanding this problem—and the elegant solution via libffi—is essential for implementing robust graph execution systems.

**The N×M Binding Explosion**

Recall from Chapter 2 that MLIR functions using memrefs have **shape-specific signatures**. A function operating on 1D memrefs looks like:

```cpp
void add_1d(float* base, float* aligned, int64_t offset, int64_t size, int64_t stride, ...)
```

While the same operation on 2D memrefs has different arity:

```cpp
void add_2d(float* base, float* aligned, int64_t offset,
            int64_t size0, int64_t size1, int64_t stride0, int64_t stride1, ...)
```

Every rank needs different marshaling code. Now consider that we have N operation types (add, mul, matmul, relu, softmax, ...) and M common shapes (1D, 2D, 3D, maybe specialized shapes like 2×3×4 for specific networks). To execute any operation with any shape, we'd need **N×M wrapper functions** in our Python bindings. For just 10 operation types and 20 shapes, that's 200 hand-written wrapper functions!

This is the binding explosion problem. Each wrapper function would look nearly identical—take Python inputs, marshal to memref descriptor, call JIT function, return result—differing only in the exact number and interpretation of descriptor arguments. Writing and maintaining these would be tedious, error-prone, and completely impractical as we add new operations or shapes. Every new operation type adds M wrappers; every new shape adds N wrappers. The cross product grows uncontrollably.

Traditional solutions include:
- **Code generation**: Write a script that generates all N×M wrappers. This works but adds build complexity and makes debugging harder (stack traces go through generated code).
- **Preprocessor macros**: Use C preprocessor to stamp out variants. Error messages become incomprehensible, and the approach doesn't scale beyond simple cases.
- **Template metaprogramming**: Use C++ templates to generate wrappers. This works in theory but gets very complex very fast, especially with pybind11's type system.

All these approaches miss the fundamental issue: we don't need N×M **different** functions; we need **one generic function** that can handle arbitrary memref signatures. This is precisely what libffi provides.

**What is libffi?**

libffi (Foreign Function Interface library) is a portable library for dynamically calling C functions **without knowing their signature at compile time**. Given a function pointer and a description of its arguments (types, count), libffi constructs the appropriate call at runtime, handling calling convention details (register allocation, stack layout, return value handling) automatically. It's the foundation of many language runtimes (Python's ctypes, Ruby's FFI, Java's JNA) and JIT compilers.

For our purposes, libffi solves the memref calling problem elegantly: we describe the function's signature **once** at runtime (based on the actual input shapes), then call the JIT-compiled function through libffi. One `execute_generic()` function handles all shapes and all operation types. No code generation, no macros, no template wizardry—just runtime introspection and dynamic calling.

**How execute_generic() Works with libffi**

Let's walk through the complete implementation in [bindings.cpp](../ch.7.Neural-ops/src/bindings.cpp):

```cpp
py::array_t<float> execute_generic(intptr_t fnPtr,
                                   const std::vector<py::array_t<float>>& inputs,
                                   const std::vector<int64_t>& output_shape)
{
    // 1. Prepare output array
    py::array_t<float> output(output_shape);

    // 2. Build argument list for the function call
    std::vector<intptr_t> args;

    // Marshal output memref
    if (output_shape.size() == 1) {
        marshal_memref_1d(output, args);
    } else if (output_shape.size() == 2) {
        marshal_memref_2d(output, args);
    }

    // Marshal input memrefs
    for (const auto& input : inputs) {
        if (input.ndim() == 1) {
            marshal_memref_1d(input, args);
        } else if (input.ndim() == 2) {
            marshal_memref_2d(input, args);
        }
    }

    // 3. Setup libffi call interface
    size_t num_args = args.size();
    std::vector<ffi_type*> arg_types(num_args, &ffi_type_pointer);
    std::vector<void*> arg_values(num_args);

    for (size_t i = 0; i < num_args; ++i) {
        arg_values[i] = &args[i];  // Address of each intptr_t
    }

    ffi_cif cif;
    ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, num_args,
                                     &ffi_type_void, arg_types.data());
    if (status != FFI_OK) {
        throw std::runtime_error("ffi_prep_cif failed");
    }

    // 4. Make the call!
    ffi_call(&cif, FFI_FN(fnPtr), nullptr, arg_values.data());

    return output;
}
```

Let's break down each step:

**Step 1: Prepare Output Array**. NumPy handles allocation for us—we just pass the shape and get back a contiguous array. This will be the first argument to our MLIR function (output parameter).

**Step 2: Marshal Arguments**. The `marshal_memref_1d` and `marshal_memref_2d` helper functions extract descriptor values from NumPy arrays and append them to the `args` vector:

```cpp
void marshal_memref_1d(const py::array_t<float>& arr, std::vector<intptr_t>& args) {
    args.push_back(reinterpret_cast<intptr_t>(arr.data()));  // base
    args.push_back(reinterpret_cast<intptr_t>(arr.data()));  // aligned
    args.push_back(0);                                        // offset
    args.push_back(arr.shape(0));                             // size
    args.push_back(arr.strides(0) / sizeof(float));          // stride
}
```

Each memref becomes 5 (for 1D) or 7 (for 2D) `intptr_t` values. By the end of this step, `args` contains all descriptor values flattened into a single vector—this is the complete argument list for the function.

**Step 3: Describe the Call to libffi**. We need to tell libffi three things:
- **Calling convention**: `FFI_DEFAULT_ABI` uses the platform's standard C calling convention (cdecl on x86, SysV on x86-64, AAPCS on ARM, etc.).
- **Number of arguments**: `num_args` is the size of our `args` vector—could be 7 (one 1D input, one 1D output) or 21 (two 2D inputs, one 2D output) or anything.
- **Argument types**: Every argument is a pointer (or integer castable to pointer). We set all argument types to `&ffi_type_pointer`, which tells libffi each argument is pointer-sized and should be passed in registers/stack as a pointer.

The `arg_values` vector contains **addresses** of our arguments. This double indirection is libffi's interface: you pass an array of `void*`, each pointing to an argument value. For pointer/integer arguments, we point to the `intptr_t` in our `args` vector.

`ffi_prep_cif()` (CIF = Call Interface) prepares the call description. This is relatively cheap—just setting up some metadata about register allocation and stack layout.

**Step 4: Execute the Call**. `ffi_call()` is where the magic happens. It:
- Allocates a small stack frame
- Loads arguments from `arg_values` into appropriate registers or stack locations according to the calling convention
- Jumps to the function at `fnPtr`
- The JIT-compiled MLIR function executes, reading inputs and writing outputs to the memref descriptors
- Returns control to `ffi_call`
- We return the output NumPy array to Python

All this happens transparently—from the MLIR function's perspective, it was called normally with standard C ABI. libffi handled the dynamic dispatch entirely.

**Why This Works**

The key insight: **MLIR's memref calling convention uses only pointer-sized values**. Base pointers, aligned pointers, offsets, sizes, strides—all are either `float*` or `int64_t`, both pointer-sized on 64-bit platforms. So every argument can be described to libffi as `ffi_type_pointer`. We don't need complex type descriptions; everything is uniformly pointer-sized, which libffi handles trivially.

This uniformity is no accident. MLIR's design deliberately keeps lowered representations simple and C-ABI-compatible, enabling exactly this kind of generic interfacing. If memref descriptors used complex structures or non-standard layouts, libffi couldn't help—we'd need shape-specific wrappers. But because MLIR flattens descriptors to scalars, one libffi-based function handles everything.

**Comparison: With and Without libffi**

**Without libffi** (static bindings):

```cpp
// Would need all of these:
py::array_t<float> execute_add_1d(...) { /* shape-specific code */ }
py::array_t<float> execute_add_2d(...) { /* shape-specific code */ }
py::array_t<float> execute_mul_1d(...) { /* shape-specific code */ }
py::array_t<float> execute_mul_2d(...) { /* shape-specific code */ }
py::array_t<float> execute_matmul_2d(...) { /* shape-specific code */ }
// ... 200+ functions for 10 ops × 20 shapes
```

Each would duplicate the marshal-call-return pattern. Every new operation or shape requires new bindings. Maintenance nightmare.

**With libffi** (dynamic calling):

```cpp
py::array_t<float> execute_generic(intptr_t fnPtr,
                                   const std::vector<py::array_t<float>>& inputs,
                                   const std::vector<int64_t>& output_shape)
{
    // One function handles everything!
    // Introspects shapes at runtime, marshals dynamically, calls via libffi
}
```

One function. Works for any operation. Works for any shape. Adding new operations or shapes requires no changes to execution code—the graph compiler generates MLIR, JIT produces a function pointer, `execute_generic` calls it. This is the scalability we need for a real ML framework.

**Performance Considerations**

Does the dynamic dispatch add overhead? In practice, **negligible**. The cost of `ffi_prep_cif` and `ffi_call` is a few dozen instructions—on the order of a function call. For ML operations computing on large tensors (thousands or millions of elements), the function call overhead (even with libffi) is unmeasurable compared to the actual computation. A 1000×1000 matrix multiply does a billion floating-point operations; spending a few hundred cycles on libffi dispatch is 0.00001% overhead.

If we were calling tiny operations (adding two scalars) millions of times in a tight loop, libffi's overhead might matter. But ML workloads batch computations into large operations precisely to amortize overheads. The graph execution model we've built—compile once, execute with large tensors—makes libffi's cost invisible in real workloads.

## 7.14 Generic Execution Function

With libffi handling dynamic dispatch, the execution interface becomes simple:

The function takes the compiled function pointer (returned by `compile()`), a list of input NumPy arrays, and the expected output shape. It:

1. Allocates the output array with the given shape
2. Marshals the output memref descriptor (first argument to MLIR functions)
3. Marshals each input memref descriptor in order
4. Uses libffi to call the function with all descriptor values
5. Returns the output array to Python

This generic approach eliminates shape-specific wrappers entirely. Whether calling a function with three 1D inputs or two 2D inputs, the same `execute_generic()` handles it. The libffi-based implementation introspects shapes at runtime and builds the appropriate call dynamically.

**Memref Convention Reminder**

From Chapter 2, recall that MLIR memrefs expand to multiple scalar arguments. For 1D `memref<4xf32>`:

```
func(float* base, float* aligned, int64_t offset, int64_t size, int64_t stride)
```

Our NumPy array provides: `data` (base/aligned pointers), offset=0, size=4, stride=1. For 2D `memref<2x3xf32>`, seven arguments: base, aligned, offset, size0, size1, stride0, stride1. The stride computation divides NumPy strides by element size and handles dimension ordering correctly—MLIR orders strides from outermost to innermost dimension.

**Testing Example**

A complete test demonstrates the workflow:

```python
def test_matmul():
    g = ch7.Graph()
    A = g.variable([2, 3])
    B = g.variable([3, 4])
    C = g.matmul(A, B)

    print(g.get_mlir(C, "matmul"))  # Inspect generated IR
    fn = g.compile(C, "matmul")      # Compile

    A_data = np.random.randn(2, 3).astype(np.float32)
    B_data = np.random.randn(3, 4).astype(np.float32)
    result = ch7.execute_generic(fn, [A_data, B_data], (2, 4))

    expected = A_data @ B_data
    assert np.allclose(result, expected)
    print("✓ MatMul test passed")
```

The test builds a graph, inspects IR (optional, useful for learning), compiles, executes with random data, and verifies against NumPy. This pattern—build graph, compile, execute, verify—applies to all operations. The compilation happens once; we can call `execute_generic(fn, ...)` repeatedly with different input data, amortizing compilation cost.

## 7.15 Looking Ahead

With computation graphs implemented, we can build complex models from simple operations, but we're still generating all operations explicitly into a single flat function. Chapter 8 will introduce **custom dialects**, allowing us to define high-level operations like `attention` or `feedforward` that encapsulate entire sub-networks. Instead of generating 50 operations for an attention mechanism, we'll generate one `transformer.attention` operation, then lower it to explicit operations later. This abstraction enables optimizations that reason about semantics (like "this attention operation can be fused with layer norm") rather than just syntax (like "these loops can be merged").

Chapter 9 will show how to define dialects using **TableGen**, MLIR's declarative specification language. Rather than writing C++ classes for every operation and type, we'll specify them in a high-level description language, and code generation produces the necessary C++ automatically. This is how MLIR's built-in dialects are implemented, and it's the approach you'll use for production dialect development.

Chapter 10 returns to our neural operations with **optimization passes**, showing how to implement transformations that improve performance: operation fusion, memory reuse, algebraic simplifications, and more. We'll see how the graph structure we built in this chapter enables these optimizations—the compiler can analyze data flow, identify patterns, and apply transformations that would be impossible with standalone operations.

Chapter 11 implements the **attention mechanism**, the core of transformer models. Attention combines many of the operations we've learned (matmul, softmax, element-wise multiplication) into a complex computation with interesting optimization opportunities. We'll see how MLIR's multi-level representation enables expressing and optimizing attention computation through progressive lowering and transformation passes.

The journey from standalone operations (Chapter 6) to compositional graphs (Chapter 7) to custom dialects (Chapters 8-9) to optimized transformations (Chapter 10) to production transformers (Chapter 11+) follows the path of building a real AI compiler. Each chapter adds capabilities while reusing previous concepts. The computation graph is the foundation: it structures our thinking about operations, dependencies, and optimization opportunities. Everything from here builds on that foundation.