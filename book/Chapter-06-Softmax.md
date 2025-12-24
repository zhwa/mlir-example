# Chapter 6: Mathematical Operations — Implementing Softmax

In the previous five chapters, we built a solid foundation in MLIR's fundamental concepts: from basic matrix operations with the Linalg dialect in Chapter 1, through dynamic shapes and bufferization in Chapters 2 and 4, to explicit control flow with SCF and Arith dialects in Chapter 5. We've learned how to work with tensors and memrefs, how to manage compilation infrastructure, and how to build explicit loops with arithmetic operations. Now we're ready to take the next step: implementing our first real machine learning operation.

This chapter introduces the **Math dialect**, which provides mathematical functions like exponential, logarithm, trigonometric operations, and other transcendental functions commonly used in scientific computing and machine learning. We'll use the Math dialect to implement **softmax**, a fundamental activation function that appears everywhere in modern neural networks—particularly in transformers, where it powers the attention mechanism that we'll explore in Chapter 11. Unlike the simple element-wise operations we've built so far (like the SAXPY example in Chapter 5), softmax requires a multi-pass algorithm with reductions and introduces important concerns about numerical stability that every ML practitioner must understand.

The implementation in this chapter will demonstrate several new patterns: using loop-carried variables with `scf.for` to accumulate results across iterations (essential for computing maximum values and sums), employing a three-pass algorithm where each pass depends on results from the previous one, and applying numerical stability techniques to prevent floating-point overflow. We'll also explore how Math dialect operations lower to standard C library calls through the `math-to-libm` pass, connecting MLIR's high-level mathematical abstractions to the battle-tested implementations in `libm`.

## 6.1 The Math Dialect

The Math dialect complements the Arith dialect we introduced in Chapter 5 by providing operations for mathematical functions beyond basic arithmetic. While Arith handles addition, multiplication, comparisons, and other elementary operations with precise semantics for integers and floating-point numbers, Math provides transcendental functions—operations whose results cannot be expressed as finite combinations of basic arithmetic operations.

The Math dialect includes operations like `math.exp` (exponential function $e^x$), `math.log` (natural logarithm), `math.sqrt` (square root), `math.sin` and `math.cos` (trigonometric functions), `math.pow` (power function), and many others. These operations maintain the same type system as Arith, supporting various floating-point precisions (f32, f64, f16) and integer types where applicable. The dialect is specifically designed for numerical computing workloads common in scientific applications and machine learning.

What makes Math dialect operations particularly useful is their flexibility in lowering strategies. Unlike Arith operations which typically lower directly to LLVM's built-in arithmetic instructions, Math operations can take multiple paths to native code. The `math-to-libm` pass converts Math operations into calls to the standard C math library (`libm`), leveraging decades of work on accurate and efficient implementations of mathematical functions. For example, `math.exp %x : f32` becomes a call to the C library's `expf()` function. This approach provides excellent accuracy and handles edge cases correctly—important considerations when dealing with numerical algorithms.

Alternatively, the `math-to-llvm` pass can generate inline polynomial approximations or use LLVM intrinsics, trading some accuracy for reduced function call overhead. This flexibility allows developers to choose between accuracy and performance based on their application's requirements. For our purposes in this chapter, we'll use the `math-to-libm` path because it provides the accuracy needed for correct softmax implementation and offers a clear illustration of how high-level mathematical operations connect to standard library implementations.

In our softmax implementation, we'll primarily use `math.exp` to compute exponentials. The exponential function is central to many machine learning algorithms: it appears in softmax (our focus here), in the cross-entropy loss function, in sigmoid activations, in attention mechanisms, and in probabilistic models throughout deep learning. Understanding how to work with exponentials and handle their numerical properties is essential for implementing ML systems correctly.

## 6.2 The Softmax Function

Softmax is a fundamental operation in machine learning that converts a vector of arbitrary real numbers into a probability distribution. Given an input vector $\mathbf{x} = [x_1, x_2, \ldots, x_n]$, softmax produces an output vector $\mathbf{y} = [y_1, y_2, \ldots, y_n]$ where each element is computed as:

$$
y_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

Each output value $y_i$ is positive (since exponentials are always positive), and the sum of all outputs equals 1, satisfying the properties of a probability distribution: $\sum_{i=1}^{n} y_i = 1$ and $0 < y_i < 1$ for all $i$. This transformation is particularly useful in classification tasks where we need to interpret neural network outputs as probabilities over different classes.

The softmax function has several important mathematical properties. It's translation-invariant, meaning that adding a constant to all inputs doesn't change the output: $\text{softmax}(\mathbf{x} + c) = \text{softmax}(\mathbf{x})$ for any constant $c$. This property, which we can verify by algebraic manipulation, turns out to be crucial for numerical stability, as we'll see shortly. Softmax also amplifies differences between inputs—larger input values receive exponentially more weight in the output, which is why it's called "softmax" rather than just "max". Unlike the hard maximum function that outputs 1 for the largest element and 0 for all others, softmax provides a differentiable alternative that assigns probabilities based on relative magnitudes.

In modern deep learning, softmax appears in multiple contexts. Most visibly, it's typically the final layer in classification networks, converting logits (raw network outputs) into class probabilities. In transformer architectures like GPT and BERT, softmax is applied to attention scores in every attention layer, determining how much each token should attend to every other token—a pattern we'll explore in detail in Chapter 11. It also appears in various loss functions, optimization algorithms, and architectural components throughout neural networks.

However, the naive implementation shown in the equation above has a critical problem: for large input values, computing $e^{x_i}$ can cause floating-point overflow. Since the exponential function grows extremely rapidly, even moderately large inputs can produce results that exceed the maximum representable floating-point value. For example, in 32-bit floating-point arithmetic (which we use throughout this book), `exp(89)` already overflows to infinity. Once overflow occurs, the softmax computation becomes meaningless, producing `nan` (not a number) values when dividing infinity by infinity.

## 6.3 Numerical Stability Through Max Subtraction

The solution to overflow in softmax leverages the translation-invariance property we mentioned earlier. We can subtract any constant from all input values without changing the final result, so we choose to subtract the maximum input value before computing exponentials. This **max subtraction technique** is the standard approach for numerically stable softmax implementation.

Mathematically, we reformulate the computation as:

$$
y_i = \frac{e^{x_i - \max(\mathbf{x})}}{\sum_{j=1}^{n} e^{x_j - \max(\mathbf{x})}}
$$

This formulation is mathematically equivalent to the original (thanks to translation invariance), but the numerical behavior is drastically different. After subtracting the maximum, all input values become zero or negative. The largest value becomes $\max(\mathbf{x}) - \max(\mathbf{x}) = 0$, and all others are negative. Since $e^0 = 1$ and exponentials of negative numbers are less than 1, the largest exponential in our computation is exactly 1, completely eliminating the possibility of overflow.

Underflow—when exponentials of very negative numbers become too small to represent—can still occur, but this is harmless. Values that underflow to zero simply contribute nothing to the sum in the denominator and produce zero probabilities in the output, which is the correct mathematical result for inputs that are much smaller than the maximum. The numerical stability achieved through max subtraction is so important that every production implementation of softmax uses this technique, from NumPy to PyTorch to TensorFlow.

This stability concern illustrates a broader principle in numerical computing: mathematically equivalent formulations can have vastly different numerical properties when implemented with finite-precision arithmetic. Throughout this book, we'll encounter other examples where algorithm design must account for floating-point arithmetic's limitations. The max subtraction technique is particularly elegant because it requires minimal additional computation (one pass to find the maximum) and completely solves the overflow problem without approximations or loss of accuracy.

## 6.4 Three-Pass Algorithm Structure

Implementing numerically stable softmax requires breaking the computation into three sequential passes over the input data. This multi-pass structure differs from the single-pass element-wise operations we've seen in previous chapters, and it demonstrates an important algorithmic pattern: some computations require global information (like the maximum value) before per-element processing can proceed safely.

The **first pass** computes the maximum value of the input vector. We must scan the entire input before we can begin computing exponentials, since we need the maximum to ensure numerical stability. This pass implements a reduction operation—combining many values into a single result—using a loop-carried variable that maintains the current maximum as we iterate through the input. We initialize the maximum to negative infinity (ensuring any real value will be larger) and update it whenever we encounter a larger value.

The **second pass** performs two operations simultaneously: computing exponentials of the shifted values ($e^{x_i - \max}$) and accumulating their sum. We store the exponential values in a temporary buffer because we'll need them again in the third pass. This pass also implements a reduction (summing the exponentials) alongside the element-wise computation and storage. By combining these operations in a single pass, we minimize memory traffic—an important consideration for performance, though our focus in this book is on correctness and clarity rather than optimization.

The **third pass** normalizes the exponentials by dividing each by the sum computed in the second pass. This final pass reads from the temporary buffer and writes the normalized probabilities to the output. Unlike the previous two passes, this one doesn't require a reduction; it's a straightforward element-wise operation. After this pass completes, our output contains valid probabilities that sum to exactly 1 (within floating-point precision).

This three-pass structure might seem inefficient compared to a hypothetical single-pass implementation, but it's necessary for correctness. We cannot compute the maximum and exponentials simultaneously because we don't know the maximum until we've seen all values. We cannot normalize while computing exponentials because we don't know the sum until we've computed all exponentials. The algorithm's structure directly reflects these data dependencies, and attempting to optimize away the multiple passes would require complex buffering strategies that ultimately perform the same amount of work while being much harder to understand and verify.

The temporary buffer adds a memory allocation to our implementation, but for vectors of size $n$, we only need $O(n)$ additional space—linear in the input size, not quadratic or higher. This is considered efficient in algorithm analysis, and in practice, the temporary buffer typically fits in CPU cache for the vector sizes common in ML workloads. When we move to more complex operations in later chapters, we'll see similar patterns where intermediate results must be stored for subsequent processing.

## 6.5 Loop-Carried Variables and Reductions

To implement the reduction operations needed in passes 1 and 2, we use **loop-carried variables** with `scf.for`, a pattern we didn't need in Chapter 5's SAXPY implementation. Loop-carried variables allow values to flow from one iteration to the next, enabling accumulation patterns like finding maximum values, computing sums, or building up results incrementally. This mechanism preserves SSA form while expressing inherently stateful computations—a key challenge in compiler IR design.

The `scf.for` operation supports loop-carried variables through its `iter_args` mechanism. When creating the loop, we specify initial values for variables that will be updated across iterations. Inside the loop body, these values are available as block arguments (similar to function parameters), and we use `scf.yield` to pass potentially updated values to the next iteration or return final values when the loop completes. The values yielded from the last iteration become the results of the `scf.for` operation itself, allowing subsequent code to use the accumulated result.

For example, to find the maximum value in our first pass, we create an `scf.for` loop with a single loop-carried variable initialized to negative infinity. In each iteration, we load a value from the input, compare it with the current maximum using `arith.maximumf` (which correctly handles IEEE 754 special cases like NaN), and yield the larger value. The loop's result is the maximum value, which we can then use in subsequent passes. This pattern is concise and maintains SSA form throughout—there's no mutable variable being repeatedly assigned in the traditional imperative programming sense.

Similarly, in the second pass, we use a loop-carried variable initialized to zero to accumulate the sum of exponentials. Each iteration loads an input value, subtracts the maximum (computed in pass 1), computes the exponential using `math.exp`, stores it to the temporary buffer, adds it to the current sum, and yields the updated sum. The loop's result is the total sum, which we use for normalization in pass 3. This demonstrates that loop-carried variables can be used alongside other operations in the loop body—the pattern is flexible enough to express complex iteration logic.

In contrast, the third pass doesn't need loop-carried variables because it only performs element-wise operations without accumulation. Each iteration loads from the temporary buffer, divides by the sum, and stores to the output—these operations don't depend on previous iterations. We create the `scf.for` loop without `iter_args`, and the loop body simply performs its operations and yields nothing. This shows that the same `scf.for` construct can express both reduction patterns (with `iter_args`) and simple iteration (without).

## 6.6 Lambda-Style Loop Construction

In Chapter 5, we constructed `scf.for` loops using a multi-step process: create the loop operation, set the insertion point inside its body, build the body operations, and finally restore the insertion point. This approach works well for simple loops, but it becomes cumbersome when working with loop-carried variables, where we need to carefully manage block arguments and yield operations. For loops with `iter_args`, MLIR provides a more concise **lambda-style construction** that we use throughout this chapter.

The lambda-style approach passes a callback function to the `scf::ForOp` constructor, and this callback receives four parameters: an `OpBuilder` for constructing operations inside the loop, a `Location` for debugging information, the induction variable (loop counter), and a `ValueRange` containing the current values of loop-carried variables. The callback builds the loop body and must call `scf.yield` to pass values to the next iteration. The loop operation automatically manages the block structure, insertion points, and block arguments—details that required manual handling in the multi-step approach.

Consider the first pass in our softmax implementation, which finds the maximum value. Using the lambda style, we write:

```cpp
auto findMaxLoop = builder.create<scf::ForOp>(
    loc, c0, size, c1, ValueRange{negInf},
    [&](OpBuilder& b, Location loc, Value i, ValueRange iterArgs) {
        Value currentMax = iterArgs[0];
        Value val = b.create<memref::LoadOp>(loc, input, ValueRange{i});
        Value newMax = b.create<arith::MaximumFOp>(loc, currentMax, val);
        b.create<scf::YieldOp>(loc, ValueRange{newMax});
    }
);
Value maxVal = findMaxLoop.getResult(0);
```

The constructor parameters specify the loop bounds (`c0` to `size`, stepping by `c1`) and initial values for loop-carried variables (`ValueRange{negInf}` provides the initial maximum). The lambda captures our outer scope by reference (`[&]`), allowing access to variables like `input` and other builders. Inside the lambda, we access the current maximum as `iterArgs[0]`, build operations using the provided builder `b`, and yield the updated maximum. After the loop completes, we retrieve the final result using `getResult(0)`.

This style is particularly clean for multiple loop-carried variables. In the second pass, we have a single loop-carried variable (the accumulating sum), but the pattern naturally extends to multiple variables by providing multiple initial values and yielding multiple results. The lambda syntax makes the data flow explicit: initial values flow in as `iter_args`, and updated values flow out through `yield`. There's no need to manually create the yield operation's block structure or carefully match types between iterations.

Compared to the multi-step approach from Chapter 5, the lambda style has several advantages for loops with `iter_args`. It's more concise, reducing boilerplate code for insertion point management. It's less error-prone because the framework handles block argument creation and type consistency automatically. It makes the code more functional in style—the loop body is a transformation from current state to next state, clearly expressed as a function. For these reasons, when working with reductions or other patterns requiring loop-carried variables, the lambda style is strongly preferred and is what we'll use throughout the remainder of this book.

However, for simple loops without `iter_args`, either style works equally well, and the multi-step approach can sometimes be clearer when the loop body is complex or requires its own internal control flow. The choice between styles is a matter of code organization and readability rather than functional capability—both ultimately generate identical MLIR IR.

## 6.7 Building Softmax with C++ APIs

Now let's see how to construct the softmax implementation using MLIR's C++ APIs. Understanding this code is essential for building your own MLIR transformations and operations, as it demonstrates the patterns you'll use throughout MLIR development. We'll walk through the complete implementation step by step, showing how each C++ API call creates the corresponding MLIR operation.

**Setting Up the Function and Context**

We begin by creating the module and function structure:

```cpp
OwningOpRef<ModuleOp> createSoftmaxModule(MLIRContext& context) {
  // Load required dialects
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();
  context.getOrLoadDialect<math::MathDialect>();

  // Create builder and module
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module.getBody());
```

The function begins by loading all required dialects into the MLIR context. Each dialect must be explicitly loaded before we can create operations from it—attempting to use operations from an unloaded dialect will fail at runtime. The `getOrLoadDialect<>` template method is idempotent: if the dialect is already loaded, it does nothing; otherwise, it loads and initializes the dialect. This pattern ensures dialects are available regardless of which order modules are initialized.

We create an `OpBuilder`, the primary interface for constructing MLIR operations. The builder maintains an insertion point—a location in the IR where new operations will be added. Initially, we create a module and set the insertion point to its body, meaning subsequent operations will be added as top-level module contents (in our case, the function definition).

The `Location` object (`loc`) represents source code location information for debugging. Using `getUnknownLoc()` indicates we're generating IR programmatically without corresponding source code. In a compiler that parses source files, you would create locations that point to specific lines and columns, enabling error messages to reference the original source.

**Creating the Function Signature**

```cpp
  // Define types
  auto f32Type = builder.getF32Type();
  auto dynamicMemRefType = MemRefType::get({ShapedType::kDynamic}, f32Type);

  // Function type: (memref<?xf32>, memref<?xf32>) -> ()
  auto funcType = builder.getFunctionType(
    {dynamicMemRefType, dynamicMemRefType},
    {}
  );

  // Create function
  auto funcOp = builder.create<func::FuncOp>(loc, "softmax", funcType);
  funcOp.setPublic();

  // Create function body
  auto& entryBlock = *funcOp.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);
```

Type creation uses the builder's factory methods. The `getF32Type()` method returns MLIR's 32-bit float type, which we use throughout our implementation. For the memref type, we use `MemRefType::get()`, specifying a single dynamic dimension (`ShapedType::kDynamic` is a special constant representing unknown size at compile time) and f32 element type. This creates the type `memref<?xf32>` that we've been using.

The function type is created with `getFunctionType()`, taking two vectors: input types and result types. Our function takes two memref parameters (input and output) and returns nothing (empty result vector), matching the out-parameter pattern from Chapter 4. We then create the actual function operation using `builder.create<func::FuncOp>()`, passing the location, name, and type signature.

The `setPublic()` call makes the function visible outside the module, necessary for the JIT compiler to find and invoke it. By default, MLIR functions are private (internal linkage), but we need public visibility for the ExecutionEngine to look up the function by name.

Creating the function body requires two steps: add an entry block (the first basic block of the function), and set the insertion point to its start. The entry block automatically receives arguments matching the function signature—two memref parameters in our case. We can access these as `entryBlock.getArgument(0)` and `getArgument(1)`.

**Preparing Constants and Buffer Allocation**

```cpp
  // Get function arguments
  Value input = entryBlock.getArgument(0);
  Value output = entryBlock.getArgument(1);

  // Get size of input
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value size = builder.create<memref::DimOp>(loc, input, c0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  // Allocate temporary buffer for exp values
  Value tempBuffer = builder.create<memref::AllocaOp>(
      loc, dynamicMemRefType, ValueRange{size});
```

We retrieve the function arguments as `Value` objects, which represent SSA values in MLIR. Every operation result is a `Value`, and operations take `Value` objects as operands. This is the fundamental abstraction for building the IR—we construct operations that consume and produce values, maintaining SSA form throughout.

Creating constants uses type-specific operations. `arith::ConstantIndexOp` creates index-typed constants (MLIR's dedicated type for array indices and loop bounds), while floating-point constants use `arith::ConstantFloatOp` (which we'll see shortly). The distinction matters because MLIR's type system is precise—you cannot use an f32 value where an index is expected, even if the numerical value would make sense.

The `memref::DimOp` queries the size of a memref dimension at runtime. We pass the input memref and dimension index 0 (the only dimension in our 1D array), and it returns an index value representing the size. This dynamic size query is exactly what we covered in Chapter 2 for handling arbitrary input sizes.

Allocating the temporary buffer uses `memref::AllocaOp`, which allocates stack memory. The operation takes the memref type and dynamic dimension sizes (if any). Since our type has one dynamic dimension, we pass the size as a `ValueRange{size}`. If we had multiple dynamic dimensions, we'd pass multiple sizes. Stack allocation is fast and automatically cleaned up when the function returns—perfect for temporary storage in our case.

**Pass 1: Finding Maximum with Loop-Carried Variables**

```cpp
  // Initialize with negative infinity
  Value negInf = builder.create<arith::ConstantFloatOp>(
      loc, 
      APFloat::getInf(f32Type.getFloatSemantics(), /*Negative=*/true), 
      f32Type);

  // Create loop to find max
  auto findMaxLoop = builder.create<scf::ForOp>(
      loc, c0, size, c1, ValueRange{negInf},
      [&](OpBuilder& b, Location loc, Value i, ValueRange iterArgs) {
        Value currentMax = iterArgs[0];
        Value val = b.create<memref::LoadOp>(loc, input, ValueRange{i});
        Value newMax = b.create<arith::MaximumFOp>(loc, currentMax, val);
        b.create<scf::YieldOp>(loc, ValueRange{newMax});
      }
  );
  Value maxVal = findMaxLoop.getResult(0);
```

Creating floating-point constants is more involved than integer constants because we need to specify the precise bit pattern. The `APFloat` class (arbitrary-precision float) provides methods for creating special values like infinity. We call `APFloat::getInf()` with the float semantics (precision and format information from f32Type) and a boolean indicating negative infinity. Then we wrap it in `arith::ConstantFloatOp`, specifying both the APFloat value and the target type.

The loop construction demonstrates the lambda style we discussed earlier. The `scf::ForOp` constructor takes six arguments: location, lower bound, upper bound, step, initial values for iter_args, and the body-building lambda. The lambda's signature is fixed by MLIR: it receives a builder (`b`) to use inside the loop, a location, the induction variable (`i`), and current values of loop-carried variables (`iterArgs`).

Inside the lambda, we must use the provided builder `b` rather than the outer `builder`, because the insertion point is different—we're building inside the loop body, not at the outer scope. The `iterArgs[0]` access retrieves the current maximum value, which for the first iteration will be negative infinity (the initial value we provided), and for subsequent iterations will be the value yielded by the previous iteration.

Creating operations inside the loop uses the same patterns as outside: `b.create<...>()` with appropriate arguments. The `memref::LoadOp` loads from memory, taking the memref and indices (wrapped in `ValueRange`). The `arith::MaximumFOp` computes the maximum of two floating-point values, handling special cases correctly. The `scf::YieldOp` must be the last operation in the loop body, passing values to the next iteration—omitting it or placing it in the wrong location will cause verification errors.

After the loop, we extract the final result using `getResult(0)`. Operations that produce values (like `scf.for` with iter_args) return `Value` objects through their result methods. The index 0 corresponds to the first (and in this case, only) result. If we had multiple loop-carried variables, we'd retrieve them as `getResult(1)`, `getResult(2)`, etc.

**Pass 2: Computing Exponentials and Accumulating Sum**

```cpp
  Value zeroFloat = builder.create<arith::ConstantFloatOp>(
      loc, APFloat(0.0f), f32Type);

  auto expSumLoop = builder.create<scf::ForOp>(
      loc, c0, size, c1, ValueRange{zeroFloat},
      [&](OpBuilder& b, Location loc, Value i, ValueRange iterArgs) {
        Value currentSum = iterArgs[0];
        Value val = b.create<memref::LoadOp>(loc, input, ValueRange{i});
        Value shifted = b.create<arith::SubFOp>(loc, val, maxVal);
        Value expVal = b.create<math::ExpOp>(loc, shifted);
        b.create<memref::StoreOp>(loc, expVal, tempBuffer, ValueRange{i});
        Value newSum = b.create<arith::AddFOp>(loc, currentSum, expVal);
        b.create<scf::YieldOp>(loc, ValueRange{newSum});
      }
  );
  Value sumExp = expSumLoop.getResult(0);
```

The second pass follows the same pattern but with more operations in the loop body. We initialize with zero (using `APFloat(0.0f)` for a simple constant), then create another loop with one iter_arg. Notice how the lambda captures `maxVal` from the outer scope (through `[&]`)—this is why the capture-by-reference syntax is essential. The loop needs access to the maximum we computed in pass 1, and the lambda closure provides that access naturally.

Inside the loop, we perform multiple operations in sequence. Each operation consumes values (either from previous operations, from iter_args, or from the outer scope) and produces a new value. The `arith::SubFOp` subtracts two floats, the `math::ExpOp` computes the exponential (our first Math dialect operation!), and the `memref::StoreOp` writes to memory. The store operation takes the value to store, the target memref, and the indices—notice the same `ValueRange{i}` pattern we used for loads.

The `math::ExpOp` is particularly interesting because it's our first transcendental function. At this stage, it's simply a high-level operation representing $e^x$. Later, during the lowering pipeline (which we'll examine in section 6.9), this will be converted to a call to `expf()` from the C math library. The separation between IR construction and lowering is a key MLIR design principle—we build high-level, semantically meaningful IR first, then progressively lower it to executable code.

**Pass 3: Normalization Without Loop-Carried Variables**

```cpp
  builder.create<scf::ForOp>(
      loc, c0, size, c1, std::nullopt,
      [&](OpBuilder& b, Location loc, Value i, ValueRange iterArgs) {
        Value expVal = b.create<memref::LoadOp>(loc, tempBuffer, ValueRange{i});
        Value normalized = b.create<arith::DivFOp>(loc, expVal, sumExp);
        b.create<memref::StoreOp>(loc, normalized, output, ValueRange{i});
        b.create<scf::YieldOp>(loc, ValueRange{});
      }
  );

  // Return
  builder.create<func::ReturnOp>(loc);
  return module;
}
```

The third pass demonstrates loops without iter_args. Instead of providing initial values, we pass `std::nullopt` (C++17's optional-empty indicator), signaling that this loop has no loop-carried variables. The lambda still receives `iterArgs` (it must match the signature), but the `ValueRange` is empty and we don't use it.

The loop body is straightforward: load from the temporary buffer, divide by the sum (computed in pass 2 and captured from outer scope), and store to the output buffer. The `arith::DivFOp` performs floating-point division, which will handle special cases like division by zero according to IEEE 754 rules (producing infinity or NaN as appropriate).

The `scf::YieldOp` at the end takes an empty `ValueRange{}` because there are no values to yield—this loop simply iterates and performs side effects. This is syntactically required even though no values flow through the yield; MLIR requires explicit control flow terminators for all regions.

Finally, we create a `func::ReturnOp` to end the function. Since our function has no return values, we pass no operands to the return operation. The return instruction is mandatory—functions must explicitly return control to their caller, even when returning no values. After constructing the return, our IR is complete, and we return the owning reference to the module.

**Key Patterns in MLIR C++ APIs**

This implementation demonstrates several patterns you'll use repeatedly when building MLIR IR:

1. **Builder-based construction**: All operations are created through `builder.create<OpType>()`, maintaining consistent insertion point management.

2. **Value-based dataflow**: Operations consume `Value` objects as operands and produce `Value` objects as results, explicitly representing SSA dataflow.

3. **Type-specific operations**: Constants, arithmetic, and other operations have different forms for different types (index vs int vs float), reflecting MLIR's precise type system.

4. **Lambda-style regions**: Complex operations like loops take callbacks to build their body regions, providing clean scoping and automatic insertion point management.

5. **Explicit type annotation**: Most operation creators require explicit type information (like `f32Type` for constants), ensuring type consistency is checked at IR construction time.

6. **Location threading**: Every operation creation takes a `Location` argument for debugging, though we use `getUnknownLoc()` for generated code.

Understanding these patterns is essential for MLIR development. The C++ APIs provide type safety and compile-time checking while maintaining the flexibility needed to construct arbitrary IR. As we build more complex operations in later chapters, these same patterns will recur with variations appropriate to each operation's semantics.

## 6.8 Generated MLIR IR

Now let's examine the complete MLIR IR that results from the C++ code we just studied, walking through each pass in detail. The high-level function signature declares a function named `softmax` that takes two parameters: an input memref and an output memref, both of type `memref<?xf32>` (one-dimensional, dynamically-sized arrays of 32-bit floats). The function returns nothing (its result type is the empty tuple), following the out-parameter pattern we established in Chapter 4 for Python integration.

The function begins by querying the size of the input using `memref.dim`, storing it in an index value we'll use as the loop bound. We create constants for zero and one (as index values for loop bounds and steps) and allocate a temporary buffer using `memref.alloca`. The `alloca` operation allocates stack memory with automatic cleanup—suitable for our temporary storage since the buffer only exists for the duration of this function call. For larger arrays in production code, we might use `memref.alloc` (heap allocation) instead, but `alloca` simplifies the implementation by avoiding explicit deallocation. The complete MLIR code that corresponds to the C++ implementation we just examined looks like this:

**Pass 1: Finding Maximum Value**

```mlir
%neg_inf = arith.constant -inf : f32
%max = scf.for %i = %c0 to %size step %c1 
       iter_args(%current_max = %neg_inf) -> (f32) {
  %val = memref.load %input[%i] : memref<?xf32>
  %new_max = arith.maximumf %current_max, %val : f32
  scf.yield %new_max : f32
}
```

The first pass creates a constant representing negative infinity using `arith.constant -inf : f32`. Negative infinity serves as the identity element for the maximum operation—any finite value is greater than it, ensuring our first comparison will replace it with an actual input value. The loop iterates from index 0 to `size` (exclusive) with step 1, carrying the current maximum as a loop variable of type `f32`.

In each iteration, we load a value from the input at index `%i` and compare it with the current maximum using `arith.maximumf`. This operation implements IEEE 754's maximum function, which correctly handles special cases: if either operand is NaN (not a number), the result is NaN, ensuring that data corruption propagates rather than being silently ignored. We yield the new maximum (the larger of the two values), which flows to the next iteration's `%current_max`. After all iterations complete, `%max` holds the maximum input value, which we'll use for numerical stability in pass 2.

**Pass 2: Computing Exponentials and Sum**

```mlir
%zero = arith.constant 0.0 : f32
%sum = scf.for %i = %c0 to %size step %c1 
       iter_args(%current_sum = %zero) -> (f32) {
  %val = memref.load %input[%i] : memref<?xf32>
  %shifted = arith.subf %val, %max : f32
  %exp_val = math.exp %shifted : f32
  memref.store %exp_val, %temp[%i] : memref<?xf32>
  %new_sum = arith.addf %current_sum, %exp_val : f32
  scf.yield %new_sum : f32
}
```

The second pass initializes a zero constant for sum accumulation. The loop again iterates over all elements, but now carries the current sum as a loop variable initialized to zero. In each iteration, we load the input value, subtract the maximum using `arith.subf`, and compute the exponential of the result using `math.exp`. This is where the Math dialect enters our computation—`math.exp` represents the mathematical exponential function $e^x$, which will later be lowered to a call to the C library's `expf()` function.

After computing the exponential, we perform two operations: store the result to our temporary buffer at index `%i` (so we can use it in pass 3), and add it to the current sum using `arith.addf`. The loop yields the updated sum, which accumulates across all iterations. After the loop completes, `%sum` contains the sum of all exponentials, which we need for normalization. This pass demonstrates the combination of reduction (summing), transformation (computing exponentials), and storage (writing to temp buffer) in a single loop—common in numerical algorithms where we want to minimize passes over data.

**Pass 3: Normalization**

```mlir
scf.for %i = %c0 to %size step %c1 {
  %exp_val = memref.load %temp[%i] : memref<?xf32>
  %normalized = arith.divf %exp_val, %sum : f32
  memref.store %normalized, %output[%i] : memref<?xf32>
  scf.yield
}
```

The third pass is the simplest: it iterates over all elements without any loop-carried variables. Each iteration loads the exponential value from the temporary buffer (computed in pass 2), divides it by the sum (also from pass 2) using `arith.divf`, and stores the result to the output buffer. The division produces the normalized probability, and after this loop completes, our output contains valid probabilities summing to 1.

Notice that this loop's `scf.yield` has no operands—there are no loop-carried variables to update. The loop simply iterates and performs side effects (loading and storing from memrefs). This demonstrates the flexibility of `scf.for`: the same construct can express both pure functional-style reductions and imperative-style iteration with side effects.

The function ends with `func.return`, completing the softmax implementation. The entire IR is self-contained: all memory allocations are automatic (stack-based with `alloca`), all operations use standard MLIR dialects (Func, MemRef, Arith, SCF, Math), and the logic clearly corresponds to our three-pass algorithm description. This IR, generated by the C++ code we examined in section 6.7, will undergo several transformation passes before becoming executable code.

## 6.9 Math Dialect Lowering

Converting Math dialect operations to executable code requires understanding how high-level mathematical functions connect to actual implementations. Unlike arithmetic operations that map directly to CPU instructions, transcendental functions like exponential require sophisticated numerical algorithms—typically hundreds of instructions implementing polynomial approximations, range reduction, and special case handling. Rather than implementing these algorithms from scratch, MLIR leverages existing, well-tested implementations through two complementary lowering strategies.

The **math-to-libm** pass converts Math dialect operations into calls to the standard C math library (`libm`). This library, included with every C compiler and operating system, provides highly optimized implementations of mathematical functions that have been refined over decades. The pass transforms `math.exp %x : f32` into `llvm.call @expf(%x) : (f32) -> f32`, where `@expf` is a reference to the C library function. At link time, the LLVM infrastructure automatically links against `libm`, and our generated code calls the same `expf()` function that C programs use.

This approach has several advantages. The libm implementations are extremely accurate, typically providing correct rounding (the result is within 0.5 ULP—units in the last place—of the true mathematical value) and handling all IEEE 754 special cases correctly: infinities, NaNs, subnormal numbers, and signed zeros all behave according to the standard. These implementations are also highly optimized, using architecture-specific features like vector instructions and careful cache management. By calling libm, we inherit decades of work on numerical accuracy and performance without reimplementing any of it.

The alternative **math-to-llvm** pass generates inline code using polynomial approximations or LLVM intrinsics. Instead of function calls, this pass might generate a sequence of multiply and add instructions that evaluate a polynomial approximation to the exponential function. These inline expansions avoid function call overhead and may enable additional optimizations (like vectorization or constant folding) that couldn't occur across function boundaries. However, they typically provide less accuracy than libm and don't always handle special cases correctly—acceptable for some applications but problematic for numerical algorithms that rely on precise behavior.

In our softmax implementation, we prioritize correctness and clarity, so we use the math-to-libm pass. The lowering pipeline first applies this pass, then continues with the other dialect conversions we've seen before. The complete pipeline is:

1. **Canonicalization**: Simplify IR by applying algebraic identities and removing redundant operations
2. **Math to LLVM**: Convert any math operations to LLVM intrinsics (we apply this first as a fallback)
3. **Math to Libm**: Convert math operations to libm calls (overrides the previous pass's results where applicable)
4. **SCF to CF**: Lower structured loops to branches (as we saw in Chapter 3)
5. **Dialect conversions to LLVM**: Convert Func, Arith, MemRef, and CF dialects to LLVM dialect
6. **Reconcile casts**: Clean up type conversions introduced during lowering

After these passes, our module contains only LLVM dialect operations, ready for translation to LLVM IR and subsequent JIT compilation. The Math dialect operations have become function calls to `expf()`, the SCF loops have become basic blocks with branches, and the memref operations have become LLVM pointer operations with offset calculations—all the patterns we've seen in previous chapters, now extended to include mathematical functions.

## 6.10 Python Integration and Testing

Integrating our softmax implementation with Python follows the same patterns we established in previous chapters, using pybind11 to expose our JIT-compiled function as a Python module. The C++ side creates a module and function using the IR generation code we examined, applies the lowering pipeline, creates an ExecutionEngine, looks up the compiled function, and wraps it in a Python-callable interface that handles NumPy arrays.

The Python wrapper function accepts a NumPy array as input, validates that it's a 1D float32 array, allocates an output array of the same size, constructs memref descriptors for both arrays (as we learned in Chapter 2), calls the JIT-compiled function, and returns the output array. This pattern is identical to what we used for SAXPY in Chapter 5, demonstrating the consistent interface that out-parameters provide: regardless of the operation's complexity, the Python integration looks essentially the same.

Testing softmax requires verifying both correctness and numerical stability. For basic correctness, we compare our MLIR implementation against NumPy's softmax implementation (which we write ourselves using NumPy operations, since NumPy doesn't have a built-in softmax). A simple test case with inputs `[1.0, 2.0, 3.0]` should produce probabilities approximately `[0.09, 0.24, 0.67]` that sum to 1.0. We use `np.allclose()` for comparison, which allows small floating-point differences, and verify that the sum is close to 1.0.

The crucial test for numerical stability uses large input values that would cause overflow without max subtraction. If we input `[1000.0, 1001.0, 1002.0]`, a naive implementation would attempt to compute `exp(1000.0)`, which vastly exceeds float32's maximum representable value (approximately 3.4 × 10^38), resulting in infinity. Our implementation subtracts the maximum (1002.0), so the largest exponential is `exp(0.0) = 1.0`, and all computations stay in representable range. The test verifies that our result matches NumPy's (which also uses max subtraction) and contains no NaN or infinity values.

Additional tests cover edge cases and random inputs. When all inputs are zero, softmax should produce a uniform distribution (all values equal to 1/n, where n is the vector size), since all exponentials are `exp(0) = 1`. For random inputs, we verify that probabilities are all positive, sum to 1.0, and match NumPy's results within floating-point tolerance. These tests exercise different code paths and ensure our implementation handles the full range of valid inputs correctly.

We can also benchmark performance against NumPy to understand the overhead of JIT compilation and the efficiency of generated code. For vectors of size 10,000, we typically see performance comparable to NumPy—sometimes faster due to LLVM's optimization, sometimes slower due to function call overhead from the libm calls. The important point is that we're within the same order of magnitude, demonstrating that MLIR can generate competitive code for numerical operations. As we build more complex operations in later chapters, MLIR's ability to optimize across operation boundaries will show increasingly large advantages over frameworks that optimize operations in isolation.

The test suite also includes functions to print the generated IR at different stages of lowering. We can examine the high-level IR (with `scf.for` and `math.exp`) to verify our IR generation, and the fully lowered IR (with only LLVM dialect operations) to see the final form before JIT compilation. These inspection capabilities are valuable for understanding MLIR's transformations and debugging when results don't match expectations.

## 6.11 Comparison with SAXPY

Comparing softmax (this chapter) with SAXPY (Chapter 5) illustrates the progression in complexity as we move toward real machine learning operations. SAXPY performed element-wise operations (multiply and add) that could be computed independently for each element, requiring a single pass over the data with no dependence between iterations. In contrast, softmax requires multiple passes with dependencies: we must find the maximum before computing exponentials, and we must sum exponentials before normalizing.

The dialects used reflect this increased complexity. SAXPY only needed SCF for loops and Arith for basic arithmetic (addition and multiplication). Softmax requires the same dialects plus Math for the exponential function—a transcendental operation beyond basic arithmetic. The addition of Math dialect brings new considerations: lowering strategy (libm vs inline approximations), numerical accuracy, and special case handling (infinities, NaNs). These concerns don't arise with simple arithmetic operations.

Algorithmic structure differs significantly. SAXPY's single loop performed independent operations, with no loop-carried variables and no coordination between iterations. Each iteration was essentially a pure function from input indices to output values. Softmax requires two reduction operations (maximum and sum) using loop-carried variables, plus storage to a temporary buffer for reuse in the third pass. The iterations in passes 1 and 2 are not independent—they accumulate state across the loop, and pass 3 depends on the results of previous passes.

Most importantly, softmax introduces **numerical stability** as a central concern. SAXPY had no numerical issues—addition and multiplication of normal floating-point values produce accurate results without special precautions. Softmax, however, can easily overflow without the max subtraction technique. This highlights a crucial principle in numerical computing: some algorithms are inherently unstable in finite-precision arithmetic and require reformulation for reliable implementation. As we implement more ML operations in subsequent chapters, we'll encounter similar situations where the straightforward mathematical formulation isn't suitable for direct implementation.

Memory usage also differs. SAXPY required no additional memory beyond inputs and outputs—each element's computation was self-contained. Softmax allocates a temporary buffer the same size as the input, doubling the memory footprint. This pattern is common in multi-pass algorithms: intermediate results must be stored for later use. In Chapter 11, when we implement attention mechanisms, we'll see even more complex memory patterns with multiple temporary buffers storing intermediate values for various stages of the computation.

## 6.12 Looking Ahead

With softmax implemented, we've completed our first real machine learning operation and demonstrated essential patterns we'll use throughout the rest of this book: multi-pass algorithms, loop-carried variables for reductions, the Math dialect for mathematical functions, and numerical stability techniques. The three-pass structure we used here—global reduction, element-wise transformation, element-wise normalization—appears in many ML operations with variations.

Chapter 7 will introduce **neural network operations** like matrix multiplication with bias and ReLU activation, building on the patterns established here but working with higher-dimensional tensors. We'll see how operations compose into computation graphs, how Linalg dialect provides high-level representations for common ML operations, and how the patterns we've learned (bufferization, lowering, Python integration) scale to more complex workloads.

Chapter 8 introduces **custom dialects**, allowing us to define domain-specific operations that better represent transformer computations. Rather than expressing everything in terms of generic operations like loops and arithmetic, we'll create operations like `transformer.attention` that encapsulate entire algorithm patterns. This abstraction will prove essential for the optimizations we'll implement in later chapters, where reasoning about high-level semantics enables transformations that would be impossible at the level of individual loops and arithmetic operations.

Chapter 11 will return to softmax in the context of **attention mechanisms**, where it appears as a key component of the attention score computation. We'll see how the softmax we implemented here integrates with query-key-value computations, how batching affects the implementation, and how the same numerical stability principles apply at larger scales. The attention mechanism ties together many concepts from earlier chapters: matrix operations from Chapter 1, dynamic shapes from Chapter 2, structured operations from Chapter 5, and softmax from this chapter.

The journey from fixed-size matrix multiplication in Chapter 1 to numerically stable softmax in Chapter 6 has equipped us with the tools needed for production ML implementations. We understand multiple abstraction levels (tensors, structured operations, explicit loops), we can manage compilation and execution, and we know how to handle numerical concerns that arise in real algorithms. The remaining chapters will build on this foundation, increasing in complexity but following the same principles: high-level representation, progressive lowering, careful numerical implementation, and thorough testing.
