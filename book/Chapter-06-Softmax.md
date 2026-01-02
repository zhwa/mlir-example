# Chapter 6: Mathematical Operations — Implementing Softmax with Tensor Reductions

In this chapter, we implement **softmax** using `linalg.reduce` for reductions and `linalg.generic` for element-wise operations. This represents a significant evolution from the explicit loop-based approach, allowing us to express complex mathematical operations declaratively while relying on the compiler to generate efficient code.

This chapter introduces the **Math dialect**, which provides mathematical functions like exponential, logarithm, and trigonometric operations. We'll use it to implement **softmax**, a fundamental activation function that appears everywhere in modern neural networks—particularly in transformers, where it powers the attention mechanism we'll explore in Chapter 11. Unlike the simple element-wise SAXPY from Chapter 5, softmax requires a four-pass algorithm with reductions and introduces important numerical stability concerns.

Our implementation demonstrates several key patterns: using `linalg.reduce` for maximum and sum operations (the standard MLIR way to express reductions), employing `linalg.generic` for element-wise exponential and normalization operations, and handling numerical stability through max subtraction. Through bufferization, these high-level operations automatically transform into efficient memref code with explicit loops—giving us both expressiveness at the high level and performance at the low level.

## 6.1 High-Level Softmax Design

Before diving into the math, let's look at how we express softmax using declarative tensor operations.

### Declarative Reductions

Instead of writing explicit loops with accumulators, we use `linalg.reduce`. This operation abstracts the reduction logic, allowing the compiler to choose the best iteration strategy.

**Key Code: Linalg Reduction**
```mlir
%max_tensor = linalg.reduce ins(%input : tensor<?xf32>)
                           outs(%init_max : tensor<f32>)
                           dimensions = [0]
  (%in: f32, %init: f32) {
    %new_max = arith.maximumf %in, %init : f32
    linalg.yield %new_max : f32
  }
```

This snippet defines *what* to compute (maximum value along dimension 0) rather than *how* to compute it. The `linalg.reduce` operation takes an input tensor and an initial value (accumulator), and applies the reduction region (the body) to combine elements.

### The Four-Pass Algorithm

Our softmax implementation uses four distinct passes, each operating on tensors:

1. **Max Reduction**: Find the maximum value in the input vector (`linalg.reduce`).
2. **Exp Transformation**: Compute $e^{x_i - \max}$ for each element (`linalg.generic`).
3. **Sum Reduction**: Sum the exponential values (`linalg.reduce`).
4. **Normalization**: Divide each exponential by the sum (`linalg.generic`).

Each pass is a high-level operation. Bufferization will later fuse these operations where possible or allocate necessary temporary buffers automatically.

## 6.2 The Math Dialect

The Math dialect complements the Arith dialect we introduced in Chapter 5 by providing operations for mathematical functions beyond basic arithmetic. While Arith handles addition, multiplication, comparisons, and other elementary operations with precise semantics for integers and floating-point numbers, Math provides transcendental functions—operations whose results cannot be expressed as finite combinations of basic arithmetic operations.

The Math dialect includes operations like `math.exp` (exponential function $e^x$), `math.log` (natural logarithm), `math.sqrt` (square root), `math.sin` and `math.cos` (trigonometric functions), `math.pow` (power function), and many others. These operations maintain the same type system as Arith, supporting various floating-point precisions (f32, f64, f16) and integer types where applicable. The dialect is specifically designed for numerical computing workloads common in scientific applications and machine learning.

What makes Math dialect operations particularly useful is their flexibility in lowering strategies. Unlike Arith operations which typically lower directly to LLVM's built-in arithmetic instructions, Math operations can take multiple paths to native code. The `math-to-libm` pass converts Math operations into calls to the standard C math library (`libm`), leveraging decades of work on accurate and efficient implementations of mathematical functions. For example, `math.exp %x : f32` becomes a call to the C library's `expf()` function. This approach provides excellent accuracy and handles edge cases correctly—important considerations when dealing with numerical algorithms.

Alternatively, the `math-to-llvm` pass can generate inline polynomial approximations or use LLVM intrinsics, trading some accuracy for reduced function call overhead. This flexibility allows developers to choose between accuracy and performance based on their application's requirements. For our purposes in this chapter, we'll use the `math-to-libm` path because it provides the accuracy needed for correct softmax implementation and offers a clear illustration of how high-level mathematical operations connect to standard library implementations.

In our softmax implementation, we'll primarily use `math.exp` to compute exponentials. The exponential function is central to many machine learning algorithms: it appears in softmax (our focus here), in the cross-entropy loss function, in sigmoid activations, in attention mechanisms, and in probabilistic models throughout deep learning. Understanding how to work with exponentials and handle their numerical properties is essential for implementing ML systems correctly.

## 6.3 The Softmax Function

Softmax is a fundamental operation in machine learning that converts a vector of arbitrary real numbers into a probability distribution. Given an input vector $\mathbf{x} = [x_1, x_2, \ldots, x_n]$, softmax produces an output vector $\mathbf{y} = [y_1, y_2, \ldots, y_n]$ where each element is computed as:

$$
y_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

Each output value $y_i$ is positive (since exponentials are always positive), and the sum of all outputs equals 1, satisfying the properties of a probability distribution: $\sum_{i=1}^{n} y_i = 1$ and $0 < y_i < 1$ for all $i$. This transformation is particularly useful in classification tasks where we need to interpret neural network outputs as probabilities over different classes.

The softmax function has several important mathematical properties. It's translation-invariant, meaning that adding a constant to all inputs doesn't change the output: $\text{softmax}(\mathbf{x} + c) = \text{softmax}(\mathbf{x})$ for any constant $c$. This property, which we can verify by algebraic manipulation, turns out to be crucial for numerical stability, as we'll see shortly. Softmax also amplifies differences between inputs—larger input values receive exponentially more weight in the output, which is why it's called "softmax" rather than just "max". Unlike the hard maximum function that outputs 1 for the largest element and 0 for all others, softmax provides a differentiable alternative that assigns probabilities based on relative magnitudes.

In modern deep learning, softmax appears in multiple contexts. Most visibly, it's typically the final layer in classification networks, converting logits (raw network outputs) into class probabilities. In transformer architectures like GPT and BERT, softmax is applied to attention scores in every attention layer, determining how much each token should attend to every other token—a pattern we'll explore in detail in Chapter 11. It also appears in various loss functions, optimization algorithms, and architectural components throughout neural networks.

However, the naive implementation shown in the equation above has a critical problem: for large input values, computing $e^{x_i}$ can cause floating-point overflow. Since the exponential function grows extremely rapidly, even moderately large inputs can produce results that exceed the maximum representable floating-point value. For example, in 32-bit floating-point arithmetic (which we use throughout this book), `exp(89)` already overflows to infinity. Once overflow occurs, the softmax computation becomes meaningless, producing `nan` (not a number) values when dividing infinity by infinity.

## 6.4 Numerical Stability Through Max Subtraction

The solution to overflow in softmax leverages the translation-invariance property we mentioned earlier. We can subtract any constant from all input values without changing the final result, so we choose to subtract the maximum input value before computing exponentials. This **max subtraction technique** is the standard approach for numerically stable softmax implementation.

Mathematically, we reformulate the computation as:

$$
y_i = \frac{e^{x_i - \max(\mathbf{x})}}{\sum_{j=1}^{n} e^{x_j - \max(\mathbf{x})}}
$$

This formulation is mathematically equivalent to the original (thanks to translation invariance), but the numerical behavior is drastically different. After subtracting the maximum, all input values become zero or negative. The largest value becomes $\max(\mathbf{x}) - \max(\mathbf{x}) = 0$, and all others are negative. Since $e^0 = 1$ and exponentials of negative numbers are less than 1, the largest exponential in our computation is exactly 1, completely eliminating the possibility of overflow.

Underflow—when exponentials of very negative numbers become too small to represent—can still occur, but this is harmless. Values that underflow to zero simply contribute nothing to the sum in the denominator and produce zero probabilities in the output, which is the correct mathematical result for inputs that are much smaller than the maximum. The numerical stability achieved through max subtraction is so important that every production implementation of softmax uses this technique, from NumPy to PyTorch to TensorFlow.

This stability concern illustrates a broader principle in numerical computing: mathematically equivalent formulations can have vastly different numerical properties when implemented with finite-precision arithmetic. Throughout this book, we'll encounter other examples where algorithm design must account for floating-point arithmetic's limitations. The max subtraction technique is particularly elegant because it requires minimal additional computation (one pass to find the maximum) and completely solves the overflow problem without approximations or loss of accuracy.

## 6.5 Four-Pass Algorithm Structure

Implementing numerically stable softmax requires breaking the computation into four sequential passes. This structure differs from the single-pass element-wise operations we've seen in previous chapters, and it demonstrates how to compose reductions and element-wise transformations.

The **first pass** computes the maximum value of the input vector using `linalg.reduce`. We must scan the entire input before we can begin computing exponentials. This operation reduces the rank-1 input tensor to a rank-0 (scalar) tensor containing the maximum value.

The **second pass** computes the exponentials of the shifted values ($e^{x_i - \max}$). This is a `linalg.generic` operation that takes the input tensor and the maximum value (extracted from the result of pass 1) and produces a new tensor of exponentials.

The **third pass** sums the exponential values using `linalg.reduce`. This takes the tensor produced by pass 2 and reduces it to a scalar sum.

The **fourth pass** normalizes the exponentials by dividing each by the sum. This `linalg.generic` operation reads the exponential tensor (from pass 2) and the sum (from pass 3) and produces the final probability distribution.

This four-pass structure is declarative. We describe *what* to compute (max, exp, sum, div) rather than *how* to iterate. The compiler's bufferization pass will later fuse these operations where possible or allocate necessary temporary buffers.

## 6.6 Understanding Reductions with Linalg

In Chapter 5, we used `linalg.generic` for element-wise operations where the output shape matches the input shape. For operations like finding the maximum or summing values, we need **reductions**—operations that reduce the number of dimensions. The `linalg.reduce` operation is designed specifically for this purpose.

A reduction operation takes an input tensor and an initial value (the accumulator). It iterates along specified dimensions, combining elements with the accumulator using a binary operation (like max or add).

### The Anatomy of linalg.reduce

```mlir
%max_tensor = linalg.reduce ins(%input : tensor<?xf32>)
                           outs(%init : tensor<f32>)
                           dimensions = [0]
  (%in: f32, %acc: f32) {
    %new_acc = arith.maximumf %in, %acc : f32
    linalg.yield %new_acc : f32
  }
```

1.  **ins**: The input tensor to reduce.
2.  **outs**: The initial value for the reduction, wrapped in a tensor. For finding a maximum, this is a 0-D tensor containing negative infinity.
3.  **dimensions**: The list of dimensions to reduce. For a 1D vector, we reduce dimension 0.
4.  **Region**: The body defines how to combine the current element (`%in`) with the accumulator (`%acc`). For max, we use `arith.maximumf`. For sum, we use `arith.addf`.

The result is a tensor with the reduced dimensions removed. Reducing a `tensor<?xf32>` along dimension 0 produces a `tensor<f32>` (scalar tensor).

## 6.7 Constructing Linalg Operations

In Chapter 5, we saw how to build `linalg.generic`. Building `linalg.reduce` follows a similar pattern but requires specifying reduction dimensions and a reducer region.

When using the C++ API, we use lambda functions to construct the bodies of these operations. This "lambda-style" construction handles block creation, argument management, and insertion points automatically.

For `linalg.reduce`, the lambda receives the builder, location, and a range of arguments. `args[0]` is the element from the input tensor, and `args[1]` is the current accumulator value. The lambda must build the reduction logic (e.g., `arith.maximumf`) and yield the result.

```cpp
auto reduceOp = builder.create<linalg::ReduceOp>(
    loc, input, initTensor,
    SmallVector<int64_t>{0}, // dimensions
    [&](OpBuilder& b, Location loc, ValueRange args) {
        Value val = args[0];
        Value acc = args[1];
        Value res = b.create<arith::MaximumFOp>(loc, val, acc);
        b.create<linalg::YieldOp>(loc, res);
    }
);
```

This declarative style abstracts away the loop mechanics. We don't manage induction variables or loop bounds; we simply define the combination logic.

## 6.8 Building Softmax with C++ APIs

Now let's see how to construct the complete softmax implementation using MLIR's C++ APIs. We'll focus on the most critical part: constructing the reduction operation. The full implementation follows the same pattern of creating operations and chaining their results.

**Key Code: Constructing linalg.reduce**

The following C++ code demonstrates how to build the first pass (finding the maximum value). It creates the initial accumulator, defines the reduction dimensions, and provides a lambda to generate the reduction body.

```cpp
  // 1. Create initial value (-inf) wrapped in a 0-D tensor
  Value negInf = builder.create<arith::ConstantOp>(
      loc, builder.getFloatAttr(f32Type, 
          APFloat::getInf(f32Type.getFloatSemantics(), /*Negative=*/true)));
  
  auto scalarTensorType = RankedTensorType::get({}, f32Type);
  Value initMaxTensor = builder.create<tensor::FromElementsOp>(
      loc, scalarTensorType, ValueRange{negInf});

  // 2. Create linalg.reduce operation
  auto reduceMaxOp = builder.create<linalg::ReduceOp>(
      loc, input, initMaxTensor,
      SmallVector<int64_t>{0}, // Reduce along dimension 0
      [&](OpBuilder& b, Location loc, ValueRange args) {
        // Body: combine input element (args[0]) with accumulator (args[1])
        Value newMax = b.create<arith::MaximumFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, newMax);
      }
  );
  
  // 3. Extract scalar result
  Value maxTensor = reduceMaxOp.getResult(0);
  Value maxVal = builder.create<tensor::ExtractOp>(loc, maxTensor, ValueRange{});
```

**Pass 2: Computing Exponentials (linalg.generic)**

With the maximum value extracted, we can now compute the exponentials. This is an element-wise operation, so we use `linalg.generic`. Unlike `linalg.reduce`, which lowers the rank, `linalg.generic` preserves the tensor shape.

We first create an empty tensor to hold the results using `tensor::EmptyOp`. Then, we construct the `linalg::GenericOp`. The key here is the lambda body, which captures `maxVal` from the outer scope. Inside the body, we subtract the maximum from the input element and apply `math::ExpOp`.

```cpp
  // Create empty output tensor
  Value emptyTensor = builder.create<tensor::EmptyOp>(
      loc, ValueRange{size}, f32Type);

  // linalg.generic: compute exp(input[i] - max)
  auto expOp = builder.create<linalg::GenericOp>(
      loc, dynamicTensorType, input, emptyTensor,
      indexingMaps, iteratorTypes,
      [&](OpBuilder& b, Location loc, ValueRange args) {
        Value val = args[0];
        // Capture maxVal from Pass 1 for numerical stability
        Value shifted = b.create<arith::SubFOp>(loc, val, maxVal);
        Value expVal = b.create<math::ExpOp>(loc, shifted);
        b.create<linalg::YieldOp>(loc, expVal);
      }
  );
  Value expTensor = expOp.getResult(0);
```

**Pass 3: Summing Exponentials (linalg.reduce)**

Now we need to sum the exponential values to compute the denominator. This is another reduction, very similar to Pass 1, but with two differences:
1.  The initial accumulator is `0.0` instead of negative infinity.
2.  The reduction operation is `arith::AddFOp` instead of `arith::MaximumFOp`.

```cpp
  // Initialize accumulator with 0.0
  Value zeroFloat = builder.create<arith::ConstantOp>(
      loc, builder.getFloatAttr(f32Type, APFloat(0.0f)));
  Value initSumTensor = builder.create<tensor::FromElementsOp>(
      loc, scalarTensorType, ValueRange{zeroFloat});

  // linalg.reduce: sum(exp_tensor)
  auto reduceSumOp = builder.create<linalg::ReduceOp>(
      loc, expTensor, initSumTensor,
      SmallVector<int64_t>{0},
      [&](OpBuilder& b, Location loc, ValueRange args) {
        Value newSum = b.create<arith::AddFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, newSum);
      }
  );
  Value sumTensor = reduceSumOp.getResult(0);
  Value sumVal = builder.create<tensor::ExtractOp>(loc, sumTensor, ValueRange{});
```

**Pass 4: Normalization (linalg.generic)**

Finally, we produce the probability distribution by dividing each exponential value by the sum. This is another element-wise `linalg.generic` operation. It reads from `expTensor` (the result of Pass 2) and captures `sumVal` (the result of Pass 3).

```cpp
  // linalg.generic: exp_tensor[i] / sum
  auto normalizeOp = builder.create<linalg::GenericOp>(
      loc, dynamicTensorType, expTensor, emptyTensor, // Reuse empty structure
      indexingMaps, iteratorTypes,
      [&](OpBuilder& b, Location loc, ValueRange args) {
        Value normalized = b.create<arith::DivFOp>(loc, args[0], sumVal);
        b.create<linalg::YieldOp>(loc, normalized);
      }
  );
  
  builder.create<func::ReturnOp>(loc, normalizeOp.getResult(0));
```

This modular construction demonstrates the power of the C++ API. We build complex algorithms by composing simple, declarative operations, letting the compiler handle the low-level details of memory management and loop generation.

## 6.9 Generated MLIR IR

The C++ code generates high-level Linalg IR that clearly reflects our four-pass algorithm. Let's examine each pass in the generated code to understand how the declarative operations map to our logical steps.

**Pass 1: Max Reduction**

The first pass finds the maximum value in the input tensor. We use `linalg.reduce` with `arith.maximumf` in the reduction body. The result is a rank-0 tensor (scalar) containing the maximum value, which we extract for use in the next pass.

```mlir
  // Pass 1: Max Reduction
  %max_tensor = linalg.reduce ins(%arg0 : tensor<?xf32>)
                             outs(%init_max : tensor<f32>)
                             dimensions = [0]
    (%in: f32, %init: f32) {
      %0 = arith.maximumf %in, %init : f32
      linalg.yield %0 : f32
    }
  %max = tensor.extract %max_tensor[] : tensor<f32>
```

**Pass 2: Exp Transformation**

The second pass computes the exponentials. This is a `linalg.generic` operation that iterates over the input tensor. Inside the body, we subtract the maximum value (computed in Pass 1) from the current element to ensure numerical stability, then apply `math.exp`. The result is a new tensor of exponentials.

```mlir
  // Pass 2: Exp Transformation
  %exp_tensor = linalg.generic { ... } 
    ins(%arg0 : tensor<?xf32>) outs(%empty : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %shifted = arith.subf %in, %max : f32
      %exp = math.exp %shifted : f32
      linalg.yield %exp : f32
    }
```

**Pass 3: Sum Reduction**

The third pass sums the exponential values. Similar to Pass 1, we use `linalg.reduce`, but this time with `arith.addf` to accumulate the sum. The input is the exponential tensor from Pass 2.

```mlir
  // Pass 3: Sum Reduction
  %sum_tensor = linalg.reduce ins(%exp_tensor : tensor<?xf32>)
                             outs(%init_sum : tensor<f32>)
                             dimensions = [0]
    (%in: f32, %init: f32) {
      %1 = arith.addf %in, %init : f32
      linalg.yield %1 : f32
    }
  %sum = tensor.extract %sum_tensor[] : tensor<f32>
```

**Pass 4: Normalization**

The final pass normalizes the exponentials to produce probabilities. We use `linalg.generic` to iterate over the exponential tensor again, dividing each element by the sum computed in Pass 3.

```mlir
  // Pass 4: Normalization
  %result = linalg.generic { ... } 
    ins(%exp_tensor : tensor<?xf32>) outs(%empty : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %norm = arith.divf %in, %sum : f32
      linalg.yield %norm : f32
    }
```

This IR is concise and captures the mathematical intent. It is much easier to analyze and optimize than explicit loops.

## 6.10 The Lowering Pipeline

To execute this high-level IR, we need a robust lowering pipeline. The pipeline in `src/lowering.cpp` transforms our declarative tensor operations into executable machine code through a series of progressive lowerings.

The process begins with **Canonicalization**, which simplifies the IR by applying algebraic identities and removing redundant operations. This ensures that subsequent passes work on a clean representation.

The most critical step is **One-Shot Bufferize**. This pass analyzes the tensor operations and allocates memory, converting `tensor.empty` to `memref.alloc` (or stack allocation) and transforming `linalg.generic` and `linalg.reduce` operations on tensors into operations on memrefs. It also updates the function signature to use out-parameters, matching the C ABI we expect.

Once bufferized, the **Linalg to Loops** pass converts the structured operations into explicit loops. `linalg.reduce` becomes a loop with `iter_args` (loop-carried variables), while `linalg.generic` becomes a parallel loop nest. This is where the high-level intent is translated into imperative control flow.

We also need to handle the mathematical operations. The **Math to Libm** pass converts `math.exp` operations into calls to the standard C library's `expf` function. This ensures we get a highly optimized and numerically accurate implementation of the exponential function.

Finally, the **SCF to Control Flow** pass lowers the structured loops to basic blocks and branches, and the **LLVM Conversion** pass translates everything into the LLVM dialect, ready for JIT compilation. This pipeline demonstrates the power of MLIR: we write high-level tensor code, and the compiler handles the complexity of memory management, loop generation, and library calls.

## 6.10 Python Integration and Testing

Integrating our softmax implementation with Python follows the same patterns we established in previous chapters, using pybind11 to expose our JIT-compiled function as a Python module. The C++ side creates a module and function using the IR generation code we examined, applies the lowering pipeline, creates an ExecutionEngine, looks up the compiled function, and wraps it in a Python-callable interface that handles NumPy arrays.

The Python wrapper function accepts a NumPy array as input, validates that it's a 1D float32 array, allocates an output array of the same size, constructs memref descriptors for both arrays (as we learned in Chapter 2), calls the JIT-compiled function, and returns the output array. This pattern is identical to what we used for SAXPY in Chapter 5, demonstrating the consistent interface that out-parameters provide: regardless of the operation's complexity, the Python integration looks essentially the same.

Testing softmax requires verifying both correctness and numerical stability. For basic correctness, we compare our MLIR implementation against a NumPy reference implementation:

```python
def test_softmax_basic():
    """Test basic softmax operation."""
    input_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    
    # NumPy reference implementation
    def numpy_softmax(x):
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)
    
    expected = numpy_softmax(input_arr)
    result = ch6_softmax.softmax(input_arr)  # MLIR JIT version
    
    assert np.allclose(result, expected, rtol=1e-5)
    assert np.abs(np.sum(result) - 1.0) < 1e-6  # Sum should be 1.0
```

A simple test case with inputs `[1.0, 2.0, 3.0]` should produce probabilities approximately `[0.09, 0.24, 0.67]` that sum to 1.0. We use `np.allclose()` for comparison, which allows small floating-point differences, and verify that the sum is close to 1.0.

The crucial test for numerical stability uses large input values that would cause overflow without max subtraction. If we input `[1000.0, 1001.0, 1002.0]`, a naive implementation would attempt to compute `exp(1000.0)`, which vastly exceeds float32's maximum representable value (approximately 3.4 × 10^38), resulting in infinity. Our implementation subtracts the maximum (1002.0), so the largest exponential is `exp(0.0) = 1.0`, and all computations stay in representable range. The test verifies that our result matches NumPy's (which also uses max subtraction) and contains no NaN or infinity values.

Additional tests cover edge cases and random inputs. When all inputs are zero, softmax should produce a uniform distribution (all values equal to 1/n, where n is the vector size), since all exponentials are `exp(0) = 1`. For random inputs, we verify that probabilities are all positive, sum to 1.0, and match NumPy's results within floating-point tolerance. These tests exercise different code paths and ensure our implementation handles the full range of valid inputs correctly.

The test suite also includes functions to print the generated IR at different stages of lowering. We can examine the high-level IR (with `scf.for` and `math.exp`) to verify our IR generation, and the fully lowered IR (with only LLVM dialect operations) to see the final form before JIT compilation. These inspection capabilities are valuable for understanding MLIR's transformations and debugging when results don't match expectations.

## 6.12 Summary

In this chapter, we implemented a numerically stable softmax function using MLIR's high-level tensor operations. We learned how to use `linalg.reduce` to perform aggregations like finding the maximum value and summing elements, and how to combine these with `linalg.generic` for element-wise transformations. We also introduced the Math dialect, which provides essential transcendental functions like `math.exp`.

The four-pass algorithm we implemented—Max Reduction, Exp Transformation, Sum Reduction, and Normalization—demonstrates a common pattern in numerical computing: decomposing complex operations into a sequence of simpler, declarative steps. By expressing these steps as high-level tensor operations, we allow the compiler to handle the details of memory management, loop generation, and optimization through the bufferization and lowering pipelines.

We also addressed the critical issue of numerical stability. By subtracting the maximum value before computing exponentials, we ensured that our implementation remains robust even for large input values, preventing floating-point overflow. This technique, combined with the precision of the `math-to-libm` lowering path, results in a production-quality implementation that matches the behavior of standard libraries like NumPy.

With these tools in hand, we are ready to tackle more complex neural network operations. In the next chapter, we will extend these concepts to higher-dimensional tensors and implement matrix multiplication, the workhorse of deep learning.