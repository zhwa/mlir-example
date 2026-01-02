# Chapter 5: Vector Operations with Tensors

In the first four chapters, we built foundational knowledge of MLIR's compilation model. Chapter 1-3 used memrefs directly to understand execution mechanics, memory layout, and JIT compilation. Chapter 4 introduced bufferization—the transformation from functional tensor IR to imperative memref IR—and explained why modern MLIR compilers separate these concerns. That chapter was the pivot point. Now, starting from Chapter 5, we adopt **tensors**: the industry-standard abstraction used in production MLIR systems worldwide.

This shift reflects how real ML compilers work. Torch-MLIR (PyTorch's MLIR backend), IREE (Google's ML runtime), StableHLO (TensorFlow/JAX), and other production systems all follow the same pattern: high-level operations use immutable tensors, optimization happens at the tensor level, and bufferization converts to executable memref code late in the compilation pipeline. By adopting this pattern, we're not just learning MLIR—we're learning the patterns that power modern AI frameworks.

Our vehicle for demonstrating this approach is **SAXPY** (Single-Precision A·X Plus Y), a fundamental operation from linear algebra: `C[i] = α · A[i] + B[i]`. SAXPY is simpler than matrix multiplication but rich enough to demonstrate tensor operations, dynamic shapes, the Linalg dialect, and the complete bufferization pipeline. We'll implement SAXPY using `linalg.generic` with tensor types, then watch as bufferization automatically transforms our functional code into efficient imperative machine code.

By the end of this chapter, you'll understand:
- Why tensors are the right abstraction for high-level ML operations
- How `linalg.generic` expresses parallel computations on tensors
- How the tensor dialect (`tensor.empty`, `tensor.dim`) handles dynamic shapes
- How One-Shot Bufferize converts functional tensors to imperative memrefs
- How the complete pipeline (canonicalization → bufferization → loop lowering → LLVM) produces executable code
- When to use tensor patterns versus direct memref operations

This knowledge forms the foundation for all subsequent chapters. From Chapter 5 onward, every operation we implement—softmax, ReLU, convolutions, attention, transformers—will use tensors.

## 5.1 The Tensor Philosophy: Why Immutable Matters

Modern ML frameworks think in terms of immutable data. When you write `result = alpha * A + B` in PyTorch, you're not mutating A or B—you're creating a new tensor that holds the result. This functional semantics isn't just convenient for users; it enables powerful compiler optimizations that would be impossible with mutable operations.

### Functional vs Imperative: A Tale of Two Styles

Consider two ways to express the same SAXPY computation. In the imperative style (used in Chapters 1-4 with memrefs), we explicitly take a pre-allocated output buffer as a parameter and loop over elements with explicit load/store operations:

```mlir
func.func @saxpy(%alpha: f32,
                 %A: memref<?xf32>,
                 %B: memref<?xf32>,
                 %C: memref<?xf32>) {
  // Mutate C in-place
  %c0 = arith.constant 0 : index
  %size = memref.dim %A, %c0 : memref<?xf32>
  %c1 = arith.constant 1 : index
  
  scf.for %i = %c0 to %size step %c1 {
    %a = memref.load %A[%i] : memref<?xf32>
    %b = memref.load %B[%i] : memref<?xf32>
    %scaled = arith.mulf %alpha, %a : f32
    %sum = arith.addf %scaled, %b : f32
    memref.store %sum, %C[%i] : memref<?xf32>
  }
  return
}
```

This approach mutates the output buffer in-place, requiring the programmer to manage memory layout, iteration order, and potential aliasing issues.

In contrast, the functional style (used from Chapter 5 onwards with tensors) returns a new tensor rather than mutating inputs. It uses declarative operations like `linalg.generic` where the compiler chooses the iteration strategy:

```mlir
#map = affine_map<(d0) -> (d0)>

func.func @saxpy(%alpha: f32,
                 %A: tensor<?xf32>,
                 %B: tensor<?xf32>) -> tensor<?xf32> {
  // Create and return new tensor
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %A, %c0 : tensor<?xf32>
  %empty = tensor.empty(%dim) : tensor<?xf32>
  
  %result = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%A, %B : tensor<?xf32>, tensor<?xf32>)
    outs(%empty : tensor<?xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %scaled = arith.mulf %alpha, %a : f32
    %sum = arith.addf %scaled, %b : f32
    linalg.yield %sum : f32
  } -> tensor<?xf32>
  
  return %result : tensor<?xf32>
}
```

There are no explicit load or store operations; they are abstracted away. This immutable semantics is key to enabling advanced optimizations.

### Why Immutability Wins

The power of immutability becomes clear when you consider optimization. Suppose you have a sequence of operations where one operation produces a temporary tensor that is consumed by the next. Because tensors are immutable, the compiler knows immediately that there is no aliasing between the temporary and the input—they are distinct values. It also knows the first operation has no side effects on global state.

This knowledge allows the compiler to perform dead value elimination (removing the temporary if it's unused) and, more importantly, operation fusion. The compiler can fuse the producer and consumer into a single operation, eliminating the intermediate memory allocation entirely. With mutable memrefs, achieving this requires complex alias analysis to prove that pointers don't overlap and that side effects are safe to reorder—a notoriously difficult problem for compilers. By making immutability explicit at the type level, MLIR enables these optimizations by default.

### The Industry Consensus

Every major ML compiler uses this architecture:

**Torch-MLIR** (PyTorch → MLIR):
```mlir
// PyTorch operations lower to Torch dialect with tensors
%result = torch.aten.relu %input : !torch.tensor -> !torch.tensor
// Then lower to Linalg (still tensors)
%result = linalg.generic {...} ins(%input : tensor<?x?xf32>) -> tensor<?x?xf32>
// Bufferize late
// (automatic conversion to memref operations)
```

**IREE** (Google's ML runtime):
- All frontends (TensorFlow, PyTorch, JAX) produce tensor IR
- Optimization passes work on tensors
- Bufferization happens per-function or per-dispatch-region
- Low-level codegen uses memrefs

**StableHLO** (TensorFlow/JAX intermediate representation):
- Entirely tensor-based
- No memref operations at the StableHLO level
- Compilers consuming StableHLO (IREE, XLA) handle bufferization

This universal adoption isn't coincidence—tensors are simply the right abstraction level for ML compilation.

## 5.2 The Linalg Dialect Revisited: Tensor Operations

We've seen Linalg before (Chapter 1's `linalg.matmul`), but now we'll use it properly: with tensor types. The Linalg dialect is designed for expressing structured linear algebra operations, and its power comes from working on immutable tensors rather than mutable memrefs.

### Understanding linalg.generic with Tensors

The `linalg.generic` operation is Linalg's Swiss Army knife—a highly flexible operation that can express any structured computation on multi-dimensional arrays. When used with tensors, it becomes a pure function: given input tensors, it produces an output tensor without side effects.

The anatomy of `linalg.generic`:

```mlir
%result = linalg.generic {
  indexing_maps = [...],      // How to map iteration space to array dimensions
  iterator_types = [...]      // parallel, reduction, or window
} ins(%input1, %input2, ... : tensor<...>, tensor<...>)
  outs(%init : tensor<...>) {
^bb0(%in1: element_type, %in2: element_type, %out_elem: element_type):
  // Body: Compute output element from input elements
  %computed = ... operations on %in1, %in2, ...
  linalg.yield %computed : element_type
} -> tensor<...>
```

**Key components:**

1. **indexing_maps**: Affine maps defining how the iteration space maps to each tensor's dimensions. For SAXPY (element-wise operation), all three tensors use identity mapping: iteration index i maps directly to element i.

2. **iterator_types**: Classification of each dimension as `parallel` (iterations are independent, can run concurrently), `reduction` (accumulate across this dimension), or `window` (sliding window, like convolution). SAXPY uses `["parallel"]` because each output element is computed independently.

3. **ins/outs**: Input tensors (read-only) and output tensor (destination). The output is called `outs` even though conceptually it's being computed, not mutated. This is for uniformity with the memref version.

4. **Body block**: A region that computes one output element. The block arguments correspond to elements from each input tensor at the current iteration point, plus the current output element. The body performs element-wise computation and yields the result.

5. **Result type**: The operation returns the computed tensor. This is crucial—with tensor semantics, `linalg.generic` is a pure function that returns a new tensor.

### SAXPY with linalg.generic

Let's break down SAXPY's tensor implementation step by step:

```mlir
%size = tensor.dim %A, %c0 : tensor<?xf32>
```

This operation is crucial for writing dimension-agnostic code. When your function receives a `tensor<?xf32>` (dynamic 1D vector), you don't know its size at compile time. The caller might pass a vector of length 100 or 10,000. To create an output tensor of the correct size, you must query the input size using `tensor.dim`.

The `tensor.dim` operation extracts the size for the specified dimension index (0 for the first dimension). This information is available at runtime. For statically-shaped tensors (like `tensor<8x16xf32>`), the compiler can optimize this away entirely, replacing the operation with a constant. This allows us to write generic code that works for both static and dynamic shapes.

### Rank and Shape Constraints

Tensors have a *rank*: the number of dimensions. A 1D vector has rank 1, a 2D matrix has rank 2. The rank is part of the type and known at compile time. A function expecting `tensor<?x?xf32>` (rank 2) cannot receive `tensor<?xf32>` (rank 1)—type checking prevents this.

However, the *shape* (the actual sizes) can be dynamic. The `?` notation indicates dynamic dimensions whose sizes are determined at runtime. You can mix static and dynamic dimensions: `tensor<8x?xf32>` is rank-2 with the first dimension statically 8 and the second dimension dynamic.

When writing generic code, we typically make all dimensions dynamic (all `?`) to handle any size. Our SAXPY function accepts three rank-1 tensors, and the runtime (or Python bindings) ensures they have compatible sizes.

### Memory Layout Considerations

Unlike memrefs, tensors in MLIR are an abstract mathematical concept and do not have a defined memory layout (strides, offsets) at this level. They are simply multi-dimensional arrays of values. The details of how they are stored in memory—row-major, column-major, or tiled—are decided later during bufferization when they are converted to memrefs. This abstraction allows the compiler to optimize the layout based on usage patterns without the programmer needing to specify it upfront.

## 5.4 Implementing SAXPY with Tensors

Now we have all the pieces needed to implement SAXPY: `C[i] = α · A[i] + B[i]`. Unlike previous chapters where we wrote explicit loops, here we will use the **Tensor** approach. We'll construct a `linalg.generic` operation that operates on immutable tensors. This matches the implementation in `src/ir.cpp`.

### The SAXPY Algorithm

SAXPY stands for "Single-Precision A·X Plus Y". The operation is conceptually simple:

Given a scalar α (alpha) and vectors A, B of length n, compute vector C where each element is:

```
C[i] = α · A[i] + B[i]  for i = 0, 1, ..., n-1
```

### Constructing the Function Signature

Our SAXPY function takes three arguments: the scalar α and two input tensors A and B. It returns a new tensor C.

```mlir
func.func @saxpy(%alpha: f32,
                 %A: tensor<?xf32>,
                 %B: tensor<?xf32>) -> tensor<?xf32> {
  // Function body will go here
}
```

This signature reflects functional semantics:
1.  **Inputs are Tensors**: `%A` and `%B` are `tensor<?xf32>`. They are immutable views of data.
2.  **Return Value**: The function returns a `tensor<?xf32>`. It does not mutate inputs.
3.  **Dynamic Shapes**: The `?` indicates the size is unknown at compile time.

### Preparing the Output Tensor

In the tensor world, we don't "allocate" memory in the traditional sense. Instead, we create an "empty" tensor to serve as the shape blueprint for the result. This `tensor.empty` operation doesn't actually allocate memory at runtime (bufferization handles that later); it just carries shape information.

```cpp
// Get dynamic size from input A
Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
Value size = builder.create<tensor::DimOp>(loc, A, c0);

// Create empty output tensor: %empty = tensor.empty(%size)
Value empty = builder.create<tensor::EmptyOp>(
    loc,
    tensorType,      // Result type (tensor<?xf32>)
    ValueRange{size} // Dynamic sizes
);
```

### Creating the Linalg Generic Operation

Now we construct the core computation using `linalg.generic`. This operation describes the element-wise arithmetic.

```cpp
// Define indexing maps: (d0) -> (d0) for all operands
// This means: "Use index i for A, i for B, and i for Output"
auto map = AffineMap::getMultiDimIdentityMap(1, &context);
SmallVector<AffineMap> indexingMaps = {map, map, map};

// Define iterator types: ["parallel"]
// This means: "Iterations are independent"
SmallVector<utils::IteratorType> iteratorTypes = {utils::IteratorType::parallel};

// Create the generic op
auto genericOp = builder.create<linalg::GenericOp>(
    loc,
    TypeRange{tensorType},          // Result types
    ValueRange{A, B},               // Inputs (ins)
    ValueRange{empty},              // Outputs (outs)
    indexingMaps,
    iteratorTypes
);
```

### Implementing the Computation Body

The body of `linalg.generic` defines what happens for a single element. We use a lambda function (the `bodyBuilder`) to construct the operations inside the body. This is the modern C++ API style, which handles block creation and argument management automatically.

```cpp
// Create linalg.generic operation with a body builder
auto genericOp = builder.create<linalg::GenericOp>(
    loc,
    /*resultTensorTypes=*/TypeRange{dynamicTensorType},
    /*inputs=*/ValueRange{A, B},
    /*outputs=*/ValueRange{empty},
    indexingMaps,
    iteratorTypes,
    /*bodyBuilder=*/[&](OpBuilder& b, Location nestedLoc, ValueRange args) {
      // args[0] = a (from A), args[1] = b (from B), args[2] = out (unused)
      Value a = args[0];
      Value bVal = args[1];

      // Compute: scaled = alpha * a
      Value scaled = b.create<arith::MulFOp>(nestedLoc, alpha, a);

      // Compute: result = scaled + bVal
      Value sum = b.create<arith::AddFOp>(nestedLoc, scaled, bVal);

      // Yield the result
      b.create<linalg::YieldOp>(nestedLoc, sum);
    }
);
```

Finally, we return the result of the generic operation:

```cpp
builder.create<func::ReturnOp>(loc, genericOp.getResult(0));
```

This C++ code generates the clean, high-level MLIR we saw in Section 5.1. It expresses *what* to compute without getting bogged down in loops, indices, or memory pointers.

## 5.5 Lowering SCF to Control Flow

After generating SAXPY IR with the SCF dialect, we must lower it to executable code. As discussed in Chapter 3, MLIR follows progressive lowering: high-level operations lower to mid-level operations (SCF), which lower to low-level operations (CF - Control Flow), which finally lower to LLVM IR and machine code. This section examines how SCF operations transform into basic blocks and branches.

### From Structured to Unstructured Control Flow

The SCF dialect provides structured control flow: loops and conditionals with well-defined entries and exits. The CF (Control Flow) dialect provides unstructured control flow: basic blocks and branches, similar to assembly language or LLVM IR. A basic block is a sequence of operations with a single entry point (the beginning) and a single exit point (a branch at the end). Control flow between basic blocks uses explicit branches.

Converting `scf.for` to basic blocks requires transforming the loop into a series of blocks with conditional branches. A typical for loop in CF looks like this:

```mlir
// Initialization block
%i_init = ...
cf.br ^loop_header(%i_init : index)

^loop_header(%i: index):
// Check loop condition
%cond = arith.cmpi slt, %i, %upper : index
cf.cond_br %cond, ^loop_body, ^loop_exit

^loop_body:
  // Loop body operations
  %i_next = arith.addi %i, %step : index
  cf.br ^loop_header(%i_next : index)

^loop_exit:
  // Continue after loop
```

This structure makes the loop's control flow explicit. The loop header checks whether to continue iterating (i < upper), branching to the loop body if true or the exit if false. The loop body executes and then branches back to the header with the incremented induction variable. This is exactly how loops work at the assembly level.

### The SCF to CF Lowering Pass

MLIR provides a pass called `convert-scf-to-cf` that performs this transformation automatically. The pass pattern-matches `scf.for`, `scf.while`, and `scf.if` operations and replaces them with equivalent CF operations. For `scf.for`, the transformation creates:

1. An initialization block that sets up the initial induction variable value
2. A loop header block that checks the loop condition
3. A loop body block containing the original body operations
4. An exit block for code after the loop

The pass threads the induction variable through block arguments, converts loop bounds to comparisons, and ensures iter_args (loop-carried values) flow correctly through the block structure. After this pass, the SCF dialect is completely eliminated from the IR.

### Why Lower SCF at All?

You might wonder why we bother with structured control flow if we'll just convert it to branches anyway. The reason is that structure enables optimization. While SCF operations exist, the compiler can apply loop-specific transformations: unrolling, loop-invariant code motion, loop fusion, and parallelization. These transformations rely on understanding loop structure—which iterations depend on which other iterations, what values are loop-invariant, etc.

Once lowered to CF's basic blocks and branches, this structure is lost. The IR is just a control flow graph of blocks and branches. Optimizations that depend on loop structure become much harder or impossible. Therefore, MLIR keeps operations at the highest possible abstraction level for as long as possible, lowering only when necessary. In our compilation pipeline, SCF lowers to CF right before converting to LLVM dialect, at the very end of optimization.

### Comparing Linalg, SCF, and CF Lowering

We've now seen three levels of abstraction:

* **Linalg**: Declarative operations expressing *what* to compute (`linalg.matmul`, `linalg.generic`). The compiler decides *how*.

* **SCF**: Structured control flow expressing *how* to compute with explicit loops (`scf.for`). The compiler translates to unstructured form.

* **CF**: Unstructured control flow with basic blocks and branches (`cf.br`, `cf.cond_br`). This is close to assembly.

Each level lowers to the next. Linalg operations lower to SCF loops (Chapter 3 showed `linalg.matmul` lowering to triple-nested `scf.for`). SCF operations lower to CF basic blocks. CF operations lower to LLVM IR, which compiles to machine code. This progressive lowering maintains optimization opportunities at each level while gradually making execution semantics more explicit.

## 5.6 The Complete Compilation Pipeline

Let's trace SAXPY through the entire compilation pipeline, from high-level IR generation to executable machine code. This pipeline matches the implementation in `src/lowering.cpp`.

### Phase 1: IR Generation

We start by generating MLIR IR using the C++ API. The `createSaxpyModule` function (from `src/ir.cpp`) constructs the tensor-based IR:

```cpp
OwningOpRef<ModuleOp> createSaxpyModule(MLIRContext& context) {
  // Create linalg.generic
  auto genericOp = builder.create<linalg::GenericOp>(...);
  // ... build body ...
  return module;
}
```

At this stage, the IR is high-level, functional, and uses tensors.

### Phase 2: Canonicalization

Before lowering, we run canonicalization to simplify the IR.

```cpp
PassManager pm(context);
pm.addPass(createCanonicalizerPass());
```

### Phase 3: Bufferization (The Bridge)

This is the critical step where we cross from the "Tensor World" to the "MemRef World". We use One-Shot Bufferize to transform our functional tensor code into imperative memref code.

```cpp
// Configure One-Shot Bufferize
bufferization::OneShotBufferizePassOptions options;
options.bufferizeFunctionBoundaries = true; // Convert args/results
pm.addPass(bufferization::createOneShotBufferizePass(options));

// Convert return values to out-parameters
pm.addPass(bufferization::createBufferResultsToOutParamsPass());

// Finalize bufferization dialect
pm.addPass(createConvertBufferizationToMemRefPass());
```

After this phase, `linalg.generic` operates on memrefs, and the function signature has changed to accept an out-parameter.

### Phase 4: Lowering Linalg to Loops

Now that we have memrefs, we can lower the declarative `linalg.generic` operation into explicit loops.

```cpp
pm.addPass(createConvertLinalgToLoopsPass());
```

This pass generates the `scf.for` loops that we manually wrote in previous chapters. It handles the iteration logic automatically.

### Phase 5: SCF to CF Conversion

Next, we convert structured control flow (loops) to unstructured control flow (branches):

```cpp
pm.addPass(createSCFToControlFlowPass());
```

This pass eliminates `scf.for`, replacing it with basic blocks and `cf.br` / `cf.cond_br` instructions.

### Phase 6: Lowering to LLVM Dialect

Now we convert all remaining operations to the LLVM dialect:

```cpp
pm.addPass(createConvertFuncToLLVMPass());
pm.addPass(createArithToLLVMConversionPass());
pm.addPass(createConvertControlFlowToLLVMPass());
pm.addPass(createFinalizeMemRefToLLVMConversionPass());
```

### Phase 7: Translation and JIT

Finally, we translate to LLVM IR and JIT compile, just as before.

```cpp
auto engine = ExecutionEngine::create(module, options);
```

This pipeline demonstrates the power of MLIR: we write high-level, clean tensor code, and the compiler handles the complexity of bufferization, loop generation, and low-level code generation.

### Performance Characteristics

After all these transformations, what does the final code look like? On modern x86_64 CPUs, the compiled SAXPY might use AVX instructions (256-bit SIMD) to process 8 floats simultaneously. A single loop iteration would:

1. Load 8 elements from A and B (two AVX load instructions)
2. Multiply 8 elements by α (one AVX multiply instruction)
3. Add 8 elements from B (one AVX add instruction)
4. Store 8 results to C (one AVX store instruction)

This processes 8 elements in roughly the same time scalar code processes 1 element—an 8x speedup. For a vector of 10,000 elements, that's 1,250 loop iterations instead of 10,000, a significant improvement. This vectorization happens automatically through LLVM's optimization passes, demonstrating why progressive lowering matters: we write clear high-level code, and the optimizer handles low-level performance tuning.

## 5.7 Python Integration and Testing

Our SAXPY implementation is written in C++ using MLIR's APIs, but we want to call it from Python for easy testing and integration with NumPy. This requires Python bindings that marshal data between Python's numpy arrays and MLIR's memrefs. The binding layer hides the complexity of memref descriptors and provides a natural Python API.

### The Python Binding Interface

The binding code (in `bindings.cpp` from the ch.5 code) uses pybind11 to expose the compiled SAXPY function to Python:

```cpp
py::array_t<float> saxpy(float alpha,
                         const py::array_t<float>& A,
                         const py::array_t<float>& B) {
  // Allocate output
  int64_t size = A.shape(0);
  auto C = py::array_t<float>(size);
  // Call compiled function
  executeSaxpy(alpha, A.data(), B.data(), C.mutable_data(), size);
  return C;
}

PYBIND11_MODULE(ch5_vector_ops, m) {
  m.def("saxpy", &saxpy, "SAXPY operation: C = alpha * A + B");
}
```

This binding performs several important tasks. First, it validates that inputs are 1D arrays (vectors, not matrices). Second, it checks that A and B have the same length, preventing out-of-bounds accesses. Third, it automatically allocates the output array C. Fourth, it calls the JIT-compiled SAXPY function with pointers to the array data. Finally, it returns C to Python.

Using `const py::array_t<float>&` for the input arguments is a best practice. It avoids unnecessary copying of the array handle (though the handle itself is lightweight) and clearly communicates that the function does not modify the input array structure.

The user-facing API is extremely simple:

```python
import ch5_vector_ops
import numpy as np

alpha = 2.0
A = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
B = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

C = ch5_vector_ops.saxpy(alpha, A, B)
# Result: array([7., 10., 13., 16.], dtype=float32)
```

From the user's perspective, it's a simple function call: pass in three arguments, get back a result. All the complexity of memref descriptors, JIT compilation, and memory management is hidden by the binding layer.

### NumPy Compatibility

The binding uses `py::array_t<float>`, pybind11's wrapper around NumPy arrays. This wrapper provides convenient access to array metadata (shape, strides, dtype) and data pointers. When we call `A.data()`, we get a raw pointer to the array's underlying memory, which we pass to the compiled SAXPY function.

Because SAXPY writes to an out-parameter (C), we allocate C in the binding code before calling the compiled function. This allocation uses NumPy's allocator, ensuring that C is a proper NumPy array that Python can manage. When C goes out of scope in Python, NumPy's garbage collector automatically deallocates it.

An important detail: we use `dtype=np.float32` for all arrays because our MLIR function operates on f32 (32-bit floats). If you passed f64 (64-bit floats) or i32 (integers), the binding would either convert them (introducing overhead) or reject them (with a type error). For performance-critical code, using the correct dtype is essential to avoid unnecessary conversions.

### Testing and Validation

Testing SAXPY is straightforward because we can compare against NumPy's implementation:

```python
def test_saxpy():
    alpha = 2.5
    A = np.random.randn(1000).astype(np.float32)
    B = np.random.randn(1000).astype(np.float32)
    
    # MLIR implementation
    C_mlir = ch5_vector_ops.saxpy(alpha, A, B)
    
    # NumPy reference
    C_numpy = alpha * A + B
    
    # Compare results (allowing small floating-point differences)
    assert np.allclose(C_mlir, C_numpy, rtol=1e-5)
```

The test generates random input vectors, computes SAXPY with both our MLIR implementation and NumPy's arithmetic, and verifies they match within floating-point tolerance. The `np.allclose` function allows small differences due to rounding—floating-point arithmetic isn't perfectly associative, so different orderings of operations can produce slightly different results.

## 5.8 Why Linalg?

We've now implemented the operation using Linalg (implicitly, through high-level operations) which lowers to SCF (explicitly, through loops). When should you use each approach? The answer depends on your priorities: optimization potential, implementation complexity, and debugging requirements.

### Use Linalg for Well-Known Operations

Linalg shines for operations with well-studied optimization strategies. Matrix multiplication, convolutions, and reductions have decades of research behind them. The Linalg dialect encodes these operations at a high level, enabling the compiler to apply specialized optimizations. For `linalg.matmul`, the compiler can choose block sizes for cache locality, reorder loops for vectorization, and apply other domain-specific transformations.

When you write `linalg.matmul`, you're leveraging all this accumulated knowledge. The compiler handles the details, freeing you to focus on high-level model architecture. Additionally, Linalg operations compose well with other high-level transformations like fusion and tiling. If your operation fits Linalg's structured patterns (`linalg.generic` is highly flexible), Linalg is usually the right choice.

Linalg also provides better abstraction for operations where the specific loop structure doesn't matter. If you don't care whether the inner loop iterates over rows or columns (the compiler can choose based on cache effects), why specify it explicitly? Declaring the operation at a high level gives the compiler maximum freedom to optimize.

### Use SCF for Custom Control Flow

SCF becomes necessary when operations don't fit Linalg's patterns. Operations with complex control flow—multiple conditionals, irregular iteration patterns, or data-dependent branches—are difficult or impossible to express with `linalg.generic`. For these cases, explicit loops with SCF provide the control you need.

SCF is also valuable for learning and debugging. When you write explicit loops, you see exactly what executes. There's no mystery about what the compiler might do—the IR directly encodes the computation. For educational purposes or when implementing novel operations, this clarity is beneficial. You can reason about correctness directly from the IR without worrying about compiler transformations.

Additionally, SCF enables fine-grained performance tuning. If you've profiled your code and identified a bottleneck, hand-optimizing with SCF lets you control every detail: loop ordering, prefetching, manual vectorization, etc. This level of control is impossible with high-level Linalg operations. Of course, you sacrifice compiler optimizations when you do this, so it's a trade-off.

### A Hybrid Approach

In practice, production MLIR code uses both Linalg and SCF at different stages. The frontend (framework import) generates Linalg operations expressing high-level semantics. The middle-end (optimization passes) operates on Linalg, applying domain-specific transformations. The backend (lowering to machine code) converts Linalg to SCF loops, then to CF branches, then to machine instructions.

This hybrid approach gets the best of both worlds: high-level optimization where possible, low-level control where necessary. As a developer, you typically work at the highest level appropriate for your task. Implementing a transformer model? Use Linalg operations for matmuls and layer norms. Implementing a custom sparse matrix operation? Drop down to SCF for explicit iteration over non-zeros.

### The Progressive Lowering Philosophy

The relationship between Linalg and SCF exemplifies MLIR's progressive lowering philosophy. Start at the highest abstraction level that expresses your computation. Apply transformations that leverage that abstraction (fusion for Linalg, loop tiling, etc.). When no more high-level transformations apply, lower to the next level (Linalg → SCF). Apply transformations at that level (loop unrolling, vectorization). Lower again (SCF → CF), and so on until reaching machine code.

This philosophy maintains optimization opportunities at every level. Lowering too early closes off high-level optimizations. Staying too high-level prevents low-level tuning. Progressive lowering navigates this trade-off by lowering gradually, applying appropriate optimizations at each level. Understanding both Linalg and SCF prepares you to work at multiple abstraction levels as needed.

## 5.9 Summary: The Shift to Tensors

This chapter marks a pivotal shift in how we approach MLIR compilation. While the chapter content above discusses SCF and memref-based explicit control (valuable for understanding low-level mechanics), our actual implementation adopts **tensors**—the industry-standard pattern used in all modern ML compilers.

### What We Actually Implemented

Our SAXPY implementation uses tensors and automatic bufferization:

```mlir
#map = affine_map<(d0) -> (d0)>

func.func @saxpy(%alpha: f32,
                 %A: tensor<?xf32>,
                 %B: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %A, %c0 : tensor<?xf32>
  %empty = tensor.empty(%dim) : tensor<?xf32>
  
  %result = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%A, %B : tensor<?xf32>, tensor<?xf32>)
    outs(%empty : tensor<?xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %scaled = arith.mulf %alpha, %a : f32
    %sum = arith.addf %scaled, %b : f32
    linalg.yield %sum : f32
  } -> tensor<?xf32>
  
  return %result : tensor<?xf32>
}
```

**Key Differences from Old Approach:**
1. **Function returns tensor** (functional) instead of void with out-parameter (imperative)
2. **All operations on tensors** (immutable) instead of memrefs (mutable)
3. **linalg.generic with tensors** instead of scf.for with memref.load/store
4. **Bufferization happens automatically** in the compilation pipeline

### The Modern Pipeline

Our actual lowering pipeline follows the pattern from Chapter 4:

```
Tensor IR (High-Level)
    ↓ Canonicalize
Tensor IR (Optimized)
    ↓ One-Shot Bufferize
MemRef IR (Out-Parameters)
    ↓ Linalg-to-Loops
SCF Loops (memref.load/store)
    ↓ SCF-to-CF
Control Flow (cf.br)
    ↓ Convert-to-LLVM
LLVM Dialect
    ↓ JIT Compile
Native Code
```

Bufferization automatically transforms the functional signature `func @saxpy(...) -> tensor<?xf32>` into the imperative `func @saxpy(..., %out: memref<?xf32>)`. It converts `linalg.generic` operations on tensors into equivalent operations on memrefs. Finally, the Linalg-to-Loops pass produces the explicit `scf.for` loops that we studied in earlier chapters.

### Why This Matters

Adopting tensors provides significant advantages. Functional semantics enable powerful optimizations like fusion, algebraic simplification, and dead code elimination without the need for complex alias analysis. This approach aligns with industry standards, as every production ML compiler (Torch-MLIR, IREE, StableHLO) uses this pattern. It leads to cleaner code where you express *what* to compute rather than *how* to manage memory, and it ensures compatibility with frameworks like PyTorch and TensorFlow that natively think in tensors. Most importantly, it delegates memory management to the compiler, allowing you to focus on correctness while the bufferization pass handles efficiency.

### Understanding Both Levels

While we implement with tensors, understanding the lower levels (SCF, memref, explicit loops) remains valuable for debugging and performance tuning. When something goes wrong, you need to understand what the compiler generated. Sometimes you may need to inspect the lowered SCF to understand why performance isn't optimal. Advanced users may even write custom lowering patterns. However, for most development, we work at the tensor level and let bufferization handle the details.

### Key Takeaways

We have learned to write tensor operations at a high level, leveraging functional semantics for cleaner and more optimizable code. We've seen how bufferization automatically transforms this into efficient memref code, bridging the gap between abstraction and execution. We understand that while we develop with tensors, the underlying execution model relies on memrefs and loops, and understanding both levels is key to mastery. This progressive lowering strategy allows us to apply appropriate optimizations at each stage, from high-level fusion to low-level vectorization.

With the tensor abstraction mastered, we're ready to tackle more complex operations. The next chapter introduces softmax—a fundamental neural network operation requiring reductions, exponentials, and careful numerics—all implemented using tensor operations and automatic bufferization.