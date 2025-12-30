# Chapter 5: Tensor-First Architecture - Modern MLIR Patterns

In the first four chapters, we built foundational knowledge of MLIR's compilation model. Chapter 1-3 used memrefs directly to understand execution mechanics, memory layout, and JIT compilation. Chapter 4 introduced bufferization—the transformation from functional tensor IR to imperative memref IR—and explained why modern MLIR compilers separate these concerns. That chapter was the pivot point. Now, starting from Chapter 5, we adopt **tensor-first architecture**: the industry-standard pattern used in production MLIR systems worldwide.

This architectural shift reflects how real ML compilers work. Torch-MLIR (PyTorch's MLIR backend), IREE (Google's ML runtime), StableHLO (TensorFlow/JAX), and other production systems all follow the same pattern: high-level operations use immutable tensors, optimization happens at the tensor level, and bufferization converts to executable memref code late in the compilation pipeline. By adopting this pattern, we're not just learning MLIR—we're learning the patterns that power modern AI frameworks.

Our vehicle for demonstrating tensor-first architecture is **SAXPY** (Single-Precision A·X Plus Y), a fundamental operation from linear algebra: `C[i] = α · A[i] + B[i]`. SAXPY is simpler than matrix multiplication but rich enough to demonstrate tensor operations, dynamic shapes, the Linalg dialect, and the complete bufferization pipeline. We'll implement SAXPY using `linalg.generic` with tensor types, then watch as bufferization automatically transforms our functional code into efficient imperative machine code.

By the end of this chapter, you'll understand:
- Why tensors are the right abstraction for high-level ML operations
- How `linalg.generic` expresses parallel computations on tensors
- How the tensor dialect (`tensor.empty`, `tensor.dim`) handles dynamic shapes
- How One-Shot Bufferize converts functional tensors to imperative memrefs
- How the complete pipeline (canonicalization → bufferization → loop lowering → LLVM) produces executable code
- When to use tensor-first patterns versus direct memref operations

This knowledge forms the foundation for all subsequent chapters. From Chapter 5 onward, every operation we implement—softmax, ReLU, convolutions, attention, transformers—will use tensor-first architecture.

## 5.1 The Tensor-First Philosophy: Why Immutable Matters

Modern ML frameworks think in terms of immutable data. When you write `result = alpha * A + B` in PyTorch, you're not mutating A or B—you're creating a new tensor that holds the result. This functional semantics isn't just convenient for users; it enables powerful compiler optimizations that would be impossible with mutable operations.

### Functional vs Imperative: A Tale of Two Styles

Consider two ways to express the same SAXPY computation:

**Imperative Style (Memref - Chapters 1-4):**
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

This imperative style explicitly:
- Takes a pre-allocated output buffer C as a parameter
- Loops over elements with explicit load/store operations
- Mutates C in-place (side effects)
- Requires understanding memory layout, iteration order, and aliasing

**Functional Style (Tensor - Chapter 5+):**
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

This functional style:
- Returns a new tensor (no mutation of inputs)
- Uses declarative `linalg.generic` (compiler chooses iteration strategy)
- No explicit load/store (abstracted away)
- Immutable semantics enable optimization

### Why Immutability Wins

The power of immutability becomes clear when you consider optimization. Suppose you have two operations:

```mlir
%tmp = some_op1 ins(%A : tensor<?xf32>) -> tensor<?xf32>
%result = some_op2 ins(%tmp : tensor<?xf32>) -> tensor<?xf32>
```

Because tensors are immutable, the compiler knows:
1. **No aliasing**: %tmp cannot be an alias of %A (they're different tensor values)
2. **No side effects**: some_op1 doesn't mutate any global state
3. **Dead value elimination**: If %tmp is only used once, it might be eliminated
4. **Fusion opportunity**: The compiler can fuse op1 and op2 into a single operation, eliminating the intermediate tensor allocation entirely

With memrefs, these optimizations require complex alias analysis:

```mlir
some_op1 ins(%A : memref<?xf32>) outs(%tmp : memref<?xf32>)
some_op2 ins(%tmp : memref<?xf32>) outs(%result : memref<?xf32>)
```

The compiler must prove:
- Does %tmp alias %A or %result? (Requires pointer analysis)
- Could some_op1 modify %A through aliasing? (Side effect analysis)
- Is it safe to eliminate %tmp? (Liveness analysis)

Alias analysis is notoriously difficult—it's why C/C++ compilers struggle to optimize code with pointers. By making immutability explicit at the type level (tensor vs memref), MLIR enables optimizations without complex analysis.

### The Industry Consensus

Every major ML compiler uses tensor-first architecture:

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

This universal adoption isn't coincidence—tensor-first architecture is simply the right abstraction level for ML compilation.

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
%rows = memref.dim %matrix, %c0 : memref<?x?xf32>  // Get row count
%cols = memref.dim %matrix, %c1 : memref<?x?xf32>  // Get column count
```

This operation is crucial for writing dimension-agnostic code. When your function receives a `memref<?xf32>` (dynamic 1D vector), you don't know its size at compile time. The caller might pass a vector of length 100 or 10,000. To iterate over all elements, you must query the size:

```mlir
%size = memref.dim %vector, %c0 : memref<?xf32>
scf.for %i = %c0 to %size step %c1 {
  // Iterate over all elements
}
```

The `memref.dim` operation lowers to simple field access in the memref descriptor. Recall from Chapter 2 that memrefs are descriptors containing the base pointer, offset, sizes, and strides. The `memref.dim` operation just extracts the size field for the specified dimension. At runtime, this is a single memory load—extremely cheap. There's no computation involved; the information already exists in the descriptor.

For statically-shaped memrefs (like `memref<8x16xf32>`), the compiler can often optimize away `memref.dim` entirely. If you query the size of dimension 0 on a `memref<8x16xf32>`, the compiler knows the answer is 8 at compile time and replaces the operation with a constant. This means you can write code that queries dimensions generically, and the compiler will optimize it appropriately for both static and dynamic shapes.

### Rank and Shape Constraints

Memrefs have a *rank*: the number of dimensions. A 1D vector has rank 1, a 2D matrix has rank 2, and a 3D tensor has rank 3. The rank is part of the type and known at compile time. A function expecting `memref<?x?xf32>` (rank 2) cannot receive `memref<?xf32>` (rank 1)—type checking prevents this at compile time.

However, the *shape* (the actual sizes of dimensions) can be dynamic. A rank-2 memref might be 8×16 or 1024×2048; both are rank-2 but different shapes. The `?` notation indicates dynamic dimensions whose sizes are determined at runtime. You can mix static and dynamic dimensions: `memref<8x?xf32>` is rank-2 with the first dimension statically 8 and the second dimension dynamic.

When writing generic code, you typically make all dimensions dynamic (all `?`) to handle any size. The only constraint is rank. Our SAXPY function accepts three rank-1 memrefs (vectors), and the Python binding layer validates that all three have the same size at runtime. The MLIR IR doesn't encode size equality—that's a runtime constraint checked by the calling code.

### Memory Layout Considerations

As discussed in Chapter 4, memrefs have layout maps that control how logical indices map to physical memory offsets. The `memref.dim` operation returns logical dimension sizes, not physical strides. For most memrefs, the default identity layout means logical and physical sizes coincide. But with permuted or strided layouts, the physical memory layout might differ from the logical shape.

For our SAXPY implementation, we use default layouts (identity maps), so the distinction doesn't matter. But when working with transposed or tiled memrefs, remember that `memref.dim` gives you logical sizes. To understand physical layout, you'd need to examine the memref's stride attributes or use the `memref.stride` operation (for extracting stride information).

## 5.4 Implementing SAXPY: Explicit Loop Construction

Now we have all the pieces needed to implement SAXPY: `C[i] = α · A[i] + B[i]`. We'll construct this operation explicitly using `scf.for` for the loop, `memref.load` and `memref.store` for element access, and `arith.mulf` and `arith.addf` for arithmetic. This implementation demonstrates the low-level control that SCF provides compared to high-level Linalg operations.

### The SAXPY Algorithm

SAXPY stands for "Single-Precision A·X Plus Y" from BLAS (Basic Linear Algebra Subprograms), the standard library for linear algebra operations. We'll implement it for single precision (f32) as the name suggests. The operation is conceptually simple:

Given a scalar α (alpha) and vectors A, B of length n, compute vector C where each element is:

```
C[i] = α · A[i] + B[i]  for i = 0, 1, ..., n-1
```

This is an element-wise operation: each output element depends only on the corresponding input elements at the same index. There are no dependencies between different indices, meaning iterations could potentially run in parallel. This property makes SAXPY a good candidate for vectorization and parallelization, though our implementation will be serial for clarity.

The algorithm's computational structure is straightforward. For each index i from 0 to n-1, we load A[i] and B[i] from memory, multiply A[i] by α, add B[i] to the result, and store the result to C[i]. This produces five operations per element: two loads, one multiply, one add, one store. With n elements, the total operation count is 5n.

### Constructing the Function Signature

Our SAXPY function takes four arguments: the scalar α and three vectors A, B, C. The scalar is type `f32` (32-bit floating-point), and the vectors are rank-1 dynamic memrefs `memref<?xf32>`. The function returns nothing; C is an out-parameter that receives the result.

```mlir
func.func @saxpy(%alpha: f32,
                 %A: memref<?xf32>,
                 %B: memref<?xf32>,
                 %C: memref<?xf32>) {
  // Function body will go here
}
```

This signature makes several design decisions explicit. First, α is passed by value as a scalar, not through a memref. Scalars are small enough that passing them directly is more efficient than pointer indirection. Second, all three vectors are memrefs, meaning they're memory buffers that the caller allocates. The function doesn't allocate C; it writes to caller-provided storage. This matches the out-parameter pattern we discussed in Chapter 4.

Third, all three memrefs have dynamic dimensions (`?`), meaning the function works with vectors of any length. The caller must ensure that A, B, and C have the same length (otherwise accessing elements would go out of bounds), but the function signature doesn't encode this constraint. Runtime validation happens in the Python bindings before calling the compiled function.

### Building the Loop Structure

Inside the function body, we first need to determine how many iterations the loop requires. Since the vectors have dynamic size, we query the dimension:

```mlir
%c0 = arith.constant 0 : index
%size = memref.dim %A, %c0 : memref<?xf32>
%c1 = arith.constant 1 : index
```

We create two constants: `%c0` (zero) for the lower bound and dimension index, and `%c1` (one) for the loop step. Then we query the size of A's first (and only) dimension. This size determines how many iterations we need.

With the size known, we construct the loop:

```mlir
scf.for %i = %c0 to %size step %c1 {
  // Loop body executes for %i = 0, 1, 2, ..., size-1
}
```

The induction variable `%i` represents the current index. On each iteration, `%i` increments by one (the step) until it reaches `%size`. The loop body executes with `%i` taking each value from 0 to size-1, giving us access to every element of the vectors.

### Implementing the Loop Body

Inside the loop, we implement the SAXPY computation for a single element. First, we load the input values:

```mlir
%a = memref.load %A[%i] : memref<?xf32>
%b = memref.load %B[%i] : memref<?xf32>
```

The `memref.load` operation reads a single element from a memref at the specified index. We provide `%i` as the index, so on the first iteration (i=0) we load A[0] and B[0], on the second iteration (i=1) we load A[1] and B[1], and so on. The operation returns an f32 value representing the element's value.

Next, we perform the arithmetic:

```mlir
%scaled = arith.mulf %alpha, %a : f32
%result = arith.addf %scaled, %b : f32
```

First, we multiply α by the loaded value from A, producing a scaled value. Then we add the value from B to this scaled value. The types on all operations are f32, and MLIR verifies type correctness. If you tried to add an f32 to an f64, the compiler would reject the IR.

Finally, we store the result:

```mlir
memref.store %result, %C[%i] : memref<?xf32>
```

The `memref.store` operation writes a value to a memref at a specified index. We store `%result` to C[i], completing the computation for this element. After the loop completes, C contains the SAXPY result for all elements.

### The Complete MLIR IR

Putting it all together, here's the complete SAXPY function in MLIR:

```mlir
func.func @saxpy(%alpha: f32,
                 %A: memref<?xf32>,
                 %B: memref<?xf32>,
                 %C: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %size = memref.dim %A, %c0 : memref<?xf32>
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %size step %c1 {
    %a = memref.load %A[%i] : memref<?xf32>
    %b = memref.load %B[%i] : memref<?xf32>
    %scaled = arith.mulf %alpha, %a : f32
    %result = arith.addf %scaled, %b : f32
    memref.store %result, %C[%i] : memref<?xf32>
  }
  return
}
```

This IR is explicit and self-documenting. Reading it, you can trace exactly what happens: create constants, query size, loop from 0 to size, load two elements, multiply one by alpha, add them, store the result. No guessing about compiler decisions—the control flow and data flow are completely specified.

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

**Linalg**: Declarative operations expressing *what* to compute (`linalg.matmul`, `linalg.generic`). The compiler decides *how*.

**SCF**: Structured control flow expressing *how* to compute with explicit loops (`scf.for`). The compiler translates to unstructured form.

**CF**: Unstructured control flow with basic blocks and branches (`cf.br`, `cf.cond_br`). This is close to assembly.

Each level lowers to the next. Linalg operations lower to SCF loops (Chapter 3 showed `linalg.matmul` lowering to triple-nested `scf.for`). SCF operations lower to CF basic blocks. CF operations lower to LLVM IR, which compiles to machine code. This progressive lowering maintains optimization opportunities at each level while gradually making execution semantics more explicit.

## 5.6 The Complete Compilation Pipeline

Let's trace SAXPY through the entire compilation pipeline, from high-level IR generation to executable machine code. Understanding this end-to-end process reinforces how all the pieces fit together: IR generation, pass management, progressive lowering, and JIT compilation.

### Phase 1: IR Generation

We start by generating MLIR IR using the C++ API. The `createSaxpyModule` function (from the ch.5 code) constructs the IR we examined in section 5.4. This function uses builder APIs to create operations, regions, and blocks:

```cpp
OwningOpRef<ModuleOp> createSaxpyModule(MLIRContext& context) {
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module.getBody());
  
  // Define types
  auto f32Type = builder.getF32Type();
  auto dynamicMemRefType = MemRefType::get({ShapedType::kDynamic}, f32Type);
  
  // Function type: (f32, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  auto funcType = builder.getFunctionType(
    {f32Type, dynamicMemRefType, dynamicMemRefType, dynamicMemRefType},
    {}
  );
  
  // Create function
  auto funcOp = builder.create<func::FuncOp>(loc, "saxpy", funcType);
  funcOp.setPublic();
  
  // Create function body
  auto& entryBlock = *funcOp.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);
  
  // Get function arguments
  Value alpha = entryBlock.getArgument(0);  // f32
  Value A = entryBlock.getArgument(1);      // memref<?xf32>
  Value B = entryBlock.getArgument(2);      // memref<?xf32>
  Value C = entryBlock.getArgument(3);      // memref<?xf32>
  
  // Create constants
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  
  // Get dynamic size: %size = memref.dim %A, %c0
  Value size = builder.create<memref::DimOp>(loc, A, c0);
  
  // Create scf.for loop: for i = 0 to size step 1
  auto forOp = builder.create<scf::ForOp>(loc, c0, size, c1);
  
  // Build loop body
  builder.setInsertionPointToStart(forOp.getBody());
  Value i = forOp.getInductionVar();
  
  // Load A[i] and B[i]
  Value a = builder.create<memref::LoadOp>(loc, A, ValueRange{i});
  Value b = builder.create<memref::LoadOp>(loc, B, ValueRange{i});
  
  // Compute: scaled = alpha * a
  Value scaled = builder.create<arith::MulFOp>(loc, alpha, a);
  
  // Compute: result = scaled + b
  Value result = builder.create<arith::AddFOp>(loc, scaled, b);
  
  // Store result to C[i]
  builder.create<memref::StoreOp>(loc, result, C, ValueRange{i});
  
  // Return to function level
  builder.setInsertionPointAfter(forOp);
  builder.create<func::ReturnOp>(loc);
  return module;
}
```

The generated IR contains SCF operations (`scf.for`), arithmetic operations (`arith.mulf`, `arith.addf`), and memref operations (`memref.load`, `memref.store`, `memref.dim`). At this stage, the IR is high-level and clearly expresses the SAXPY algorithm's structure.

### Phase 2: Canonicalization

Before lowering, we run canonicalization to simplify the IR. Canonicalization applies algebraic simplifications, constant folding, and dead code elimination. For SAXPY, there isn't much to simplify—the IR is already fairly lean. But canonicalization might eliminate unused constants or simplify index arithmetic if we had more complex indexing expressions.

```cpp
PassManager pm(context);
pm.addPass(createCanonicalizerPass());
```

Canonicalization is a general cleanup pass that runs multiple times throughout compilation. It's cheap and effective, so we run it liberally.

### Phase 3: SCF to CF Conversion

Next, we convert structured control flow to unstructured control flow:

```cpp
pm.addPass(createSCFToControlFlowPass());
```

This pass eliminates all `scf.for` operations, replacing them with basic blocks and branches. After this pass, the IR contains only CF dialect operations. The loop structure is implicit in the control flow graph rather than explicit in the operation types.

### Phase 4: Lowering to LLVM Dialect

Now we convert all remaining high-level operations to LLVM dialect operations:

```cpp
pm.addPass(memref::createExpandStridedMetadataPass());
pm.addPass(createFinalizeMemRefToLLVMConversionPass());
pm.addPass(createConvertFuncToLLVMPass());
pm.addPass(createArithToLLVMConversionPass());
pm.addPass(createConvertControlFlowToLLVMPass());
pm.addPass(createReconcileUnrealizedCastsPass());
```

Each pass handles one dialect. The memref-to-LLVM pass converts `memref.load` and `memref.store` to LLVM pointer arithmetic. The arith-to-LLVM pass converts `arith.addf` and `arith.mulf` to LLVM's `llvm.fadd` and `llvm.fmul`. The CF-to-LLVM pass converts `cf.br` to `llvm.br`. After these passes, the entire IR is in LLVM dialect.

### Phase 5: Translation to LLVM IR

With the IR fully in LLVM dialect, we translate it to LLVM IR (the intermediate representation used by the LLVM compiler):

```cpp
llvm::LLVMContext llvmContext;
auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
```

This translation is straightforward because MLIR's LLVM dialect operations have direct LLVM IR equivalents. The LLVM IR is textually similar but uses LLVM's syntax and type system instead of MLIR's.

### Phase 6: LLVM Optimization

LLVM IR goes through LLVM's optimization passes:

```cpp
llvm::PassManager llvmPM;
llvm::PassManagerBuilder builder;
builder.OptLevel = 3;  // -O3 optimization level
builder.populateModulePassManager(llvmPM);
llvmPM.run(*llvmModule);
```

LLVM's optimizer is mature and powerful. It performs loop unrolling, vectorization (converting scalar operations to SIMD instructions), instruction scheduling, register allocation, and many other optimizations. For SAXPY, LLVM might vectorize the loop, processing multiple elements per iteration using SIMD instructions like AVX on x86_64.

### Phase 7: JIT Compilation

Finally, we use MLIR's ExecutionEngine (wrapping LLVM's JIT compiler) to compile to machine code:

```cpp
auto engine = ExecutionEngine::create(module, options);
auto funcPtr = engine->lookup("saxpy");
```

The JIT compiler translates LLVM IR to native assembly instructions (x86_64, ARM64, etc.) and stores them in memory. We get a function pointer that we can call directly from C++ or through Python bindings.

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
                         py::array_t<float> A,
                         py::array_t<float> B) {
  // Validate inputs
  if (A.ndim() != 1 || B.ndim() != 1) {
    throw std::runtime_error("Inputs must be 1D arrays");
  }
  
  if (A.shape(0) != B.shape(0)) {
    throw std::runtime_error("Arrays must have same length");
  }
  
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

## 5.8 When to Use SCF vs Linalg

We've now implemented the same operation (element-wise computation) using both Linalg (implicitly, through high-level operations) and SCF (explicitly, through hand-written loops). When should you use each approach? The answer depends on your priorities: optimization potential, implementation complexity, and debugging requirements.

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

## 5.9 Summary: The Tensor-First Transformation

This chapter marks a pivotal shift in how we approach MLIR compilation. While the chapter content above discusses SCF and memref-based explicit control (valuable for understanding low-level mechanics), our actual implementation adopts **tensor-first architecture**—the industry-standard pattern used in all modern ML compilers.

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

**Bufferization automatically transforms:**
- `func @saxpy(...) -> tensor<?xf32>` → `func @saxpy(..., %out: memref<?xf32>)`
- `linalg.generic` with tensors → `linalg.generic` with memrefs
- Then Linalg-to-Loops produces the `scf.for` loops you learned about earlier

### Why This Matters

The tensor-first approach provides:

1. **Better Optimization**: Functional semantics enable fusion, algebraic simplification, and dead code elimination without complex alias analysis.

2. **Industry Alignment**: Every production ML compiler (Torch-MLIR, IREE, StableHLO) uses this pattern. You're learning real-world practices.

3. **Cleaner Code**: Writing `%result = operation(%input)` is clearer than pre-allocating buffers and mutating them.

4. **Framework Compatibility**: PyTorch, TensorFlow, and JAX all think in tensors. Tensor-first MLIR matches their semantics naturally.

5. **Automatic Memory Management**: The compiler (via bufferization) decides when to allocate, when to reuse buffers, when to copy. You focus on correctness; the compiler handles efficiency.

### Understanding Both Levels

While we implement with tensors, understanding the lower levels (SCF, memref, explicit loops) remains valuable:

- **Debugging**: When something goes wrong, you need to understand what the compiler generated
- **Performance Tuning**: Sometimes you inspect the lowered SCF to understand why performance isn't optimal
- **Custom Lowering**: Advanced users may write custom lowering patterns that produce specific SCF structures
- **Educational Value**: Understanding explicit loops helps you reason about what high-level operations actually do

The chapter content above explains these lower levels because they're foundational knowledge. But in practice, starting from Chapter 5, we work at the tensor level and let bufferization handle the details.

### The Path Forward

From this chapter onward, every operation follows the tensor-first pattern:

- **Chapter 6 (Softmax)**: Tensor operations with reductions
- **Chapter 7 (Neural Ops)**: ReLU, Conv2D with tensors
- **Chapter 8 (Custom Dialect)**: Define tensor-based operations in C++
- **Chapter 9 (TableGen)**: Generate tensor operations from specs
- **Chapter 10 (Optimizations)**: Optimize at tensor level, then at memref level
- **Chapters 11-14 (Transformers/GPT)**: Complete tensor pipelines

This is the modern way to build ML compilers with MLIR. Chapter 4 taught you bufferization theory. Chapter 5 puts it into practice. Now every subsequent chapter builds on this foundation.

### Key Takeaways

1. **Write tensor operations** at high level (functional, clean, optimizable)
2. **Bufferization transforms automatically** to efficient memref code
3. **Understand both levels**: tensors for development, memrefs/SCF for understanding execution
4. **Industry standard**: This pattern powers PyTorch, TensorFlow/JAX, and other frameworks
5. **Progressive lowering**: Each stage (tensor → memref → loops → branches → LLVM) enables appropriate optimizations

With tensor-first architecture mastered, we're ready to tackle more complex operations. The next chapter introduces softmax—a fundamental neural network operation requiring reductions, exponentials, and careful numerics—all implemented using tensor operations and automatic bufferization.