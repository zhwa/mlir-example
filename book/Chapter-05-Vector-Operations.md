# Chapter 5: Structured Control Flow and Explicit Loops

In the first four chapters, we relied on high-level operations from the Linalg dialect to express our computations. Operations like `linalg.matmul` and `linalg.fill` are declarative—they specify *what* to compute without detailing *how* the computation should proceed. In Chapter 3, we introduced progressive lowering and mentioned that Linalg operations eventually lower to explicit loops (the SCF dialect). This chapter reveals how that transformation works and when you might want to work with explicit loops directly.

The compiler decides how to implement these operations, choosing loop orders, vectorization strategies, and memory access patterns. This abstraction is powerful for optimization but sometimes limits control.

This chapter introduces a different approach: **explicit control flow** using MLIR's SCF (Structured Control Flow) dialect. Instead of declaring "compute matrix multiplication," we'll write explicit loops that specify exactly how computation proceeds, element by element, iteration by iteration. This lower-level control becomes essential when implementing operations that don't fit neatly into high-level abstractions or when you need precise control over execution for performance tuning.

Our vehicle for learning is **SAXPY** (Single-Precision A·X Plus Y), a fundamental operation from linear algebra: `C[i] = α · A[i] + B[i]` for all elements i. SAXPY is simpler than matrix multiplication but rich enough to demonstrate explicit loop construction, element-wise operations, and the SCF dialect's capabilities. By the end of this chapter, you'll understand when to use high-level abstractions versus explicit control flow, and you'll be comfortable constructing loops directly in MLIR IR.

## 5.1 The SCF Dialect: Structured Control Flow in MLIR

Control flow is how programs make decisions and repeat operations. Every programming language has control flow constructs: C has `for` and `while` loops, Python has `for` and `while` loops with `if/else` statements, and assembly has conditional and unconditional branches. MLIR provides control flow at multiple abstraction levels, and the SCF (Structured Control Flow) dialect occupies the middle ground between high-level declarative operations and low-level branches.

### What is Structured Control Flow?

Structured control flow refers to control constructs that have well-defined entry and exit points, following a hierarchical structure. A `for` loop in C is structured: execution enters at the loop header, iterates through the body, and exits when the condition fails. There's no jumping arbitrarily to different parts of the program with `goto` statements. Structured control flow makes programs easier to reason about, optimize, and verify for correctness.

In contrast, unstructured control flow uses explicit branches and labels, as found in assembly language or MLIR's CF (Control Flow) dialect. With unstructured flow, you can jump to any label at any time, creating complex control graphs that are harder to analyze. Compilers prefer structured control flow during optimization phases because the structure provides guarantees that enable transformations like loop unrolling, fusion, and parallelization.

MLIR's SCF dialect provides structured control operations that closely resemble high-level language constructs. The three primary operations are `scf.for` (counted loops), `scf.while` (condition-based loops), and `scf.if` (conditional branches). These operations maintain structured control flow properties while operating at a lower level than Linalg's declarative operations. When you write a `scf.for` loop, you're explicitly stating "iterate from this starting value to this ending value, executing this body on each iteration," which gives you fine-grained control over computation.

### The scf.for Operation

The `scf.for` operation implements a counted loop, similar to C's `for(int i = 0; i < n; i++)` or Python's `for i in range(n)`. The operation takes three operands: a lower bound, an upper bound, and a step size. The loop body executes repeatedly with an induction variable that starts at the lower bound and increments by the step size until reaching (but not including) the upper bound.

Here's what a `scf.for` loop looks like in MLIR:

```mlir
%c0 = arith.constant 0 : index
%c10 = arith.constant 10 : index
%c1 = arith.constant 1 : index

scf.for %i = %c0 to %c10 step %c1 {
  // Loop body executes with %i = 0, 1, 2, ..., 9
  // ... operations using %i ...
}
```

The loop executes ten times with the induction variable `%i` taking values from 0 through 9. The `index` type is MLIR's type for array indices and loop bounds—it's typically a machine word size (32-bit or 64-bit depending on the target platform). Notice that the upper bound is exclusive: the loop runs while `%i < %c10`, not `%i <= %c10`. This matches Python's `range()` semantics and C's typical `i < n` loop conditions.

The loop body is a region—a block of operations that can reference the induction variable. Everything inside the curly braces executes on each iteration. Importantly, `scf.for` maintains SSA (Static Single Assignment) form: the induction variable `%i` is defined once and has a different value on each iteration, but within a single iteration, it's immutable. This property is crucial for enabling optimizations like loop parallelization, where iterations can potentially execute in any order or concurrently.

One powerful feature of `scf.for` is that it can carry values across iterations, implementing loop-carried dependencies. If a computation in iteration i needs a result from iteration i-1 (like accumulating a sum), `scf.for` supports this through iter_args:

```mlir
%sum = scf.for %i = %c0 to %c10 step %c1 
    iter_args(%acc = %initial_value) -> (f32) {
  // Use %acc (value from previous iteration)
  %new_acc = arith.addf %acc, %some_value : f32
  // Yield new value for next iteration
  scf.yield %new_acc : f32
}
// After loop, %sum contains the final accumulated value
```

This pattern implements a reduction: accumulating values across all iterations. The `iter_args` specify values that flow from one iteration to the next, and `scf.yield` produces the values for the next iteration (or returns them after the loop completes). This is more sophisticated than simple counted loops and enables expressing complex computations while maintaining structured control flow.

### Why SCF Instead of Linalg?

You might wonder why we need explicit loops when Linalg already handles operations like element-wise addition and multiplication. The answer lies in the trade-off between abstraction and control. Linalg operations are opaque boxes: you specify *what* to compute, and the compiler decides *how*. This is excellent for operations with well-studied optimization strategies (matrix multiplication, convolutions) where decades of research have established best practices. The compiler can apply these optimizations automatically.

However, not all operations fit neatly into Linalg's structured patterns. Custom operations, irregular computations, or operations where you need specific execution ordering may not map well to `linalg.generic`. Additionally, when learning or debugging, explicit control flow makes execution semantics crystal clear. There's no guessing about what the compiler might do—the IR directly expresses the computation sequence.

Consider the SAXPY operation we'll implement: `C[i] = α · A[i] + B[i]`. We could express this with `linalg.generic`, specifying an iterator and a computation body. But writing it explicitly with `scf.for` teaches us how loops work in MLIR, how to load and store individual elements, and how arithmetic operations compose. Later, when we implement more complex operations like softmax (Chapter 6) or attention mechanisms (Chapter 11), this knowledge of explicit control flow becomes essential.

In practice, production MLIR code uses both approaches. High-level framework code (PyTorch, TensorFlow frontends) generates Linalg operations. The compiler optimizes these, potentially lowering some to explicit loops during transformations. Eventually, all high-level operations lower to explicit loops (SCF), then to unstructured control flow (CF), and finally to machine instructions. By understanding both Linalg and SCF, you understand the full compilation pipeline.

## 5.2 The Arith Dialect: Arithmetic and Comparison Operations

Before we can implement SAXPY, we need operations for basic arithmetic: multiplication, addition, and constants. MLIR's Arith (Arithmetic) dialect provides these fundamental operations. While arithmetic might seem trivial—every language has `+` and `*`—MLIR's approach is more sophisticated, handling different numeric types, signedness, and overflow semantics explicitly.

### Integer and Floating-Point Operations

The Arith dialect separates integer and floating-point arithmetic because they have fundamentally different semantics. Floating-point operations follow IEEE 754 standards, handling infinities, NaNs (Not-a-Number), and approximate arithmetic. Integer operations are exact but must handle overflow and signedness. This separation makes numeric semantics explicit and enables target-specific optimizations.

For floating-point operations, the dialect provides operations like:

- `arith.addf`: Floating-point addition (`a + b`)
- `arith.subf`: Floating-point subtraction (`a - b`)
- `arith.mulf`: Floating-point multiplication (`a * b`)
- `arith.divf`: Floating-point division (`a / b`)
- `arith.negf`: Floating-point negation (`-a`)

Each operation takes values of the same floating-point type (f16, f32, f64, etc.) and produces a result of that type. The operations follow IEEE 754 semantics: adding infinity to a finite number yields infinity, dividing by zero yields infinity, and operations on NaN propagate NaN. MLIR doesn't hide these details—they're part of the operation's specification.

For integer operations, the dialect provides separate signed and unsigned variants where semantics differ:

- `arith.addi`: Integer addition (works for both signed and unsigned)
- `arith.muli`: Integer multiplication (works for both signed and unsigned)
- `arith.divsi`: Signed integer division (rounds toward zero)
- `arith.divui`: Unsigned integer division
- `arith.remsi`: Signed remainder (modulo operation)
- `arith.remui`: Unsigned remainder

The separation between signed and unsigned division matters because they behave differently with negative numbers. Signed division of `-7 / 2` yields `-3` (rounding toward zero), while the bit pattern interpreted as unsigned would give a completely different result. By making this explicit, MLIR ensures that lowering to machine instructions chooses the correct assembly operation (`idiv` vs `div` on x86).

### Constants and Index Operations

Creating constant values requires the `arith.constant` operation, which takes a value and a type:

```mlir
%zero = arith.constant 0.0 : f32
%alpha = arith.constant 2.5 : f32
%one = arith.constant 1 : i32
```

Constants are SSA values like any other, meaning they're defined once and immutable. The constant's value is embedded in the IR as an attribute (compile-time data), not computed at runtime. During optimization, the compiler can perform constant folding: if you multiply two constants, the compiler replaces them with the result constant, eliminating runtime computation.

For array indexing and loop bounds, MLIR uses the `index` type, which represents machine-word-sized integers (32-bit on 32-bit systems, 64-bit on 64-bit systems). Index operations ensure that array indices and sizes use the correct width for the target architecture. The operation `arith.constant` can create index constants:

```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
```

MLIR also provides convenience operations for common index constants: `arith.ConstantIndexOp` is a builder-level operation that creates index constants directly, though in textual IR you'll see `arith.constant` with type `index`. The distinction between index and regular integers matters because some targets have specialized addressing modes or restrictions on index operations.

### Comparison and Boolean Operations

The Arith dialect also provides comparison operations that produce boolean values (type `i1`, a 1-bit integer). These operations compare two values and yield true (1) or false (0):

```mlir
%cmp = arith.cmpf olt, %a, %b : f32
// %cmp is true if %a < %b (ordered less-than)
```

The `arith.cmpf` operation compares floating-point values with a predicate (olt = ordered less-than, oge = ordered greater-or-equal, etc.). The "ordered" terminology comes from IEEE 754: ordered comparisons return false if either operand is NaN, while unordered comparisons return true if either operand is NaN. This distinction matters for robust numerical code.

Integer comparisons use `arith.cmpi` with predicates like `slt` (signed less-than), `ult` (unsigned less-than), `eq` (equal), and `ne` (not equal). Again, the separation between signed and unsigned comparisons ensures correct semantics. Comparing `-1` as a signed integer is less than `1`, but the same bit pattern interpreted as unsigned is a very large positive number, greater than `1`.

Boolean operations like `arith.andi` (logical AND), `arith.ori` (logical OR), and `arith.xori` (logical XOR) combine boolean values, enabling complex conditional logic. These operations work on any integer type but are most commonly used with `i1` for boolean logic.

### Why Explicit Arithmetic Matters

You might think arithmetic operations are too low-level to worry about—just let the compiler handle them. But explicit arithmetic operations in MLIR serve several purposes. First, they make numeric semantics precise and auditable. When you see `arith.mulf`, you know it's IEEE 754 floating-point multiplication with all the associated semantics. Second, they enable target-specific optimizations. A compiler can recognize sequences of arithmetic operations and replace them with specialized instructions (fused multiply-add, SIMD operations, etc.). Third, they maintain the progressive lowering philosophy: high-level operations lower to arithmetic operations, which eventually lower to machine instructions.

## 5.3 The MemRef Dialect: Dynamic Dimensions and Runtime Queries

We've worked with memrefs extensively in previous chapters, but always with the assumption that we knew dimensions at IR construction time (even if they were dynamic `?` placeholders). Now we need to query memref dimensions at runtime because our SAXPY implementation must work with vectors of any length. The MemRef dialect provides operations for querying shape information at runtime, which is essential for shape-agnostic code.

### The memref.dim Operation

The `memref.dim` operation extracts the size of a specific dimension from a memref at runtime. Its signature is:

```mlir
%size = memref.dim %memref, %dimension : memref<?x?xf32>
```

The operation takes two operands: the memref to query and the dimension index (0 for first dimension, 1 for second, etc.). It returns an `index` value representing the size of that dimension. For a memref of type `memref<?x?xf32>` (dynamic 2D matrix), you'd query both dimensions:

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

SAXPY stands for "Single-Precision A·X Plus Y" from BLAS (Basic Linear Algebra Subprograms), the standard library for linear algebra operations. Despite its name, we'll implement it for any floating-point precision, not just single precision (f32). The operation is conceptually simple:

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
  auto module = ModuleOp::create(builder.getUnknownLoc());
  
  // ... build function, loop, and body operations ...
  
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
pm.addPass(createConvertSCFToCFPass());
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

For performance testing, we can benchmark execution time:

```python
import time

size = 10_000_000  # 10 million elements
A = np.random.randn(size).astype(np.float32)
B = np.random.randn(size).astype(np.float32)

# Warm up
_ = ch5_vector_ops.saxpy(2.0, A, B)

# Time MLIR implementation
start = time.time()
for _ in range(100):
    C = ch5_vector_ops.saxpy(2.0, A, B)
end = time.time()
mlir_time = (end - start) / 100

# Time NumPy implementation
start = time.time()
for _ in range(100):
    C = 2.0 * A + B
end = time.time()
numpy_time = (end - start) / 100

print(f"MLIR: {mlir_time*1000:.2f} ms")
print(f"NumPy: {numpy_time*1000:.2f} ms")
```

This benchmark runs SAXPY 100 times and averages the execution time. On modern hardware, the MLIR implementation should match or exceed NumPy's performance because both use vectorized operations (SIMD instructions). MLIR might be slightly faster due to JIT optimization for the specific CPU, or slightly slower due to JIT compilation overhead if the function isn't cached.

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

## 5.9 Summary

This chapter introduced explicit control flow in MLIR through the SCF (Structured Control Flow) dialect. We moved from declarative high-level operations to imperative explicit loops, gaining fine-grained control over computation. The key insights:

First, structured control flow provides an intermediate abstraction between declarative operations (Linalg) and unstructured branches (CF). The `scf.for` operation makes loop structure explicit while maintaining properties that enable optimization. We can see exactly how computation proceeds, iteration by iteration, without dropping all the way to assembly-level branches.

Second, explicit control flow requires explicit arithmetic. The Arith dialect provides operations for addition, multiplication, constants, and comparisons. These operations make numeric semantics precise, specifying floating-point versus integer arithmetic, signed versus unsigned operations, and overflow behavior. This precision is necessary when lowering to machine instructions, where every detail matters.

Third, working with dynamic shapes requires runtime queries. The `memref.dim` operation extracts dimension sizes at runtime, enabling dimension-agnostic code. We write functions that work with vectors of any length, querying sizes when needed rather than hardcoding dimensions. This flexibility is essential for production systems where input sizes vary.

Fourth, implementing operations explicitly teaches us how they work. SAXPY is conceptually simple—scale and add vectors—but implementing it with explicit loops reveals the mechanics: querying sizes, iterating indices, loading elements, performing arithmetic, storing results. This understanding becomes crucial when debugging or optimizing more complex operations.

Fifth, progressive lowering connects abstraction levels. We saw how SCF operations lower to CF branches, which lower to LLVM IR, which compiles to machine code. Each lowering step makes execution semantics more explicit while enabling level-appropriate optimizations. Understanding this pipeline helps you write code that optimizes well and debug issues when optimization falls short.

Finally, choosing the right abstraction level matters. Linalg for well-known operations that benefit from high-level optimization. SCF for custom operations that need explicit control or when learning how operations work. The two approaches aren't mutually exclusive—production systems use both, operating at the highest level appropriate for each operation. As MLIR developers, we should be comfortable moving between abstraction levels as needed.

With control flow mastered, we're ready to tackle more complex operations. The next chapter introduces the Math dialect and implements softmax, a fundamental operation in neural networks that requires multiple passes, careful numerics, and reduction patterns—building on everything we've learned about explicit loops and arithmetic operations.
