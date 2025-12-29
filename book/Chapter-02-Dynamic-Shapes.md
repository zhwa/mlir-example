# Chapter 2: Dynamic Shapes and Tensors

In Chapter 1, we compiled and executed our first MLIR program: an 8×32 matrix multiplied by a 32×16 matrix using the `linalg.matmul` operation. The operation worked perfectly, but with a significant limitation—the dimensions were hardcoded into the function signature. If we wanted to multiply matrices of different sizes, we would need to generate and compile separate functions for each dimension combination. This approach is impractical for real machine learning workloads, where batch sizes vary, sequences have different lengths, and models process inputs of diverse shapes.

This chapter addresses the fundamental question: **How do we write MLIR code that works with matrices and tensors of any size?** The answer involves understanding MLIR's distinction between **tensors** and **memrefs**, the concept of dynamic dimensions, and the runtime data structures that make shape-polymorphic code possible.

## 2.1 The Problem with Static Shapes

Consider a typical machine learning scenario: You're deploying a language model that processes user queries. Query lengths vary—some users write short questions, others provide detailed context. Your model's attention mechanism must handle input sequences of 10 tokens, 100 tokens, or 1000 tokens. With static shapes, you would need to compile separate kernels for each possible sequence length, or worse, pad all inputs to a maximum length and waste computation on padding tokens.

The fixed-size approach from Chapter 1 has several practical limitations:

1. **Compilation Overhead**: Generating and compiling a new function for each shape combination is slow and memory-intensive. Modern serving systems like vLLM and SGLang handle hundreds of concurrent requests with varying batch sizes—compiling specialized code for each configuration would be prohibitively expensive.

2. **Code Bloat**: Each function for different dimensions requires separate IR, optimization passes, and generated machine code. A model serving multiple batch sizes and sequence lengths could explode into thousands of compiled variants.

3. **Inflexibility at Runtime**: Machine learning inference systems must adapt to dynamic conditions. Batch sizes change based on request arrival patterns, sequence lengths vary by user input, and some optimizations (like continuous batching in Chapter 16) require splitting and merging batches dynamically.

To illustrate the problem, here's what Chapter 1's static approach looks like versus what we need:

```mlir
// Chapter 1: Static shapes - dimensions hardcoded in the function signature
func.func @gemm_8x32x16(%A: memref<8x32xf32>,    // Must be exactly 8×32
                        %B: memref<32x16xf32>,   // Must be exactly 32×16
                        %C: memref<8x16xf32>) {  // Must be exactly 8×16
  linalg.matmul ins(%A, %B : memref<8x32xf32>, memref<32x16xf32>)
                outs(%C : memref<8x16xf32>)
  return
}

// To handle different sizes, we'd need separate functions:
func.func @gemm_16x64x32(...)  // Different function for 16×64×32
func.func @gemm_4x128x64(...)  // Yet another for 4×128×64
// ... hundreds of variants for different batch/sequence combinations

// What we want: Dynamic shapes - ONE function for any compatible sizes
func.func @gemm(%A: memref<?x?xf32>,    // ? means "any size, determined at runtime"
                %B: memref<?x?xf32>,
                %C: memref<?x?xf32>) {
  linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
                outs(%C : memref<?x?xf32>)
  return
}
```

The `?` symbol is the key: it tells MLIR "this dimension is unknown at compile time." The solution is **dynamic shapes**—the ability to write and compile code once that works correctly for any compatible dimensions, with actual sizes determined at runtime.

## 2.2 Tensors vs Memrefs: Two Dialects for Multi-Dimensional Data

Before diving into dynamic shapes, we need to understand how MLIR represents multi-dimensional arrays. This is a crucial design choice that separates MLIR from most other compiler IRs: MLIR provides **two distinct dialects** (groups of related operations and types) for working with arrays, each with different semantics and use cases.

### The Tensor Dialect: Immutable Values

**The Tensor Dialect** defines immutable, value-semantic multi-dimensional arrays. Types like `tensor<8x32xf32>` represent abstract mathematical tensors—similar to how you think about matrices in linear algebra or immutable arrays in functional programming.

The tensor dialect emphasizes immutable values: operations create new tensors rather than modifying existing ones, following the same SSA semantics as integers or floats—once defined, a tensor value never changes. This immutability eliminates aliasing concerns entirely; there's no question about whether two tensor values point to the same memory because they're separate values in the dataflow graph. As a result, tensors are extremely optimization-friendly—the compiler can aggressively reorder, fuse, and eliminate operations without worrying about hidden side effects or memory dependencies.

**Connection to Chapter 1's SSA Semantics**: Remember from Chapter 1 that SSA (Static Single Assignment) means each *value* is assigned exactly once? Tensors embody this principle perfectly—they represent *mathematical values* with no memory location. A tensor like `%t0 : tensor<8x32xf32>` is a value in the SSA graph, just like `%x : i32` is a value. When you "modify" a tensor with `tensor.insert`, you're not mutating memory; you're creating a new value. This is pure SSA.

Contrast this with memrefs, which represent *memory references*. A memref like `%m0 : memref<8x32xf32>` is also an SSA value (the *pointer* is assigned once), but the data it points to is mutable—`memref.store` can write to the same location multiple times. This is exactly the distinction we emphasized in Chapter 1: SSA for pointers, not for the memory contents. Tensors take SSA to its logical conclusion by eliminating memory entirely at the high level.

**Example operations:**

```mlir
// Create an empty tensor
%t0 = tensor.empty() : tensor<8x32xf32>

// Extract a single element (returns a scalar value)
%val = tensor.extract %t0[%i, %j] : tensor<8x32xf32>

// Insert a value (returns a NEW tensor, %t0 unchanged)
%t1 = tensor.insert %val into %t0[%i, %j] : tensor<8x32xf32>

// Extract a slice (returns a new tensor view)
%slice = tensor.extract_slice %t0[0, 0] [4, 16] [1, 1] 
         : tensor<8x32xf32> to tensor<4x16xf32>

// Arithmetic on tensors (linalg operates on tensors)
%sum = linalg.add ins(%t0, %t1 : tensor<8x32xf32>, tensor<8x32xf32>)
                  outs(%t2 : tensor<8x32xf32>) -> tensor<8x32xf32>
```

Notice how each operation either creates a new tensor or extracts a value—there's no mutation of existing tensors.

### The MemRef Dialect: Mutable Memory Buffers

**The MemRef Dialect** defines mutable, reference-semantic memory buffers. Types like `memref<8x32xf32>` represent pointers to concrete memory locations with shape and stride information—essentially typed pointers with multi-dimensional indexing.

Unlike tensors, memrefs embrace mutability: operations directly read from and write to memory. A memref is fundamentally a pointer to a buffer, which means aliasing is possible—multiple memrefs can reference the same underlying memory, and the compiler must be conservative about reordering operations. However, this mutable, pointer-based model is exactly what makes memrefs execution-ready: they map directly to the CPU and GPU memory operations that hardware actually supports.

**Example operations:**

```mlir
// Allocate a buffer in memory
%m0 = memref.alloc() : memref<8x32xf32>

// Load a single element (reads from memory)
%val = memref.load %m0[%i, %j] : memref<8x32xf32>

// Store a value (MUTATES the memory at that location)
memref.store %val, %m0[%i, %j] : memref<8x32xf32>

// Get a view into existing memory (no allocation, just pointer arithmetic)
%view = memref.subview %m0[0, 0] [4, 16] [1, 1]
        : memref<8x32xf32> to memref<4x16xf32>

// Get the size of a dimension (queries the descriptor)
%dim = memref.dim %m0, 0 : memref<8x32xf32>

// Deallocate memory
memref.dealloc %m0 : memref<8x32xf32>
```

Notice how operations directly modify memory (`memref.store`) or query memory layout information (`memref.dim`, `memref.subview`). There's explicit allocation and deallocation—we're managing concrete memory.

### Comparing the Semantics: A Concrete Example

To make the distinction crystal clear, consider adding two matrices:

```mlir
// Tensor version (functional style)
func.func @add_tensors(%a: tensor<8x32xf32>, %b: tensor<8x32xf32>) 
                      -> tensor<8x32xf32> {
  // Returns a NEW tensor, %a and %b unchanged
  %result = linalg.add ins(%a, %b : tensor<8x32xf32>, tensor<8x32xf32>)
                       outs(%empty : tensor<8x32xf32>) -> tensor<8x32xf32>
  return %result : tensor<8x32xf32>
}

// MemRef version (imperative style)
func.func @add_memrefs(%a: memref<8x32xf32>, 
                       %b: memref<8x32xf32>,
                       %result: memref<8x32xf32>) {
  // MUTATES %result in place, no return value needed
  linalg.add ins(%a, %b : memref<8x32xf32>, memref<8x32xf32>)
             outs(%result : memref<8x32xf32>)
  return
}
```

The tensor version is like a pure function—it takes inputs and returns a new output. The memref version is like a C function with output parameters—it writes results into a pre-allocated buffer.

### Why Both? The Two-Phase Compilation Strategy

The dual representation reflects a fundamental tradeoff in compiler design:

**Tensors are easier to optimize** because immutability eliminates aliasing concerns. When the compiler sees `%2 = arith.addf %0, %1 : tensor<8x32xf32>`, it knows that `%0` and `%1` remain valid after the operation—no hidden modifications. The compiler can aggressively reorder, fuse, or eliminate operations based purely on the SSA dataflow graph. Many high-level transformations in MLIR (fusion, tiling, distribution) are designed to work on tensor operations specifically because the mathematical semantics are clean and unambiguous.

**Memrefs are necessary for execution** because at some point, we must generate code that runs on real hardware. CPUs and GPUs don't have "immutable tensors"—they have memory addresses, load instructions, and store instructions. When we call our JIT-compiled matrix multiplication from Python, we pass NumPy arrays, which are fundamentally pointers to buffers. The execution engine needs concrete memory locations to read inputs and write outputs.

This leads to a **two-phase compilation strategy** common in modern AI compilers:

1. **High-level optimizations on tensors**: Fusion, tiling, distribution, and other transformations operate on immutable tensor operations, taking advantage of clean mathematical semantics.

2. **Bufferization converts tensors to memrefs**: A transformation pass (discussed in Chapter 4) analyzes tensor dataflow and determines concrete memory allocation strategies, converting immutable tensors into mutable memory references.

3. **Low-level optimizations on memrefs**: Allocation hoisting, buffer reuse, memory layout optimizations, and cache-aware transformations operate at the memref level.

4. **Lowering to LLVM**: Finally, memrefs are lowered to raw pointers and array descriptors that LLVM can compile to machine code.

### Our Pragmatic Choice: Skip Tensors Entirely

For this book, we take a pragmatic shortcut: **we work directly with memrefs** and skip the tensor phase entirely. This decision has important implications.

Working with memrefs offers several advantages for learning MLIR fundamentals. First, it's simpler to understand because we avoid the complexity of bufferization (though Chapter 4 will cover the conceptual foundations of this transformation in depth). Second, the IR we write directly corresponds to how the code will execute—there are no hidden transformations that might surprise us along the way. Third, NumPy arrays are memory buffers at their core, so they map naturally to memrefs, making Python integration straightforward (Chapter 8 will show how the binding code works in detail). Finally, this approach enables faster initial development because we can focus on MLIR's core concepts without navigating the bufferization pipeline.

However, this choice involves trade-offs that readers should understand. By skipping tensors, we forgo some high-level optimizations—tensor-level transformations like fusion and tiling are more difficult or less effective when working directly with mutable memrefs. We must also be explicit about allocation and mutation, whereas tensor operations delegate these concerns to the bufferization pass. Finally, our approach isn't fully representative of production AI compiler pipelines; real systems like TensorFlow/MLIR, PyTorch 2.0, and JAX use tensors extensively at high levels before lowering to memrefs.

This is an acceptable trade-off for learning. As we build increasingly sophisticated examples (attention mechanisms in Chapter 11, transformers in Chapter 12, serving engines in Chapter 14), working with memrefs keeps the compilation pipeline transparent and the concepts grounded. Readers interested in production-scale tensor-based optimizations can explore MLIR's bufferization documentation and the Linalg-on-tensors workflow after mastering the fundamentals presented here.

### 2.2.5 MLIR's Hierarchical Structure: Operation/Block/Region

Before diving into dynamic shapes, we need to understand MLIR's structural organization. MLIR IR isn't flat—it's a **tree** with three distinct structural levels that nest recursively. This hierarchy appears everywhere in MLIR code, and understanding it is essential for reading IR, navigating code, and implementing transformations.

#### The Three Structural Levels

MLIR organizes IR using three concepts that build on each other:

```
┌────────────────────────────────────────────┐
│ OPERATION (Node in computation graph)      │
│  • Contains: operands, results, attributes │
│  • May contain: Regions ──────────┐        │
└───────────────────────────────────│────────┘
                                    │
                    ┌───────────────▼────────────┐
                    │ REGION (Code container)    │
                    │  • Contains: Blocks        │
                    │  • Examples: func body,    │
                    │    loop body, if branches  │
                    └───────────┬────────────────┘
                                │
                ┌───────────────▼────────────────┐
                │ BLOCK (Straight-line sequence) │
                │  • Contains: Operations        │
                │  • Has: Block arguments        │
                │  • Ends: Terminator op         │
                └────────────────────────────────┘
                         (back to Operations)
```

**Operations** are the fundamental units of computation—nodes in the computation graph. Each operation can contain **Regions**, which are code containers (like function bodies or loop bodies). Each Region contains one or more **Blocks**, which are straight-line sequences of operations. This creates a recursive structure: operations contain regions, regions contain blocks, blocks contain operations.

#### Example: Function Operation Breakdown

Let's examine a simple function to see this structure:

```mlir
func.func @example(%arg0: i32, %arg1: i32) -> i32 {
  %sum = arith.addi %arg0, %arg1 : i32
  %doubled = arith.muli %sum, %c2 : i32
  func.return %doubled : i32
}
```

**Structural Analysis**:

1. **Operation**: `func.func` is an operation (specifically, a `FuncOp`)
   - Operands: none (functions don't take operand *values*, they define parameters)
   - Results: none (the return type is in the signature, not an SSA result)
   - Attributes: function name (`@example`), type signature
   - **Contains a Region**: The function body

2. **Region**: The function body (everything between `{` and `}`)
   - Contains one Block (unlabeled blocks don't need `^label` syntax)
   - Purpose: holds the function's implementation

3. **Block**: The entry block
   - **Block arguments**: `%arg0` and `%arg1` (function parameters are block arguments!)
   - **Contains operations**: `arith.addi`, `arith.muli`, `func.return`
   - **Terminator**: `func.return` (every block must end with a terminator)

4. **Operations in the block**:
   - `arith.addi`: add operation (operands: `%arg0`, `%arg1`; result: `%sum`)
   - `arith.muli`: multiply operation (operands: `%sum`, `%c2`; result: `%doubled`)
   - `func.return`: return operation (operand: `%doubled`; no results—it terminates the block)

#### Why This Structure Matters

**Regions enable nested scoping**. A loop operation contains a region for its body. An if-then-else contains two regions (then-branch and else-branch). This hierarchical structure makes control flow explicit in the IR.

**Blocks enable control flow**. Within a region, control can flow between blocks via branch operations. Each block is a landing site for control flow—branches target blocks, not arbitrary instructions.

**Operations are uniform**. Everything is an operation: arithmetic (`arith.addi`), control flow (`cf.br`), function definitions (`func.func`), even module declarations (`builtin.module`). This uniformity simplifies IR manipulation.

#### Navigation APIs

When writing MLIR transformations (which we'll do starting in Chapter 8), you'll use these APIs to navigate the structure:

```cpp
// Navigate up the tree
Operation *parentOp = op.getParentOp();              // Get containing operation
Block *parentBlock = op->getBlock();                 // Get containing block  
Region *parentRegion = block->getParent();           // Get containing region
Operation *regionOwner = region->getParentOp();      // Get operation owning this region

// Navigate down the tree
Region &body = funcOp.getBody();                     // Get operation's region
Block &entryBlock = body.front();                    // Get first block in region
for (Operation &op : entryBlock) {                   // Iterate operations in block
  // Process each operation
}

// Access block arguments
BlockArgument arg0 = entryBlock.getArgument(0);      // Get first block argument
```

**Practical Example**: Walking all operations in a function:

```cpp
void processFunction(func::FuncOp funcOp) {
  // Get the function's body region
  Region &body = funcOp.getBody();
  
  // Iterate over all blocks in the region
  for (Block &block : body) {
    // Process block arguments (function parameters in entry block)
    for (BlockArgument arg : block.getArguments()) {
      llvm::outs() << "Block argument: " << arg << "\n";
    }
    
    // Process all operations in the block
    for (Operation &op : block) {
      llvm::outs() << "Operation: " << op.getName() << "\n";
    }
  }
}
```

#### Connection to SSA Form

Remember from Chapter 1 that MLIR uses SSA (Static Single Assignment)? The structural hierarchy clarifies what "assignment" means:

- **Values** (SSA values) are defined by operations and block arguments
- **Operations** produce results (new SSA values)
- **Block arguments** introduce values at block entry (we'll see why in section 2.6.5)
- **Values** are visible within their defining region and nested regions (lexical scoping)

This structure makes MLIR's IR both human-readable (the hierarchy mirrors code structure) and machine-analyzable (tree traversal is straightforward).

#### Looking Ahead

Understanding this structure is crucial for:
- **Reading IR** (Chapter 2 onwards): You'll see regions and blocks everywhere
- **Building IR** (Chapter 8-9): Creating operations with regions and blocks
- **Transforming IR** (Chapter 9-10): Navigating and modifying the tree structure
- **Pattern matching** (Chapter 9): Patterns match operations, rewrite blocks

Now that we understand MLIR's structural foundation, let's explore how dynamic dimensions work.

## 2.3 Dynamic Dimensions: The Question Mark Notation

With the tensor-vs-memref distinction clarified, we can now address dynamic shapes. In MLIR, dynamic dimensions are indicated by a **question mark** (`?`) in type signatures. Compare these type declarations:

```mlir
// Static shape: dimensions known at compile time
%static = ... : memref<8x32xf32>

// Dynamic shape: dimensions unknown at compile time
%dynamic = ... : memref<?x?xf32>
```

The `?` is syntactic sugar for a special constant: `ShapedType::kDynamic`, defined as `-1` in MLIR's C++ API. When the compiler sees `memref<?x?xf32>`, it understands: "This is a 2D array of float32 values, but I don't know the dimensions yet—they'll be provided at runtime."

### What Does "Dynamic" Actually Mean?

It's crucial to understand what "dynamic" means in this context. The dimensions are **unknown at compile time** but **fixed at runtime** for any particular invocation. This is not the same as a resize operation or a dynamically growing vector. Consider this generated function signature:

```mlir
func.func @gemm(%arg0: memref<?x?xf32>,
                %arg1: memref<?x?xf32>,
                %arg2: memref<?x?xf32>)
```

When we call this function from Python with specific NumPy arrays:

```python
A = np.random.randn(8, 32).astype(np.float32)
B = np.random.randn(32, 16).astype(np.float32)
C = np.zeros((8, 16), dtype=np.float32)

gemm(A, B, C)  # Dimensions are 8×32, 32×16, 8×16 for this call
```

The `?` dimensions are **resolved to concrete values** (8, 32, 16) for the duration of this function call. Inside the compiled code, the dimensions don't change—they're just not hardcoded at compile time. This is analogous to C++ templates or Java generics: one compiled function works for any compatible types, but within each invocation, types are concrete.

### Mixed Static and Dynamic Shapes

MLIR allows **mixing static and dynamic dimensions** in the same type. This flexibility is useful when some dimensions are known (e.g., feature dimensions in a model) while others vary (e.g., batch size or sequence length):

```mlir
// Batch size is dynamic, but feature dimension is always 768
%embeddings = ... : memref<?x768xf32>

// Batch and sequence length are dynamic, but attention heads and head dimension are static
%attention = ... : memref<?x?x12x64xf32>
```

This mixed-shape capability appears frequently in transformer models, where model dimensions (hidden size, number of heads) are fixed at model design time, but batch sizes and sequence lengths are dynamic based on input data.

### Shape-Polymorphic Operations

The key insight that makes dynamic shapes practical is that **most operations don't need to know exact dimensions**. Consider matrix multiplication semantics:

```
C[i,j] = Σ_k A[i,k] * B[k,j]
```

This definition is purely in terms of loop indices (`i`, `j`, `k`) and doesn't depend on whether the dimensions are 8×32×16 or 1024×2048×512. The operation is **shape-polymorphic**—it works for any dimensions as long as they're compatible (the inner dimension `K` matches).

MLIR's `linalg.matmul` operation exploits this property:

```mlir
linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
              outs(%C : memref<?x?xf32>)
```

The operation's semantics are defined abstractly, and the generated loop nest adapts to whatever dimensions are actually provided at runtime. This is why we can write our matrix multiplication code once in Chapter 2 and use it for any matrix sizes.

## 2.4 The MemRef Descriptor: How Dynamic Shapes Work at Runtime

Understanding the `?` notation is one thing, but how does the compiled code actually **use** dynamic dimensions? This is where MLIR's **memref descriptor** comes into play—a runtime data structure that carries shape information alongside the actual data pointer.

### The Problem: Loops Need Bounds

Consider what a compiled matrix multiplication must do at runtime:

```mlir
// High-level MLIR (conceptual)
linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
              outs(%C : memref<?x?xf32>)

// After lowering to loops (conceptual)
scf.for %i = 0 to %M {      // How does the code know what M is?
  scf.for %j = 0 to %N {    // Where does N come from?
    scf.for %k = 0 to %K {  // And K?
      %a = memref.load %A[%i, %k]   // How to compute the memory address?
      %b = memref.load %B[%k, %j]
      %c = memref.load %C[%i, %j]
      %prod = arith.mulf %a, %b
      %sum = arith.addf %c, %prod
      memref.store %sum, %C[%i, %j]
    }
  }
}
```

To execute these loops, the compiled code needs:
1. **Loop bounds**: The values of `M`, `N`, and `K`
2. **Address computation**: How to calculate the memory address of `A[i,k]` given indices `i` and `k`

The memref descriptor provides this information. Instead of passing a simple pointer, MLIR passes a **structure** containing five key pieces of information. The base pointer holds the actual memory address of the first element in the buffer. The aligned pointer provides a potentially aligned pointer for optimized access, often used for SIMD or cache-efficient operations (in simple cases, this equals the base pointer). The offset stores a base offset into the buffer, which becomes important when working with views and subranges. The sizes array holds the dimensions—for example, `[8, 32]` for an 8×32 matrix—with dynamic dimensions determined at runtime. Finally, the strides array specifies how many elements to skip to move one step along each dimension, enabling flexible memory layouts like row-major, column-major, or transposed views.

### The Structure in Detail

For a 2D memref like `memref<?x?xf32>`, the descriptor is logically:

```cpp
struct MemRefDescriptor_2D_f32 {
  float* allocated;       // Pointer to the allocated memory
  float* aligned;         // Potentially aligned pointer for SIMD/cache
  int64_t offset;         // Offset from base pointer
  int64_t sizes[2];       // [dim0, dim1] - the actual dimensions
  int64_t strides[2];     // [stride0, stride1] - elements to skip per dimension
};
```

For a specific call with an 8×32 matrix stored in row-major order:

```cpp
MemRefDescriptor_2D_f32 desc_A = {
  .allocated = 0x7f8a3c000000,  // Address of A's buffer
  .aligned   = 0x7f8a3c000000,  // Same if no special alignment
  .offset    = 0,               // No offset from base
  .sizes     = {8, 32},         // 8 rows, 32 columns
  .strides   = {32, 1}          // Skip 32 elements for next row, 1 for next column
};
```

### Computing Memory Addresses

With this descriptor, computing the address of `A[i,k]` is straightforward:

```
address(A[i,k]) = aligned_ptr + offset + (i * stride[0]) + (k * stride[1])
                = aligned_ptr + 0 + (i * 32) + (k * 1)
                = aligned_ptr + 32*i + k
```

This formula works regardless of the actual dimensions—it's computed at runtime using the `sizes` and `strides` fields. The compiled code doesn't hardcode `32`; it loads `strides[0]` from the descriptor.

### Why Strides Matter

Strides enable flexible memory layouts without changing the indexing logic. Consider different ways to store an 8×32 matrix:

**Row-major layout** (C/NumPy default):
```
Strides: [32, 1]
A[i,k] is at: base + 32*i + k
```

**Column-major layout** (Fortran/MATLAB):
```
Strides: [1, 8]
A[i,k] is at: base + i + 8*k
```

**Transposed view** (no data copy):
```
// A is 8×32, but we want to view it as 32×8 (transposed)
Sizes: [32, 8]
Strides: [1, 32]  // Swap the strides
A_transposed[k,i] refers to the same memory as A[i,k]
```

This is why operations like `memref.subview` or `memref.transpose` can be **zero-cost**—they just create new descriptors with different sizes/strides/offsets pointing to the same underlying buffer. No data is copied; only the metadata changes.

### Passing Descriptors Across the ABI

**What is an ABI?** Before we dive into how descriptors cross language boundaries, we need to understand a fundamental concept in systems programming: the **Application Binary Interface** (ABI).

When we write code in high-level languages (Python, C++, MLIR), we think in terms of functions, objects, and data structures. But when that code actually runs, it's compiled to machine instructions, and function calls become very concrete: arguments are placed in specific CPU registers or stack locations, the program counter jumps to a memory address, and results are returned in designated registers. The **ABI** is the specification that defines these low-level details—how to pass arguments, where return values go, which registers the caller vs callee must preserve, how the stack is organized, and so on.

Different languages and platforms have different ABIs. The **C ABI** is particularly important because it's the "lingua franca" of systems programming—most languages can call C functions and be called by C. When we compile MLIR to machine code and want to call it from Python (or any other language), we need to follow the C ABI conventions so the two sides agree on how to pass data.

**Connection to Chapter 1**: Remember the "Python ↔ C++ Boundary" section in Chapter 1? This is the same boundary, but now we understand what crosses it: not just raw pointers, but complex memref descriptor structures. The ABI is the contract that makes this crossing possible.

This matters especially for dynamic shapes because the memref descriptor isn't a simple C type—it's a structure with multiple fields (pointers, sizes, strides). The ABI determines exactly how this structure is "flattened" into function arguments. Does it get passed as a pointer to a struct? As individual arguments? In registers or on the stack? MLIR's lowering passes make these decisions, and understanding the ABI helps us understand what's happening under the hood.

**From MLIR Functions to C Calling Convention**: When Python calls our JIT-compiled MLIR function, the descriptors are constructed from NumPy arrays and passed via MLIR's **C calling convention**. The function signature:

```mlir
func.func @gemm(%arg0: memref<?x?xf32>,
                %arg1: memref<?x?xf32>,
                %arg2: memref<?x?xf32>)
```

gets compiled to a C-compatible function that expects 21 arguments per memref (for a 2D memref):

```cpp
// Conceptual C signature (actual ABI is more complex)
void gemm(
  // First memref (A): 7 parameters
  float* A_allocated, float* A_aligned, int64_t A_offset,
  int64_t A_size0, int64_t A_size1,
  int64_t A_stride0, int64_t A_stride1,
  
  // Second memref (B): 7 parameters
  float* B_allocated, float* B_aligned, int64_t B_offset,
  int64_t B_size0, int64_t B_size1,
  int64_t B_stride0, int64_t B_stride1,
  
  // Third memref (C): 7 parameters
  float* C_allocated, float* C_aligned, int64_t C_offset,
  int64_t C_size0, int64_t C_size1,
  int64_t C_stride0, int64_t C_stride1
);
```

This is why the "21 parameters" number appears in MLIR tutorials—it's 3 memrefs × 7 fields per memref. Our Python binding code (via `pybind11` and `libffi`) extracts pointer, shape, and stride information from NumPy arrays and marshals them into this calling convention. The details are handled automatically by MLIR's execution engine, but understanding the underlying structure demystifies what's happening when we cross the Python/C boundary.

## 2.5 Building Dynamic IR with the C++ API

Now that we understand the conceptual foundations, let's see how to generate MLIR IR with dynamic shapes using the C++ API. The changes from Chapter 1's static version are minimal—testament to MLIR's design.

### Creating Dynamic MemRef Types

In Chapter 1, we created static types with explicit dimensions:

```cpp
// Chapter 1: Static dimensions
auto matrixA_type = MemRefType::get({8, 32}, builder.getF32Type());
auto matrixB_type = MemRefType::get({32, 16}, builder.getF32Type());
auto matrixC_type = MemRefType::get({8, 16}, builder.getF32Type());
```

For dynamic shapes, we use `ShapedType::kDynamic` (represented as `-1`) for unknown dimensions:

```cpp
// Chapter 2: Dynamic dimensions
auto f32Type = builder.getF32Type();
auto matrixA_type = MemRefType::get(
    {ShapedType::kDynamic, ShapedType::kDynamic},  // ?x?
    f32Type
);
auto matrixB_type = MemRefType::get(
    {ShapedType::kDynamic, ShapedType::kDynamic},  // ?x?
    f32Type
);
auto matrixC_type = MemRefType::get(
    {ShapedType::kDynamic, ShapedType::kDynamic},  // ?x?
    f32Type
);
```

This generates the MLIR type `memref<?x?xf32>`. The rest of the IR generation code is **identical** to Chapter 1:

```cpp
// Create function type
auto funcType = builder.getFunctionType(
  {matrixA_type, matrixB_type, matrixC_type},
  {}  // No return values (output written to C)
);

// Create the function
auto funcOp = builder.create<func::FuncOp>(loc, "gemm", funcType);
funcOp.setPublic();

// Create function body
auto* entryBlock = funcOp.addEntryBlock();
builder.setInsertionPointToStart(entryBlock);

Value argA = entryBlock->getArgument(0);
Value argB = entryBlock->getArgument(1);
Value argC = entryBlock->getArgument(2);

// Fill output with zeros
auto zeroAttr = builder.getFloatAttr(f32Type, 0.0);
auto zeroConstant = builder.create<arith::ConstantOp>(loc, zeroAttr);
builder.create<linalg::FillOp>(loc, zeroConstant, argC);

// Matrix multiplication
builder.create<linalg::MatmulOp>(
  loc,
  ValueRange{argA, argB},  // Inputs
  ValueRange{argC}         // Output
);

builder.create<func::ReturnOp>(loc);
```

The `linalg.matmul` operation doesn't change—it's already shape-polymorphic. The key difference is that the **types** of the arguments now carry `?` dimensions, which tells MLIR's lowering passes to generate runtime dimension queries instead of hardcoded bounds.

### The Generated IR

The generated IR looks like this:

```mlir
func.func @gemm(%arg0: memref<?x?xf32>,
                %arg1: memref<?x?xf32>,
                %arg2: memref<?x?xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill ins(%cst : f32) outs(%arg2 : memref<?x?xf32>)
  linalg.matmul ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
                outs(%arg2 : memref<?x?xf32>)
  return
}
```

Notice the `?` dimensions in the function signature and operation types. This IR is **shape-polymorphic**—it represents matrix multiplication for any compatible dimensions.

## 2.6 Lowering Dynamic Shapes to Loops

The compilation pipeline for dynamic shapes is identical to Chapter 1's pipeline, but the **generated code** adapts to runtime dimensions. Let's trace what happens when we apply the standard lowering passes.

### Phase 1: Linalg to Loops

The `createConvertLinalgToLoopsPass()` transforms `linalg.matmul` into nested loops. For static shapes, the loop bounds were constants. For dynamic shapes, the pass generates code to **query dimensions at runtime**:

```mlir
// Conceptual lowering (simplified)
%M = memref.dim %arg0, 0 : memref<?x?xf32>  // Get first dimension of A
%K = memref.dim %arg0, 1 : memref<?x?xf32>  // Get second dimension of A
%N = memref.dim %arg1, 1 : memref<?x?xf32>  // Get second dimension of B

scf.for %i = %c0 to %M step %c1 {
  scf.for %j = %c0 to %N step %c1 {
    scf.for %k = %c0 to %K step %c1 {
      %a = memref.load %arg0[%i, %k] : memref<?x?xf32>
      %b = memref.load %arg1[%k, %j] : memref<?x?xf32>
      %c = memref.load %arg2[%i, %j] : memref<?x?xf32>
      %prod = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %prod : f32
      memref.store %sum, %arg2[%i, %j] : memref<?x?xf32>
    }
  }
}
```

The `memref.dim` operation extracts the size of a specific dimension from the memref descriptor. At runtime, this compiles to simply loading the appropriate field from the descriptor structure we discussed earlier.

### Phase 2: SCF to Control Flow

This phase transforms structured control flow (loops, if-then-else) into low-level control flow (branches and basic blocks). But why does MLIR have **two different dialects** for control flow in the first place?

**The SCF (Structured Control Flow) Dialect** provides high-level control flow constructs that programmers understand: `scf.for` (loops with induction variables), `scf.while` (conditional loops), `scf.if` (conditional branches). These operations have **structured semantics**—a `scf.for` loop has a clear beginning, end, loop body, and induction variable that increments. This structure makes them easy to analyze and transform. The compiler can reason about loop bounds, apply loop optimizations (unrolling, vectorization, tiling), and understand the relationship between iterations. SCF operations also compose nicely with SSA values—loop-carried dependencies are expressed explicitly through "iter_args" rather than mutable variables. We'll work extensively with the SCF dialect starting in Chapter 5, where vector operations require explicit loop construction, and again in Chapter 6 when implementing softmax with its reduction patterns.

**The CF (Control Flow) Dialect** provides low-level, unstructured control flow: `cf.br` (unconditional branch), `cf.cond_br` (conditional branch), and basic blocks. This is the control flow model that CPUs actually understand—jump instructions and labels. There's no notion of "loops" or "structured if-then-else" at this level, just "goto" operations between blocks. CF operations are harder to optimize (determining loop structure from arbitrary branches is a hard problem) but necessary for final code generation because LLVM expects this form.

**Why Both?** This is a manifestation of MLIR's **progressive lowering** philosophy. We want to keep operations at the highest useful abstraction level for as long as possible. Early optimization passes work with `scf.for` loops because structured loops are easier to analyze and transform. Only when we're ready to generate LLVM IR do we lower to basic blocks and branches.

Think of it like compiling a high-level language: You wouldn't want to optimize Python directly at the assembly level (too hard to recognize patterns), and you wouldn't want to generate machine code directly from Python syntax (too many semantic details). The SCF → CF lowering is analogous—we progressively simplify the representation while preserving correctness.

### Block Arguments: MLIR's SSA Innovation

Before proceeding with control flow lowering, we need to understand a fundamental design choice in MLIR's SSA representation: **block arguments** vs traditional **phi functions**. This distinction affects how control flow merges values from different paths.

#### Traditional SSA: Phi Functions

In traditional SSA representations (like LLVM IR), control flow merge points use **phi functions** to select values based on which predecessor block was executed:

```llvm
; LLVM IR example: conditional selection
entry:
  %cond = icmp sgt i32 %x, %threshold
  br i1 %cond, label %then, label %else

then:
  %a = mul i32 %x, 2          ; If x > threshold, double it
  br label %merge

else:
  %b = mul i32 %x, 3          ; If x ≤ threshold, triple it
  br label %merge

merge:
  %result = phi i32 [ %a, %then ], [ %b, %else ]  ; Phi function
  ret i32 %result
```

The phi function `%result = phi i32 [ %a, %then ], [ %b, %else ]` means: "If control came from `%then`, use `%a`; if it came from `%else`, use `%b`."

**Problems with Phi Functions**:
1. **Position ambiguity**: Phi functions must appear at the start of a block, but their semantics reference predecessors
2. **Transformation complexity**: Adding/removing predecessors requires updating all phi functions
3. **Redundancy**: Every merged value needs a separate phi function

#### MLIR's Approach: Block Parameters

MLIR eliminates phi functions by treating blocks like functions with parameters. When branching to a block, you pass values as arguments:

```mlir
// MLIR equivalent: block parameters instead of phi
^entry(%x: i32, %threshold: i32):
  %cond = arith.cmpi sgt, %x, %threshold : i32
  cf.cond_br %cond, ^then(%x : i32), ^else(%x : i32)

^then(%x_then: i32):         // Block parameter (like function parameter)
  %a = arith.muli %x_then, %c2 : i32
  cf.br ^merge(%a : i32)     // Pass %a to ^merge

^else(%x_else: i32):         // Separate block parameter
  %b = arith.muli %x_else, %c3 : i32
  cf.br ^merge(%b : i32)     // Pass %b to ^merge

^merge(%result: i32):        // Block parameter receives merged value
  func.return %result : i32
```

**Key Observations**:

1. **Block signatures**: `^merge(%result: i32)` declares the block takes one i32 parameter
2. **Passing arguments**: `cf.br ^merge(%a : i32)` branches to `^merge` and passes `%a` as argument
3. **No phi needed**: The `%result` parameter automatically receives the appropriate value
4. **Uniform syntax**: Block parameters look just like function parameters

#### Why Block Parameters Are Better

**Cleaner Semantics**: Values flow explicitly through branches. You can see what data each branch provides:
```mlir
cf.cond_br %cond, ^then(%x : i32), ^else(%y : i32)
// Clear: ^then gets %x, ^else gets %y
```

**Simpler IR Manipulation**: Adding a predecessor? Just pass an argument. No need to find and update all phi functions.

**Better Composability with Regions**: Since blocks can be nested in regions (section 2.2.5), block parameters integrate naturally with region arguments (like function parameters).

#### Connection to Section 2.2.5: Structure and Values

Remember from section 2.2.5 that blocks are part of MLIR's hierarchical structure. Block arguments are **values** defined at block entry—they're part of the SSA value system:

- **Operations** produce results (SSA values)
- **Block arguments** introduce values (also SSA values)
- Both follow SSA: defined once, used multiple times

When you navigate IR with `block.getArguments()`, you're accessing these block parameters.

#### Practical Example: Loop Iteration Variable

Consider a for-loop. In traditional SSA, the loop counter would need phi functions. In MLIR, it's a block parameter:

```mlir
// Loop counter as block parameter
^loop_header(%i: index):              // Counter is block parameter
  %cond = arith.cmpi slt, %i, %N : index
  cf.cond_br %cond, ^loop_body(%i : index), ^exit

^loop_body(%i_body: index):
  // ... loop body using %i_body ...
  %i_next = arith.addi %i_body, %c1 : index
  cf.br ^loop_header(%i_next : index)  // Pass updated counter back

^exit:
  func.return
```

The counter `%i` flows through block parameters—no phi functions needed.

#### When You'll See Block Arguments

- **Function entry blocks**: Function parameters are block arguments of the entry block
- **Loop headers**: Loop iteration variables and carried values
- **Conditionals**: Merged values from different branches
- **Pattern matching** (Chapter 9): Patterns must handle block arguments correctly

**Looking Ahead**: When we implement custom dialects (Chapter 8-9) and write transformations (Chapter 9-10), understanding block arguments is essential. Pattern rewrites often create new blocks with parameters, and getting the argument types wrong causes verification failures.

### Safe IR Modification: Critical Patterns to Avoid Crashes

Now that we understand MLIR's structure (operations, blocks, regions) and how values flow (SSA, block arguments), we need to address a critical topic: **safely modifying IR**. Starting in Chapter 8, we'll implement custom dialects and transformations that rewrite IR. Incorrect modification patterns cause segmentation faults that are difficult to debug. This section covers essential safety rules.

#### **[CRITICAL WARNING]**: The Use-Def Chain Corruption Problem

MLIR maintains **use-def chains**—data structures tracking which operations use each value. These chains are implemented as linked lists. **Modifying IR while iterating use-def chains corrupts the list and causes crashes.**

#### DANGEROUS Pattern #1: Iterating Uses While Modifying

```cpp
// ❌ DANGEROUS - DO NOT DO THIS
// Modifying use-def chain while iterating it
for (auto &use : value.getUses()) {
  use.set(newValue);  // CRASHES! Corrupts iterator
}
```

**Why it crashes**: `value.getUses()` returns an iterator over a linked list. Calling `use.set(newValue)` modifies that list (removes old use, adds new use). The iterator is now invalid—next iteration accesses freed memory → segfault.

This is the **#1 beginner mistake** in MLIR. It seems logical ("iterate over uses and change them"), but it's fundamentally unsafe.

#### SAFE Pattern #1: Use MLIR's API

```cpp
// ✅ SAFE - Use MLIR's built-in method
value.replaceAllUsesWith(newValue);
```

MLIR's `replaceAllUsesWith` handles iteration correctly. It updates the use-def chain safely without exposing the internal linked list to corruption. **Always use this API** instead of manual iteration.

#### SAFE Pattern #2: Collect Then Modify

If you need conditional replacement (not all uses), collect uses first:

```cpp
// ✅ SAFE - Collect first, modify later
SmallVector<OpOperand*> usesToUpdate;
for (auto &use : value.getUses()) {
  if (shouldReplace(use)) {  // Conditional logic
    usesToUpdate.push_back(&use);
  }
}
// Now modify (no longer iterating the original use-chain)
for (auto *use : usesToUpdate) {
  use->set(newValue);
}
```

**Why this works**: The first loop doesn't modify the use-chain—it just reads it. The second loop operates on a separate vector, so there's no iterator invalidation.

#### SAFE Pattern #3: IRMapping for Bulk Replacements

When replacing multiple values simultaneously (common in pattern rewriting):

```cpp
// ✅ SAFE - IRMapping for bulk value remapping
IRMapping mapping;
mapping.map(oldOp->getResults(), newOp->getResults());

// Safe bulk replacement across multiple operands
for (auto &operand : consumerOp->getOpOperands()) {
  operand.set(mapping.lookupOrDefault(operand.get()));
}
```

`IRMapping` is a value-to-value dictionary. `lookupOrDefault(value)` returns the replacement if it exists, otherwise returns the original value unchanged. This pattern appears frequently in Chapter 9's dialect lowering passes.

#### DANGEROUS Pattern #2: Deleting Operations with Users

```cpp
// ❌ DANGEROUS - Operation still has users!
op->erase();  // CRASHES if op.getUsers() is non-empty
```

**Why it crashes**: If any other operation uses `op`'s results, those operations now have dangling references. Accessing those references → segfault.

#### SAFE Pattern #4: Remove Users First

```cpp
// ✅ SAFE - Ensure no users before erasing
if (!op->use_empty()) {
  op.replaceAllUsesWith(replacement);  // or signal error if no replacement
}
op->erase();  // Now safe
```

Always verify the operation has no users before erasing. MLIR's verification will catch this in debug builds, but release builds will crash silently.

#### Understanding `remove()` vs `erase()` Semantics

MLIR provides two operation deletion methods with different semantics:

```cpp
// op->remove(): Remove from block, keep alive
op->remove();  // Removes from parent block but doesn't destroy
// Can reinsert later:
block->push_back(op);  // Re-insert into different block

// op->erase(): Remove AND destroy
op->erase();   // Removes from block AND deallocates memory
// Cannot reuse - pointer is now dangling!
```

**Use `remove()` when**: Moving operations between blocks (control flow restructuring)
**Use `erase()` when**: Permanently deleting dead code

#### Deletion Order Constraint

**Rule**: Operations must be erased in reverse post-order (users before definitions).

```cpp
// ❌ WRONG ORDER
producer->erase();  // Producer deleted first
consumer->erase();  // Consumer references deleted value - crash!

// ✅ CORRECT ORDER
consumer->erase();  // Consumer deleted first (no dangling refs)
producer->erase();  // Now safe to delete producer
```

In practice, this means: **delete users before definitions**. If `%result = op1()` and `op2(%result)` uses it, delete `op2` first.

#### Pattern Rewriting (Preview of Chapter 9)

When we write dialect lowering in Chapter 9, we'll use `PatternRewriter` which handles safety automatically:

```cpp
// Chapter 9 preview: PatternRewriter is safe
LogicalResult matchAndRewrite(MyOp op, PatternRewriter &rewriter) const {
  auto newOp = rewriter.create<ReplacementOp>(...);
  rewriter.replaceOp(op, newOp.getResult());  // Safe!
  return success();
}
```

`PatternRewriter` methods like `replaceOp`, `eraseOp`, and `replaceAllUsesWith` handle use-def chain updates correctly. This is why we use pattern rewriting instead of manual IR mutation—it's safe by construction.

#### Debugging Tips: When Things Crash

1. **Enable assertions**: Compile MLIR with `LLVM_ENABLE_ASSERTIONS=ON`. Many use-def violations are caught in debug builds.

2. **Check use-def chain**: Before modifying, print uses:
   ```cpp
   for (auto &use : value.getUses()) {
     llvm::errs() << "User: " << *use.getOwner() << "\n";
   }
   ```

3. **Verify after modifications**: Call `module.verify()` after transformations to catch structural violations early.

4. **Use MLIR's diagnostic system**: Don't ignore verification errors—they pinpoint exactly what's wrong.

#### Summary: The Golden Rules

1. **Never iterate uses while modifying** → Use `replaceAllUsesWith()`
2. **Never erase operations with users** → Remove users first or replace with other values
3. **Collect, then modify** → If conditional logic needed, collect pointers first
4. **Use IRMapping for bulk replacement** → Handles multiple value remapping safely
5. **Delete users before definitions** → Reverse post-order deletion
6. **Prefer PatternRewriter** → Let MLIR handle safety automatically (Chapter 9)

These patterns prevent 80% of beginner MLIR crashes. When you encounter a segfault in Chapters 8-10, revisit this section and verify you're following these rules.

Now let's continue with the lowering phases.

In this phase, the `createSCFToControlFlowPass()` transforms our structured loops:

```mlir
// Before: Structured loop (SCF dialect)
scf.for %i = %c0 to %M step %c1 {
  // Loop body
}

// After: Basic blocks and branches (CF dialect)
^loop_header:
  %cond = arith.cmpi ult, %i, %M
  cf.cond_br %cond, ^loop_body, ^loop_exit
^loop_body:
  // Loop body
  %i_next = arith.addi %i, %c1
  cf.br ^loop_header
^loop_exit:
  // Continue after loop
```

The structured `scf.for` becomes a header block (check condition), a body block (execute iteration), and an exit block (continue after loop). The induction variable `%i` is explicitly incremented and passed between blocks. This representation has no special "loop" semantics—it's just branches, exactly what LLVM expects.

For our dynamic shapes case, the only difference from Chapter 1 is that loop bounds (`%M`, `%N`, `%K`) are now **variables** (loaded from memref descriptors) rather than constants. The lowering logic is identical—SCF's structured loops become CF's branches, regardless of whether bounds are static or dynamic.

### Phase 3: MemRef to LLVM

The `memref.dim` operations are lowered to simple struct field accesses:

```llvm
// Conceptual LLVM IR (simplified)
%desc = ... // The memref descriptor
%M_ptr = getelementptr %desc, 0, 3  // sizes[0] is at offset 3
%M = load i64, i64* %M_ptr
```

Similarly, `memref.load` and `memref.store` are lowered to address computations using the stride information:

```llvm
// address(A[i,k]) = aligned_ptr + offset + i*stride[0] + k*stride[1]
%stride0_ptr = getelementptr %desc, 0, 5  // strides[0]
%stride1_ptr = getelementptr %desc, 0, 6  // strides[1]
%stride0 = load i64, i64* %stride0_ptr
%stride1 = load i64, i64* %stride1_ptr

%offset_i = mul i64 %i, %stride0
%offset_k = mul i64 %k, %stride1
%linear_index = add i64 %offset_i, %offset_k
%element_ptr = getelementptr float, float* %aligned, i64 %linear_index
%value = load float, float* %element_ptr
```

This is the machinery that makes dynamic shapes work—the compiled code performs a few extra loads and multiplications to compute addresses and bounds, but the overall structure is the same as the static version.

## 2.7 Calling from Python: No Changes Required

One of the elegant aspects of MLIR's approach is that **the Python side doesn't change** when we move from static to dynamic shapes. Our test code from Chapter 1 works without modification:

```python
import numpy as np
import ch2_dynamic_size as gemm

# Create test matrices
A = np.random.randn(8, 32).astype(np.float32)
B = np.random.randn(32, 16).astype(np.float32)
C = np.zeros((8, 16), dtype=np.float32)

# Call JIT-compiled GEMM
gemm.gemm(A, B, C)

# Verify correctness
expected = A @ B
np.testing.assert_allclose(C, expected, rtol=1e-5)
```

The `pybind11` binding code (covered in detail in Chapter 8) automatically extracts shape information from NumPy arrays and constructs the memref descriptors. When we pass different-sized matrices:

```python
# Different sizes work with the same compiled function
A2 = np.random.randn(16, 64).astype(np.float32)
B2 = np.random.randn(64, 32).astype(np.float32)
C2 = np.zeros((16, 32), dtype=np.float32)

gemm.gemm(A2, B2, C2)  # Same function, different shapes
```

The compiled `gemm` function adapts at runtime. The first call processes 8×32 × 32×16 matrices; the second call processes 16×64 × 64×32 matrices. The loop bounds and address computations adjust based on the descriptor fields passed from Python.

### Performance: Static vs Dynamic

A natural question: Does using dynamic shapes hurt performance? The answer is **typically no** for reasonably large workloads. The overhead is:

1. **A few extra loads**: Reading dimensions and strides from the descriptor (maybe 5-10 loads per memref)
2. **Indirect addressing**: Loop bounds and strides are variables instead of constants

For a matrix multiplication of size 8×32 × 32×16 (4096 multiply-add operations), the overhead of loading 21 descriptor fields is negligible—less than 0.1% of the total work. Modern CPUs are very good at this kind of indirection.

However, for **very small operations** (e.g., adding two 4-element vectors), static shapes might be measurably faster because the compiler can unroll loops and eliminate overhead entirely. Production AI compilers often use heuristics: generate specialized kernels for common shapes (e.g., batch size 1, sequence length 512) and fall back to dynamic kernels for rare shapes.

## 2.8 Beyond 2D: Higher-Rank Tensors

While we've focused on 2D matrices, MLIR's dynamic shape machinery extends naturally to higher-rank tensors. A 4D activation tensor from a transformer model might have type:

```mlir
%activations : memref<?x?x12x64xf32>
```

This represents a tensor with a dynamic batch size (`?`), dynamic sequence length (`?`), static number of attention heads (`12`), and static head dimension (`64`). The flexibility of mixing static and dynamic dimensions is powerful—we can fix aspects of the computation that never change (like architectural hyperparameters) while remaining flexible for dimensions that vary with input data.

The memref descriptor simply has more fields:

```cpp
struct MemRefDescriptor_4D_f32 {
  float* allocated;
  float* aligned;
  int64_t offset;
  int64_t sizes[4];     // [batch, seq_len, 12, 64]
  int64_t strides[4];   // Strides for each dimension
};
```

Address computation generalizes:

```
address(T[b,s,h,d]) = aligned + offset + b*stride[0] + s*stride[1] 
                                       + h*stride[2] + d*stride[3]
```

Operations like multi-head attention (Chapter 11) work with these 4D tensors naturally, with the compiler generating appropriate dimension queries and address calculations.

## 2.9 Shape Constraints and Compatibility

While dynamic shapes provide flexibility, **not all dimension combinations are valid**. Matrix multiplication requires the inner dimension to match: if `A` is `M×K` and `B` is `K×N`, the `K` dimensions must be equal. MLIR doesn't automatically verify this at compile time for dynamic shapes—it's the caller's responsibility.

If we pass incompatible shapes:

```python
A = np.random.randn(8, 32).astype(np.float32)   # 8×32
B = np.random.randn(16, 16).astype(np.float32)  # 16×16 (wrong K!)
C = np.zeros((8, 16), dtype=np.float32)

gemm.gemm(A, B, C)  # What happens?
```

The behavior is **undefined**. The lowered loop will iterate with `K=32` (from `A`'s second dimension), but `B` only has 16 columns. The code will read out of bounds, likely crashing or producing garbage results.

### Runtime Assertions (Advanced)

Production compilers often insert **runtime checks** to validate shapes:

```mlir
func.func @gemm(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  %M = memref.dim %A, 0 : memref<?x?xf32>
  %K_A = memref.dim %A, 1 : memref<?x?xf32>
  %K_B = memref.dim %B, 0 : memref<?x?xf32>
  %N = memref.dim %B, 1 : memref<?x?xf32>
  
  // Assert: K dimensions must match
  %compatible = arith.cmpi eq, %K_A, %K_B : i64
  assert %compatible, "Incompatible matrix dimensions for GEMM"
  
  // ... rest of computation
}
```

These checks add overhead but prevent subtle bugs. In this book, we omit them for clarity, but they're common in libraries like TensorFlow/MLIR and JAX.

## 2.10 When to Use Static vs Dynamic Shapes

The choice between static and dynamic shapes involves trade-offs:

Static shapes make sense when dimensions are truly fixed—for example, model architectures with constant hidden sizes that never change across batches. They're also preferable when you need maximum performance for small operations, where the overhead of querying descriptors becomes measurable. If you can precompile for a small set of common shapes (say, batch sizes 1, 2, 4, and 8), static shapes let the compiler generate specialized code for each case. Finally, some target hardware has specialized kernels for specific sizes; TensorCore GEMM operations, for instance, require dimensions that are multiples of 8, making static shapes a natural fit.

Dynamic shapes, on the other hand, are essential when dimensions vary at runtime—think variable batch sizes in serving systems or changing sequence lengths in language models. They're also valuable when code size matters, since one dynamic function replaces many static variants. If compilation time becomes a bottleneck, compiling once with dynamic shapes beats compiling repeatedly for each possible dimension combination. Dynamic shapes also provide flexibility crucial for research and rapid prototyping, where you don't want to recompile every time you tweak model dimensions.

Modern serving systems like vLLM and SGLang use hybrid approaches that combine both strategies. They generate specialized kernels for "hot" shapes—commonly occurring dimensions like batch sizes 1, 2, 4, and 8—while falling back to dynamic kernels for rare shapes that don't justify the compilation cost. Some systems use JIT compilation to generate kernels on-demand when new shapes appear, caching the results for future use. This gives the best of both worlds: performance for common cases, flexibility for edge cases. Chapter 14 explores these optimization strategies in detail within the context of GPT inference serving.

## 2.11 Looking Ahead: Bufferization

We deliberately chose to work with memrefs directly in this chapter to avoid the complexity of **bufferization**—the process of converting tensor operations to memref operations. However, it's important to understand this transformation conceptually, as it's central to how production AI compilers work.

### The Tensor-to-MemRef Pipeline

In a typical tensor-based workflow, high-level code uses immutable tensors:

```mlir
func.func @compute(%x: tensor<?x?xf32>, %y: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %sum = arith.addf %x, %y : tensor<?x?xf32>
  %scaled = arith.mulf %sum, %cst : tensor<?x?xf32>
  return %scaled : tensor<?x?xf32>
}
```

Each operation produces a new tensor value. The question bufferization answers: **Where do these tensors actually live in memory?** 

**Connection to Chapter 1's Output Buffer Pattern**: Remember the "output buffer pattern" we introduced in Chapter 1, where functions take pre-allocated output memrefs as parameters? Bufferization is the transformation that converts functional-style tensor operations (which return new values) into imperative-style memref operations (which write to output buffers). This is how `%result = linalg.add(%a, %b)` (tensor version) becomes `linalg.add ins(%a, %b) outs(%result)` (memref version). The bufferization pass determines allocation strategies and inserts the output buffer pattern automatically.

The bufferization pass:

1. **Analyzes tensor dataflow** to determine when values can share storage
2. **Inserts memory allocations** for tensors that need new buffers
3. **Rewrites operations** to use in-place memref updates where possible
4. **Eliminates redundant copies** by aliasing buffers

After bufferization, the code becomes:

```mlir
func.func @compute(%x: memref<?x?xf32>, %y: memref<?x?xf32>, %result: memref<?x?xf32>) {
  // Allocate temporary buffer for sum (if needed)
  %temp = memref.alloc(%M, %N) : memref<?x?xf32>
  
  // In-place operations
  linalg.add ins(%x, %y) outs(%temp)
  linalg.mul ins(%temp, %cst) outs(%result)
  
  // Free temporary
  memref.dealloc %temp : memref<?x?xf32>
}
```

This transformation is non-trivial—it must reason about data dependencies, operation semantics, and memory lifetimes. Chapter 4 explores bufferization concepts in depth, including one-shot bufferization, destination-passing style, and allocation strategies.

### Why We Skipped Tensors

By working directly with memrefs, we've avoided the bufferization pipeline but also lost some optimization opportunities. Tensor-level transformations like **fusion** (combining multiple operations into a single loop) and **tiling** (blocking for cache efficiency) are more straightforward when operations are expressed as immutable tensor computations.

For the matrix multiplications and attention mechanisms in this book, the loss is minor—our operations are already at the right granularity. But for complex computation graphs with many small operations, tensor-based optimization can yield significant performance improvements. Readers building production compilers should study MLIR's bufferization documentation after mastering the memref-level fundamentals presented here.

## 2.12 Summary

This chapter introduced dynamic shapes—the foundation for writing MLIR code that works with matrices and tensors of any size. The key concepts were:

1. **Tensors vs Memrefs**: MLIR distinguishes immutable tensor values (good for optimization) from mutable memory references (necessary for execution). We work with memrefs directly for simplicity.

2. **Dynamic Dimensions**: The `?` notation (backed by `ShapedType::kDynamic`) indicates dimensions unknown at compile time but fixed for each invocation.

3. **MemRef Descriptors**: Runtime structures containing pointers, sizes, strides, and offsets that enable shape-polymorphic code execution.

4. **Shape-Polymorphic Operations**: Operations like `linalg.matmul` work for any compatible dimensions, lowering to loops that query descriptors at runtime.

5. **Minimal Code Changes**: Moving from static to dynamic shapes requires changing only type declarations—the IR generation and lowering logic remain identical.

6. **Performance**: Dynamic shapes add modest overhead (a few extra loads) that's negligible for typical ML workloads.

With dynamic shapes mastered, we can now write MLIR programs that adapt to real-world ML scenarios: variable batch sizes, diverse sequence lengths, and changing input dimensions. The next chapters build on this foundation, exploring JIT compilation caching (Chapter 3) and the conceptual underpinnings of bufferization (Chapter 4) before moving into neural network operations (Chapters 5-9) and eventually production transformer serving (Chapters 10-16).
