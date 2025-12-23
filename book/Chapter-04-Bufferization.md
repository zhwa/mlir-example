# Chapter 4: Bufferization - Bridging Functional and Imperative Worlds

In Chapters 1-3, we worked directly with memrefs—mutable memory buffers that behave like C pointers. This imperative style is intuitive for programmers familiar with C/C++, but it has limitations. Modern MLIR code typically starts with **tensors**—immutable values with functional semantics—and transforms them into memrefs through a process called **bufferization**.

This chapter answers a fundamental question: **Why does MLIR have two ways to represent multi-dimensional data (tensors and memrefs), and how do you convert between them?** Understanding bufferization is crucial because it's the bridge between high-level ML frameworks (which think in terms of immutable tensors) and low-level execution (which requires mutable memory buffers).

The transformation is non-trivial. Converting functional semantics (where operations return new values) to imperative semantics (where operations mutate memory in-place) requires careful analysis to ensure correctness and efficiency. MLIR's One-Shot Bufferize pass solves this problem systematically.

## 4.1 Why Two Representations? Tensors vs MemRefs

MLIR's dual representation—tensors and memrefs—reflects a fundamental tension in compiler design. Each serves different needs at different compilation stages.

### The Tensor Abstraction: Functional Semantics

**Tensors** in MLIR represent immutable, value-semantic multi-dimensional arrays. Think of them like integers or floats—when you add two integers, you don't mutate them; you create a new result:

```mlir
%a = arith.constant 5 : i32
%b = arith.constant 3 : i32
%c = arith.addi %a, %b : i32  // Creates new value, doesn't mutate %a or %b
```

Tensors work the same way:

```mlir
func.func @add_tensors(%A: tensor<8x16xf32>, %B: tensor<8x16xf32>) 
    -> tensor<8x16xf32> {
  // Create new output tensor
  %empty = tensor.empty() : tensor<8x16xf32>
  
  // Compute result (returns NEW tensor, doesn't mutate inputs)
  %result = linalg.add ins(%A, %B : tensor<8x16xf32>, tensor<8x16xf32>)
                       outs(%empty : tensor<8x16xf32>)
                       -> tensor<8x16xf32>
  
  return %result : tensor<8x16xf32>
}
```

Key characteristics:

**Immutability**: Once created, a tensor's values never change. Operations produce new tensors rather than modifying existing ones.

**SSA (Static Single Assignment)**: Each tensor value has exactly one defining operation. You never reassign or mutate. This makes data flow explicit in the IR.

**No aliasing**: Since tensors are immutable, two tensor values never refer to the same underlying storage. No pointer aliasing issues.

**Optimization-friendly**: Functional semantics enable aggressive transformations. The compiler can reorder, fuse, or eliminate operations freely without worrying about side effects.

### The MemRef Reality: Imperative Semantics

**MemRefs** represent mutable memory buffers—pointers to storage that can be read from and written to:

```mlir
func.func @add_memrefs(%A: memref<8x16xf32>, 
                        %B: memref<8x16xf32>,
                        %C: memref<8x16xf32>) {
  // C is an OUT-PARAMETER (caller-allocated)
  // We MUTATE its contents in-place
  linalg.add ins(%A, %B : memref<8x16xf32>, memref<8x16xf32>)
             outs(%C : memref<8x16xf32>)
  return
}
```

Key characteristics:

**Mutability**: MemRefs point to storage that can be modified. Operations mutate the contents in-place.

**Aliasing**: Multiple memrefs can point to the same memory. Aliasing analysis becomes essential for correctness.

**Explicit memory management**: Allocations (`memref.alloc`), deallocations (`memref.dealloc`), and lifetimes must be managed explicitly.

**Execution reality**: This is how programs actually execute. CPUs operate on memory addresses, not abstract immutable values.

### Why Both? Different Stages, Different Needs

The duality exists because different compilation stages have different priorities:

**High-level optimization** (framework → MLIR):
- **Input**: Models from PyTorch, TensorFlow, JAX (all tensor-based)
- **Priority**: Aggressive optimization (fusion, reordering, constant folding)
- **Representation**: Tensors (functional, optimization-friendly)
- **Freedom**: Can reorder operations, fuse ops, eliminate redundancy without worrying about side effects

**Low-level compilation** (MLIR → machine code):
- **Output**: Assembly code for CPU/GPU
- **Priority**: Correct memory management, efficient execution
- **Representation**: MemRefs (imperative, matches hardware)
- **Constraints**: Must respect memory layout, cache hierarchy, call ABI

Bufferization is the bridge between these worlds. It converts high-level tensor IR (optimized easily) into low-level memref IR (executable efficiently).

### A Concrete Example: Matrix Addition

**Tensor version** (functional):
```mlir
func.func @add(%A: tensor<8x16xf32>, %B: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %empty = tensor.empty() : tensor<8x16xf32>
  %result = linalg.add ins(%A, %B) outs(%empty) -> tensor<8x16xf32>
  return %result : tensor<8x16xf32>
}
```

Semantics: "Given two tensor values A and B, create a new tensor that holds their elementwise sum."

**MemRef version** (imperative):
```mlir
func.func @add(%A: memref<8x16xf32>, %B: memref<8x16xf32>, %C: memref<8x16xf32>) {
  linalg.add ins(%A, %B) outs(%C)
  return
}
```

Semantics: "Given three memory buffers A, B, C, write A[i,j] + B[i,j] into C[i,j] for all i,j."

Notice the signature change: the tensor version **returns** a result (functional), while the memref version takes an **out-parameter** (imperative). This transformation—converting return values to out-parameters—is a key part of bufferization.

### The Optimization Story

Why go through this trouble? Because tensor-level optimizations are dramatically simpler and more powerful:

**Fusion** at tensor level:
```mlir
%t1 = linalg.add ins(%A, %B) outs(%empty1) -> tensor<8x16xf32>
%t2 = linalg.mul ins(%t1, %C) outs(%empty2) -> tensor<8x16xf32>
// Can fuse: single loop computing A+B*C
```

**Fusion** at memref level requires alias analysis:
```mlir
linalg.add ins(%A, %B) outs(%T1)  // Does %T1 alias %A or %B?
linalg.mul ins(%T1, %C) outs(%T2) // Does %T2 alias %T1 or %C?
// Fusion only safe if no aliasing
```

With tensors, immutability guarantees no aliasing—fusion is always safe. With memrefs, the compiler must conservatively check aliasing, often missing optimization opportunities.

## 4.2 One-Shot Bufferization: The Transformation Architecture

Bufferization is the process of converting tensor-based IR to memref-based IR. MLIR provides **One-Shot Bufferize**, a sophisticated pass that handles this transformation systematically.

### The Challenge: More Than Simple Substitution

Naïve bufferization might seem simple: replace `tensor<8x16xf32>` with `memref<8x16xf32>`. But this approach fails immediately:

```mlir
// Tensor version (before)
func.func @compute(%arg: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %result = some_op(%arg) -> tensor<8x16xf32>
  return %result
}

// Naïve "bufferization" (WRONG!)
func.func @compute(%arg: memref<8x16xf32>) -> memref<8x16xf32> {
  %result = some_op(%arg) -> memref<8x16xf32>  // Where does this memref come from?
  return %result  // Returning a pointer? Who allocated it? Who frees it?
}
```

The problems:

1. **Memory allocation**: Where do memrefs come from? Who allocates them?
2. **Ownership**: Who is responsible for freeing allocated memory?
3. **ABI mismatch**: Returning pointers is problematic (who owns the memory?)
4. **Mutability**: Tensors are immutable values; memrefs are mutable pointers. Different semantics!

One-Shot Bufferize solves these problems through a multi-step transformation.

### The Read-After-Write Problem

Before diving into the algorithm, we need to understand a critical correctness issue that naive bufferization can introduce: the **read-after-write (RAW) hazard**.

Consider this seemingly innocent tensor code:

```mlir
func.func @problem(%input: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %c1 = arith.constant 1.0 : f32
  
  // Step 1: Add 1.0 to input
  %modified = linalg.map ins(%input : tensor<8xf32>)
                         outs(%empty1 : tensor<8xf32>) {
    ^bb0(%val: f32):
      %sum = arith.addf %val, %c1 : f32
      linalg.yield %sum : f32
  } -> tensor<8xf32>
  
  // Step 2: Return BOTH original and modified
  return %input, %modified : tensor<8xf32>, tensor<8xf32>
}
```

With tensors, this is perfectly safe: `%input` is immutable, so returning both original and modified values makes sense. They're different values.

**Naive bufferization** (WRONG!):
```mlir
func.func @problem(%input: memref<8xf32>,
                   %out1: memref<8xf32>,
                   %out2: memref<8xf32>) {
  // Mutate %input in-place to save memory
  linalg.map ins(%input) outs(%input) {  // ← BUG! Writing to %input
    ^bb0(%val: f32):
      %sum = arith.addf %val, %c1
      linalg.yield %sum : f32
  }
  
  // Copy results
  memref.copy %input, %out1  // ← Should be original, but it's modified!
  memref.copy %input, %out2  // ← Same as out1 (both modified)
}
```

**The bug**: We wanted to return the original `%input` unchanged, but naive in-place bufferization modified it. This is a **read-after-write hazard**—we read `%input` after writing to it, expecting the old value, but got the new value.

This is a **correctness bug**, not just an optimization issue. Naive bufferization produces wrong results!

### The One-Shot Bufferize Algorithm

One-Shot Bufferize performs a whole-program analysis to **prevent read-after-write hazards** while maximizing in-place updates. The algorithm:

**Step 1: Alias analysis and conflict detection**
- Analyze the entire function to understand tensor data flow
- Determine which tensors can share buffers (in-place updates) **safely**
- Build an alias set: which operations create new values vs reuse existing buffers
- **Detect RAW conflicts**: If a tensor is read after being potentially written, mark it as requiring a separate buffer

**Step 2: Buffer allocation insertion**
- Insert `memref.alloc` operations where new buffers are needed
- Place allocations as late as possible (minimize live ranges)
- Deallocations inserted automatically (based on liveness analysis)
- **Allocate separate buffers** when RAW analysis detects conflicts

**Step 3: Type conversion**
- Replace `tensor<...>` types with `memref<...>` types throughout
- Update operation signatures to use memrefs
- Convert tensor operations to memref operations

**Step 4: In-place mutation (when safe)**
- Where possible, convert functional "return new tensor" style to imperative "mutate buffer" style
- This eliminates unnecessary copies
- **Only when alias analysis proves no RAW hazards exist**

### Example: Bufferization in Action

**Before** (tensor IR):
```mlir
func.func @matmul_add(%A: tensor<8x32xf32>,
                      %B: tensor<32x16xf32>,
                      %C: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // Create empty output tensor
  %empty = tensor.empty() : tensor<8x16xf32>
  
  // Matrix multiply (returns new tensor)
  %matmul = linalg.matmul ins(%A, %B : tensor<8x32xf32>, tensor<32x16xf32>)
                          outs(%empty : tensor<8x16xf32>)
                          -> tensor<8x16xf32>
  
  // Add (returns new tensor)
  %result = linalg.add ins(%matmul, %C : tensor<8x16xf32>, tensor<8x16xf32>)
                       outs(%empty : tensor<8x16xf32>)
                       -> tensor<8x16xf32>
  
  return %result : tensor<8x16xf32>
}
```

**After** One-Shot Bufferize (conceptual):
```mlir
func.func @matmul_add(%A: memref<8x32xf32>,
                      %B: memref<32x16xf32>,
                      %C: memref<8x16xf32>,
                      %out: memref<8x16xf32>) {  // Out-parameter added!
  // Allocate temporary buffer for matmul result
  %temp = memref.alloc() : memref<8x16xf32>
  
  // Matrix multiply (mutates %temp in-place)
  linalg.matmul ins(%A, %B : memref<8x32xf32>, memref<32x16xf32>)
                outs(%temp : memref<8x16xf32>)
  
  // Add (mutates %out in-place)
  linalg.add ins(%temp, %C : memref<8x16xf32>, memref<8x16xf32>)
             outs(%out : memref<8x16xf32>)
  
  // Deallocate temporary
  memref.dealloc %temp : memref<8x16xf32>
  
  return
}
```

Key transformations:
1. Return type removed, replaced with out-parameter `%out`
2. `tensor.empty` → `memref.alloc` (explicit allocation)
3. Operations now mutate buffers in-place (no return values)
4. Temporary buffer allocated for intermediate result
5. Deallocation inserted for temporary (memory management)

### Configuration: bufferizeFunctionBoundaries

One-Shot Bufferize has a critical option: **bufferizeFunctionBoundaries**. This controls whether the pass bufferizes function signatures (arguments and return types) or only function bodies.

```cpp
bufferization::OneShotBufferizationOptions options;
options.bufferizeFunctionBoundaries = true;  // Bufferize signatures
```

**With `bufferizeFunctionBoundaries = true`** (what we use):
- Function signatures change: `tensor<...>` → `memref<...>`
- Return types converted to out-parameters
- The entire IR becomes memref-based

**With `bufferizeFunctionBoundaries = false`**:
- Function signatures unchanged (still use tensors)
- Only function bodies bufferized
- Useful for gradual migration or when interfacing with tensor-based APIs

For standalone compilation (Chapters 1-4), we use `true` because we're compiling the entire program to native code. For library APIs or when integrating with Python (where tensors might be more natural), `false` can be appropriate.

### Layout Maps: Controlling Memory Layout

Bufferization must decide memory layout—how multi-dimensional tensors map to linear memory. The layout map specifies this:

```cpp
options.setFunctionBoundaryTypeConversion(
    bufferization::LayoutMapOption::IdentityLayoutMap);
```

**IdentityLayoutMap** means "use row-major (C-style) layout with unit strides":
- For `tensor<8x16xf32>` → `memref<8x16xf32>` with default strides
- Element `[i,j]` stored at offset `i*16 + j`
- Matches NumPy default layout (`C_CONTIGUOUS`)

Other options exist for specialized layouts (column-major, tiled, padded), but identity layout is standard for CPU execution and Python integration.

## 4.3 Buffer-Results-To-Out-Params: ABI Transformation

After One-Shot Bufferize converts tensors to memrefs, we still have a problem: functions might return memrefs. This is problematic for calling conventions and memory management. The **Buffer-Results-To-Out-Params** pass solves this.

### The Problem: Returning Pointers

Consider a bufferized function that returns a memref:

```mlir
func.func @compute(%A: memref<8x16xf32>) -> memref<8x16xf32> {
  %result = memref.alloc() : memref<8x16xf32>
  // ... compute into %result ...
  return %result : memref<8x16xf32>
}
```

Issues:
1. **Ownership ambiguity**: Who owns the returned memref? Caller or callee?
2. **Deallocation responsibility**: Who calls `memref.dealloc`? If caller, they must track ownership. If callee, can't return the buffer!
3. **ABI complexity**: Returning pointers requires special calling conventions (NRVO, RVO in C++)
4. **Optimization barriers**: Return values complicate inlining and optimization

### The Solution: Out-Parameters

The Buffer-Results-To-Out-Params pass transforms return values into out-parameters:

**Before**:
```mlir
func.func @compute(%A: memref<8x16xf32>) -> memref<8x16xf32> {
  %result = memref.alloc() : memref<8x16xf32>
  // ... compute into %result ...
  return %result : memref<8x16xf32>
}
```

**After**:
```mlir
func.func @compute(%A: memref<8x16xf32>, %out: memref<8x16xf32>) {
  // Caller provides %out (already allocated!)
  // ... compute into %out ...
  return  // No return value
}
```

Now:
- Caller allocates output buffer before calling
- Callee writes results into caller-provided buffer
- No ambiguity about ownership (caller owns everything)
- Standard C calling convention (all arguments passed, nothing returned)

### How It Works: Signature Transformation

The pass analyzes function signatures and transforms them:

**Pattern**: Any function returning one or more memrefs gets those return values moved to out-parameters.

**Original**:
```mlir
func.func @multi_return(%in: memref<8xf32>) 
    -> (memref<8xf32>, memref<8xf32>) {
  %out1 = memref.alloc() : memref<8xf32>
  %out2 = memref.alloc() : memref<8xf32>
  // ... compute ...
  return %out1, %out2 : memref<8xf32>, memref<8xf32>
}
```

**Transformed**:
```mlir
func.func @multi_return(%in: memref<8xf32>,
                        %out1: memref<8xf32>,  // New parameter
                        %out2: memref<8xf32>) {// New parameter
  // Caller-allocated buffers, just write to them
  // ... compute into %out1 and %out2 ...
  return  // No return values
}
```

Call sites are updated automatically:
```mlir
// Before: %r1, %r2 = call @multi_return(%arg)
// After:
%r1 = memref.alloc() : memref<8xf32>
%r2 = memref.alloc() : memref<8xf32>
call @multi_return(%arg, %r1, %r2)
// Now %r1, %r2 contain results
```

### Why This Matters for Python Integration

This transformation is crucial for Python bindings. Python's ctypes and pybind11 work naturally with functions that take all arguments as inputs:

```python
# Memref with return value (hard to bind)
result_ptr = lib.compute(input_ptr)  # Who frees result_ptr?

# Memref with out-parameter (easy to bind)
result = np.zeros((8, 16), dtype=np.float32)  # Python allocates
lib.compute(input_ptr, result.ctypes.data)    # C++ fills it in
# Python manages result lifetime automatically
```

With out-parameters, Python controls memory allocation and deallocation through NumPy's garbage collector. No manual memory management needed.

### Performance Benefits

Out-parameters enable important optimizations:

**Caller-side allocation**:
```python
# Reuse buffer across calls (amortize allocation)
result = np.zeros((8, 16), dtype=np.float32)
for i in range(1000):
    lib.compute(inputs[i], result.ctypes.data)  # Same buffer each time
```

**Pre-allocated output arrays**:
```python
# Allocate once, reuse forever
output_buffer = np.zeros((1024, 1024), dtype=np.float32)
for batch in data_loader:
    lib.process(batch, output_buffer)  # Zero allocation cost
```

This pattern is fundamental to high-performance Python libraries (NumPy, PyTorch, TensorFlow). By matching this calling convention, MLIR-compiled code integrates seamlessly.

## 4.4 The Complete Bufferization Pipeline

Let's trace a complete example through all bufferization passes to see how each transformation builds on the previous one.

### Starting Point: Tensor-Based IR

Our input is idiomatic high-level MLIR with tensors:

```mlir
func.func @gemm(%A: tensor<?x?xf32>,
                %B: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // Extract dimensions from input tensors
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %A, %c0 : tensor<?x?xf32>
  %K = tensor.dim %A, %c1 : tensor<?x?xf32>
  %N = tensor.dim %B, %c1 : tensor<?x?xf32>
  
  // Create empty output tensor (shape: MxN)
  %empty = tensor.empty(%M, %N) : tensor<?x?xf32>
  
  // Initialize to zero
  %cst = arith.constant 0.0 : f32
  %zero_filled = linalg.fill ins(%cst : f32)
                             outs(%empty : tensor<?x?xf32>)
                             -> tensor<?x?xf32>
  
  // Matrix multiplication
  %result = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                          outs(%zero_filled : tensor<?x?xf32>)
                          -> tensor<?x?xf32>
  
  return %result : tensor<?x?xf32>
}
```

Characteristics:
- Purely functional (no mutation)
- Operations return new tensors
- Dynamic shapes (`?` dimensions)
- Shape queries (`tensor.dim`)
- Declarative operations (`linalg.fill`, `linalg.matmul`)

### Pass 1: Canonicalization

First, we simplify the IR with general cleanup passes:

```mlir
// Canonical form (minor simplifications)
func.func @gemm(%A: tensor<?x?xf32>,
                %B: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %A, %c0 : tensor<?x?xf32>
  %N = tensor.dim %B, %c1 : tensor<?x?xf32>
  
  %empty = tensor.empty(%M, %N) : tensor<?x?xf32>
  
  %cst = arith.constant 0.0 : f32
  %zero_filled = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  
  %result = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                          outs(%zero_filled : tensor<?x?xf32>)
                          -> tensor<?x?xf32>
  
  return %result : tensor<?x?xf32>
}
```

Changes are minimal (constant folding, dead code elimination). The IR structure remains tensor-based.

### Pass 2: One-Shot Bufferize

This is where the magic happens. The entire tensor IR transforms to memref IR:

```mlir
func.func @gemm(%A: memref<?x?xf32>,  // tensor → memref
                %B: memref<?x?xf32>) -> memref<?x?xf32> {  // Still returns memref
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  
  // tensor.dim → memref.dim
  %M = memref.dim %A, %c0 : memref<?x?xf32>
  %N = memref.dim %B, %c1 : memref<?x?xf32>
  
  // tensor.empty → memref.alloc (explicit allocation!)
  %buffer = memref.alloc(%M, %N) : memref<?x?xf32>
  
  %cst = arith.constant 0.0 : f32
  // linalg.fill now operates on memrefs (in-place mutation)
  linalg.fill ins(%cst : f32) outs(%buffer : memref<?x?xf32>)
  
  // linalg.matmul operates on memrefs (in-place mutation)
  linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
                outs(%buffer : memref<?x?xf32>)
  
  return %buffer : memref<?x?xf32>
}
```

Key changes:
- All `tensor<...>` → `memref<...>`
- `tensor.empty` → `memref.alloc` (explicit allocation)
- `tensor.dim` → `memref.dim` (same operation, different operand type)
- Linalg operations no longer return values (mutate outputs in-place)
- Function still returns memref (problem for next pass!)

### Pass 3: Buffer-Results-To-Out-Params

Transform return value to out-parameter:

```mlir
func.func @gemm(%A: memref<?x?xf32>,
                %B: memref<?x?xf32>,
                %C: memref<?x?xf32>) {  // Out-parameter added! No return value!
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %A, %c0 : memref<?x?xf32>
  %N = memref.dim %B, %c1 : memref<?x?xf32>
  
  // No longer allocates! Uses caller-provided %C
  
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%C : memref<?x?xf32>)
  
  linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
                outs(%C : memref<?x?xf32>)
  
  return  // No return value
}
```

Critical changes:
- New parameter `%C` added (output buffer)
- `memref.alloc` removed (caller allocates!)
- Operations write to `%C` instead of temporary buffer
- Function returns void
- Signature matches C calling convention

### Pass 4: Bufferization-To-MemRef

Lower remaining bufferization dialect operations to memref dialect. This is mostly cleanup—converting any bufferization-specific ops that weren't already lowered:

```mlir
// (In our example, IR unchanged—already fully memref-based)
func.func @gemm(%A: memref<?x?xf32>,
                %B: memref<?x?xf32>,
                %C: memref<?x?xf32>) {
  // ... same as before ...
}
```

### Passes 5-7: Progressive Lowering to LLVM

After bufferization, the remaining passes lower from high-level memref/linalg operations to low-level LLVM IR. We covered this in Chapter 3, but here's a quick recap in the context of bufferization:

**Pass 5: Linalg → Loops**
```mlir
func.func @gemm(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %M = memref.dim %A, %c0 : memref<?x?xf32>
  %K = memref.dim %A, %c1 : memref<?x?xf32>
  %N = memref.dim %B, %c1 : memref<?x?xf32>
  
  // linalg.fill → explicit loop
  scf.for %i = %c0 to %M step %c1 {
    scf.for %j = %c0 to %N step %c1 {
      memref.store %cst, %C[%i, %j] : memref<?x?xf32>
    }
  }
  
  // linalg.matmul → triple nested loop
  scf.for %i = %c0 to %M step %c1 {
    scf.for %j = %c0 to %N step %c1 {
      scf.for %k = %c0 to %K step %c1 {
        %a = memref.load %A[%i, %k] : memref<?x?xf32>
        %b = memref.load %B[%k, %j] : memref<?x?xf32>
        %c = memref.load %C[%i, %j] : memref<?x?xf32>
        %prod = arith.mulf %a, %b : f32
        %sum = arith.addf %c, %prod : f32
        memref.store %sum, %C[%i, %j] : memref<?x?xf32>
      }
    }
  }
  return
}
```

**Pass 6: SCF → Control Flow** (loops to branches)

**Pass 7: Everything → LLVM Dialect** (memref operations to LLVM pointer arithmetic)

The final LLVM dialect IR is ready for compilation to machine code.

### The Complete Pipeline in Code

Here's how the passes are configured in C++:

```cpp
LogicalResult applyOptimizationPasses(ModuleOp module) {
  MLIRContext* context = module.getContext();
  PassManager pm(context);

  // Register bufferization interfaces
  DialectRegistry registry;
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  context->appendDialectRegistry(registry);

  // Phase 1: Simplification
  pm.addPass(createCanonicalizerPass());

  // Phase 2: Bufferization (tensor → memref)
  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  pm.addPass(bufferization::createOneShotBufferizePass(options));
  pm.addPass(bufferization::createBufferResultsToOutParamsPass());
  pm.addPass(createBufferizationToMemRefPass());
  pm.addPass(createCanonicalizerPass());

  // Phase 3: Progressive lowering (Linalg → Loops → CF → LLVM)
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  return pm.run(module);
}
```

## 4.5 Clean Python APIs: Hiding Complexity

The bufferization pipeline produces functions with out-parameter signatures (`func(in1, in2, out)` instead of `func(in1, in2) -> result`). While this is efficient for execution, it's awkward for users. The Python binding layer hides this complexity.

### The User Experience: What We Want

From a user's perspective, matrix multiplication should be simple:

```python
import ch4_tensor_bufferization as gemm
import numpy as np

A = np.random.randn(100, 200).astype(np.float32)
B = np.random.randn(200, 150).astype(np.float32)

# Natural API: input → output
C = gemm.gemm(A, B)  # Returns result, no manual allocation!
```

No mention of output buffers, no manual allocation, no memory management. The API feels like NumPy's `np.matmul(A, B)`.

### The Implementation: Automatic Allocation

Behind the scenes, the binding code handles allocation:

```cpp
py::array_t<float> gemm(py::array_t<float> A, py::array_t<float> B) {
  // Extract shapes from input arrays
  auto buf_A = A.request();
  auto buf_B = B.request();
  
  if (buf_A.ndim != 2 || buf_B.ndim != 2) {
    throw std::runtime_error("Inputs must be 2D arrays");
  }
  
  int64_t M = buf_A.shape[0];  // Rows in A
  int64_t K = buf_A.shape[1];  // Cols in A
  int64_t K_B = buf_B.shape[0];  // Rows in B
  int64_t N = buf_B.shape[1];  // Cols in B
  
  if (K != K_B) {
    throw std::runtime_error("Inner dimensions must match");
  }
  
  // AUTOMATIC ALLOCATION: Create output array
  auto C = py::array_t<float>({M, N});
  auto buf_C = C.request();
  
  // Call JIT-compiled function with out-parameter
  mlir::executeGemm(
      static_cast<float*>(buf_A.ptr),
      static_cast<float*>(buf_B.ptr),
      static_cast<float*>(buf_C.ptr),
      M, N, K
  );
  
  // Return the populated output array
  return C;
}
```

The binding allocates the output array, calls the MLIR function (which fills it in), and returns it to Python. The user never sees the out-parameter complexity.

### Shape Validation and Error Handling

Good bindings validate inputs before calling compiled code:

```cpp
// Check dimensions are compatible
if (A.shape[1] != B.shape[0]) {
  throw std::runtime_error(
    "Matrix dimension mismatch: A is " +
    std::to_string(A.shape[0]) + "x" + std::to_string(A.shape[1]) +
    ", B is " + std::to_string(B.shape[0]) + "x" + std::to_string(B.shape[1])
  );
}

// Check data types
if (A.dtype() != pybind11::dtype::of<float>()) {
  throw std::runtime_error("A must be float32, got " + std::string(A.dtype().str()));
}
```

This provides clear error messages at the Python level instead of mysterious crashes in compiled code.

### Memory Reuse: Advanced Pattern

For performance-critical code, you can expose an in-place API that reuses buffers:

```cpp
// In-place API: user provides output buffer
void gemm_inplace(py::array_t<float> A,
                  py::array_t<float> B,
                  py::array_t<float> C) {  // User-allocated!
  // Validate shapes
  if (A.shape[0] != C.shape[0] || B.shape[1] != C.shape[1]) {
    throw std::runtime_error("Output shape doesn't match A @ B");
  }
  
  // Call MLIR function
  mlir::executeGemm(A.ptr, B.ptr, C.ptr, ...);
  // C is modified in-place
}
```

Usage:
```python
# Preallocate output buffer (reuse across calls)
C = np.zeros((100, 150), dtype=np.float32)

for i in range(1000):
    gemm.gemm_inplace(A_list[i], B_list[i], C)
    # C contains result, no allocation overhead
```

This pattern is crucial for low-latency serving systems where allocation overhead is unacceptable.

### The Layered API Design

A well-designed binding layer offers multiple levels:

**Level 1: Beginner-friendly** (automatic allocation):
```python
C = gemm.gemm(A, B)  # Simple, allocates result
```

**Level 2: Performance-conscious** (reuse buffers):
```python
C = np.empty((M, N), dtype=np.float32)
gemm.gemm_inplace(A, B, C)  # No allocation, faster
```

**Level 3: Expert** (full control):
```python
# Access raw function pointer for zero-overhead calls
func_ptr = gemm.get_compiled_function()
func_ptr(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, N, K)
```

Each level trades convenience for control. Most users stay at Level 1; performance engineers use Level 2; system developers might use Level 3 for integration with other C++ code.

## 4.6 MemRef Layout Maps and Zero-Cost Transformations

MemRefs are not just pointers—they're **descriptors** that bundle a base pointer with metadata about how to interpret that memory. A critical piece of this metadata is the **layout map**, which controls how multi-dimensional indices map to flat memory offsets. Understanding layout maps is essential because they enable **zero-cost transformations** like transpose and reshape.

### The Anatomy of a MemRef: More Than a Pointer

In C, an array is just a pointer. In MLIR, a `memref` is a structured descriptor containing:

1. **Allocated pointer**: Base address returned by `malloc` (for alignment tracking)
2. **Aligned pointer**: Actual data start (may be offset from allocated for alignment)
3. **Offset**: Additional offset from aligned pointer to first element
4. **Sizes**: Dimension sizes (e.g., `[8, 16]` for 8×16 matrix)
5. **Strides**: Step size in elements between adjacent indices (e.g., `[16, 1]` for row-major)
6. **Layout map** (optional): Affine transformation from indices to offsets

The layout map is what makes MemRefs powerful. It determines how `memref.load %M[%i, %j]` computes the physical memory address.

### Layout Maps: The Indexing Recipe

A **layout map** is an **affine map** that transforms logical indices to physical offsets. The general form:

```
physical_offset = map(i, j, k, ...) + base_offset
```

**Identity layout** (default, row-major):
```mlir
#identity_map = affine_map<(d0, d1) -> (d0, d1)>
memref<8x16xf32>  // Same as memref<8x16xf32, #identity_map>
```

For element `[i, j]`, offset = `i * stride[0] + j * stride[1]` = `i * 16 + j * 1`

This is standard C/NumPy row-major layout: consecutive elements in the last dimension are adjacent in memory.

**Transposed layout** (column-major):
```mlir
#transpose_map = affine_map<(d0, d1) -> (d1, d0)>
memref<8x16xf32, #transpose_map>
```

For element `[i, j]`, the map swaps indices: offset = `j * stride[0] + i * stride[1]` = `j * 8 + i * 1`

This is column-major (Fortran) layout: consecutive elements in the *first* dimension are adjacent.

### Permutation Maps: Zero-Cost Reordering

A **permutation map** is a layout map that only reorders dimensions—no arithmetic, just index shuffling. This enables **zero-cost transformations**:

**Problem**: You want to transpose a matrix (swap rows and columns).

**Naive approach** (expensive):
```mlir
func.func @transpose_naive(%input: memref<8x16xf32>) -> memref<16x8xf32> {
  %output = memref.alloc() : memref<16x8xf32>
  
  // Copy with swapped indices (16×8 loads + 16×8 stores)
  scf.for %i = 0 to 8 {
    scf.for %j = 0 to 16 {
      %val = memref.load %input[%i, %j] : memref<8x16xf32>
      memref.store %val, %output[%j, %i] : memref<16x8xf32>
    }
  }
  return %output
}
```

Cost: 128 loads + 128 stores = **256 memory operations** + allocation overhead.

**Smart approach with permutation map** (free!):
```mlir
#transpose = affine_map<(d0, d1) -> (d1, d0)>

func.func @transpose_smart(%input: memref<8x16xf32>) -> memref<16x8xf32, #transpose> {
  // Just reinterpret the layout—NO data movement!
  %transposed = memref.cast %input : memref<8x16xf32> to memref<16x8xf32, #transpose>
  return %transposed
}
```

Cost: **Zero memory operations**. Just changed the metadata (layout map).

The physical data remains in the same memory locations. When you `load %transposed[i, j]`, it internally loads `%input[j, i]`—the permutation map adjusts the indexing logic.

### How Bufferization Uses Layout Maps

When One-Shot Bufferize encounters tensor operations with layout changes (transpose, reshape, broadcast), it tries to avoid data copies by using layout maps:

**High-level (Tensor IR)**:
```mlir
func.func @example(%A: tensor<8x16xf32>) -> tensor<16x8xf32> {
  // Transpose operation (functional semantics)
  %transposed = linalg.transpose ins(%A : tensor<8x16xf32>)
                                 outs(%empty : tensor<16x8xf32>)
                                 permutation = [1, 0]
                                 -> tensor<16x8xf32>
  return %transposed
}
```

**After Bufferization** (MemRef IR with permutation map):
```mlir
#transpose_map = affine_map<(d0, d1) -> (d1, d0)>

func.func @example(%A: memref<8x16xf32>, %out: memref<16x8xf32>) {
  // If %out is just a view, use cast (zero cost)
  // If %out is a separate buffer, must copy
  
  // Smart path: Create view with layout map
  %view = memref.cast %A : memref<8x16xf32> to memref<16x8xf32, #transpose_map>
  
  // Copy from view to output (unavoidable due to out-param ABI)
  linalg.copy ins(%view : memref<16x8xf32, #transpose_map>)
              outs(%out : memref<16x8xf32>)
  return
}
```

The bufferizer recognizes that transpose is just an indexing change and uses a permutation map. The physical copy only happens when writing to the out-parameter (which the caller allocated).

### Real-World Example: Multi-Head Attention Reshape

Transformer attention requires frequent reshapes between `[batch, seq, hidden]` and `[batch, heads, seq, head_dim]`:

**Without layout maps** (4 memory copies per attention layer):
```mlir
%reshaped = memref.alloc() : memref<2x8x128x64xf32>
// Copy all data with new indexing
```

**With layout maps** (zero extra memory):
```mlir
#reshape_map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 * 64 + d3)>
%reshaped = memref.cast %input : memref<2x8x128x64xf32, #reshape_map>
// No data movement!
```

For a 12-layer transformer, this saves **48 memory copies** (4 per layer × 12 layers)—potentially gigabytes of memory traffic.

### When Layout Maps Can't Help

Permutation maps only work when the transformation is a pure reordering. They **cannot** help with:

**Data-dependent operations**: Gather, scatter, masked stores
**Padding/striding**: Dilated convolutions, strided slices
**Non-affine indexing**: Dynamic indices computed from data values

In these cases, bufferization must allocate new buffers and physically copy/compute data.

### Summary: The MemRef Descriptor

Putting it all together, a MemRef like `memref<8x16xf32, #transpose_map>` contains:

| Field | Value | Purpose |
|-------|-------|---------|
| Allocated ptr | `0x1000000` | Base for deallocation |
| Aligned ptr | `0x1000000` | Actual data start |
| Offset | `0` | Additional offset |
| Sizes | `[8, 16]` | Logical dimensions |
| Strides | `[1, 8]` | Physical strides (transposed!) |
| Layout map | `(d0,d1)->(d1,d0)` | Index transformation |

When you `load [i, j]`:
1. Apply layout map: `(i, j)` → `(j, i)`
2. Compute offset: `j * stride[0] + i * stride[1]` = `j * 1 + i * 8`
3. Load from: `aligned_ptr + offset`

This architecture enables zero-cost logical transformations while maintaining straightforward physical memory access.

## 4.7 Comparing Approaches: Direct MemRef vs Tensor + Bufferization

We now have experience with two approaches to generating MLIR IR:

**Chapters 1-3**: Direct memref-based IR
**Chapter 4**: Tensor-based IR with bufferization

Let's compare them systematically.

### Direct MemRef Approach (Chapters 1-3)

**Code generation**:
```cpp
auto matrixA_type = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32Type);
auto funcType = builder.getFunctionType({matrixA_type, matrixB_type, matrixC_type}, {});
auto funcOp = builder.create<func::FuncOp>(loc, "gemm", funcType);

// Function body directly operates on memrefs
Value argA = entryBlock->getArgument(0);
Value argC = entryBlock->getArgument(2);
builder.create<linalg::MatmulOp>(loc, TypeRange{}, ValueRange{argA, argB}, ValueRange{argC});
```

**Advantages**:
- Simpler: No bufferization passes needed
- Direct: IR matches execution model (mutable buffers)
- Fewer passes: Shorter compilation pipeline
- Easier to debug: What you write is what executes

**Disadvantages**:
- Less optimization: Aliasing restricts transformations
- Manual memory: Must think about buffer lifetimes from the start
- Not idiomatic: Most ML frameworks produce tensor IR
- Limited fusion: Alias analysis required for safety

### Tensor + Bufferization Approach (Chapter 4)

**Code generation**:
```cpp
auto tensorA_type = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32Type);
auto funcType = builder.getFunctionType({tensorA_type, tensorB_type}, {resultTensorType});
auto funcOp = builder.create<func::FuncOp>(loc, "gemm", funcType);

// Function body works with immutable tensors
Value empty = builder.create<tensor::EmptyOp>(loc, ...);
Value result = builder.create<linalg::MatmulOp>(loc, resultType, ValueRange{argA, argB}, ValueRange{empty});
builder.create<func::ReturnOp>(loc, result);
```

**Advantages**:
- Better optimization: Functional semantics enable aggressive transformations
- Idiomatic: Matches how ML frameworks represent models
- Automatic memory: Bufferization inserts allocations intelligently
- More fusion: No aliasing concerns at tensor level

**Disadvantages**:
- More complex: Additional passes and concepts to understand
- Longer compilation: Bufferization adds overhead
- Debugging harder: Multiple IR transformations to trace
- Learning curve: Must understand bufferization semantics

### When to Use Each Approach

**Use direct memrefs** when:
- Building low-level libraries where you need full control over memory layout
- Interfacing with existing C APIs that use pointers
- Implementing custom memory management strategies
- Working on memory-constrained systems (embedded, mobile)
- Compilation time is critical (development, interactive use)

**Use tensors + bufferization** when:
- Building high-level ML compilers (framework frontends)
- Optimization is the top priority (production serving)
- Interfacing with tensor-based systems (PyTorch, TensorFlow, JAX)
- Leveraging MLIR's optimization passes (fusion, tiling, vectorization)
- Following MLIR best practices for new code

### The Practical Reality

In practice, production MLIR systems use a **hybrid approach**:

1. **Frontend** (framework → MLIR): Generates tensor-based IR
   - Import PyTorch/TensorFlow models as tensor ops
   - Apply high-level optimizations (fusion, constant folding)
   - Perform shape inference and specialization

2. **Middle-end** (optimization): Operates on tensors
   - Aggressive fusion (attention, FFN, etc.)
   - Tiling for cache locality
   - Vectorization

3. **Bufferization** (tensor → memref): One-Shot Bufferize pass
   - Converts to imperative form
   - Inserts allocations and deallocations
   - Adjusts calling conventions (out-parameters)

4. **Backend** (lowering → code gen): Operates on memrefs
   - Lower to loops
   - Target-specific optimizations
   - Code generation

Each stage uses the representation best suited to its task. Tensors for optimization, memrefs for execution.

## 4.8 Bufferization in the ML Compilation Stack

To understand where bufferization fits in real systems, let's examine how production ML compilers use it.

### TensorFlow → XLA → MLIR

TensorFlow's XLA (Accelerated Linear Algebra) compiler uses MLIR internally:

**Stage 1: TensorFlow Graph → MLIR HLO** (High-Level Operations)
- Tensor-based operations (`mhlo.add`, `mhlo.dot`, etc.)
- Functional semantics throughout
- Graph-level optimizations (constant folding, algebraic simplification)

**Stage 2: HLO Optimizations**
- Operation fusion (combine multiple ops into fused kernels)
- Layout assignment (decide NCHW vs NHWC, etc.)
- All optimizations on tensor IR (no aliasing concerns)

**Stage 3: Bufferization**
- One-Shot Bufferize converts tensors → memrefs
- Insert buffer allocations for intermediate results
- Convert to imperative calling conventions

**Stage 4: Lowering to LLVM/PTX**
- MemRef-based IR lowers to target code
- CPU: LLVM IR → x86/ARM assembly
- GPU: LLVM IR → PTX (NVIDIA) or AMDGPU code

Bufferization happens late, after all high-level optimizations complete.

### PyTorch → torch-mlir → LLVM

PyTorch 2.0's `torch.compile` can use MLIR through torch-mlir:

**Stage 1: PyTorch Model → TorchScript/FX Graph**
- Trace or script Python model
- Capture computation graph

**Stage 2: TorchScript → torch-mlir**
- Import into `torch` dialect (tensor-based)
- Operations like `torch.aten.matmul`, `torch.aten.relu`

**Stage 3: torch dialect → linalg dialect**
- Lower high-level torch ops to linalg structured ops
- Still tensor-based
- Enable generic optimizations (tiling, vectorization)

**Stage 4: Bufferization**
- One-Shot Bufferize (linalg tensors → linalg memrefs)
- Memory management inserted

**Stage 5: Lowering to execution**
- Linalg → Loops → LLVM
- Generate CPU/GPU code

Again, bufferization is the bridge between high-level optimization and low-level execution.

### The Pattern: Optimize High, Execute Low

All modern ML compilers follow this pattern:

```
High-level IR (tensors)
  → Optimizations (fusion, rewriting)
  → Bufferization (tensor → memref)
  → Code generation (memref → assembly)
```

Bufferization is the **phase transition**—the boundary between functional (optimization-friendly) and imperative (execution-ready) representations.

## 4.9 Advanced Topics: Copy Elimination and In-Place Updates

One-Shot Bufferize doesn't just mechanically convert tensors to memrefs—it performs sophisticated analysis to minimize memory traffic.

### The Copy Problem

Naïve bufferization would allocate a new buffer for every tensor operation:

```mlir
// Tensor IR
%t1 = linalg.add ins(%A, %B) outs(%empty1) -> tensor<1024x1024xf32>
%t2 = linalg.mul ins(%t1, %C) outs(%empty2) -> tensor<1024x1024xf32>
%t3 = linalg.relu ins(%t2) outs(%empty3) -> tensor<1024x1024xf32>

// Naïve bufferization
%buf1 = memref.alloc() : memref<1024x1024xf32>
linalg.add ins(%A, %B) outs(%buf1)  // Write to buf1

%buf2 = memref.alloc() : memref<1024x1024xf32>
linalg.mul ins(%buf1, %C) outs(%buf2)  // Read buf1, write to buf2

%buf3 = memref.alloc() : memref<1024x1024xf32>
linalg.relu ins(%buf2) outs(%buf3)  // Read buf2, write to buf3
```

This allocates three 16MB buffers (1024×1024×4 bytes each) and performs 32MB of memory writes. Wasteful!

### In-Place Updates

One-Shot Bufferize analyzes tensor usage to enable **in-place updates**—reusing buffers when safe:

```mlir
// Smart bufferization (single buffer reused)
%buf = memref.alloc() : memref<1024x1024xf32>
linalg.add ins(%A, %B) outs(%buf)    // Write to buf
linalg.mul ins(%buf, %C) outs(%buf)  // REUSE buf (in-place!)
linalg.relu ins(%buf) outs(%buf)     // REUSE buf (in-place!)
```

Only one 16MB buffer, writes happen in-place. Memory usage reduced 3x, memory traffic reduced dramatically.

### When Is In-Place Safe?

In-place updates are safe when:

1. **No future reads**: The input tensor isn't used after the operation
2. **Exclusive ownership**: No other operations reference the buffer
3. **Compatible shape**: Input and output have identical shapes/strides

One-Shot Bufferize performs **liveness analysis** to determine when these conditions hold.

**Example where in-place is safe**:
```mlir
%t1 = some_op(%t0)
%t2 = other_op(%t1)  // %t1 never used again after this
// ✓ Can reuse %t1's buffer for %t2 (in-place)
```

**Example where in-place is unsafe**:
```mlir
%t1 = some_op(%t0)
%t2 = other_op(%t1)
%t3 = final_op(%t1, %t2)  // %t1 used again!
// ✗ Cannot reuse %t1's buffer—still needed for %t3
```

### Controlling Bufferization Behavior

One-Shot Bufferize provides options to control in-place updates:

```cpp
bufferization::OneShotBufferizationOptions options;

// Allow in-place updates (default)
options.allowReturnAllocsFromLoops = true;

// More conservative (safer but more copies)
options.allowReturnAllocsFromLoops = false;

// Control memory alignment
options.setMemorySpace(0);  // Default memory space
```

For learning and development, defaults work well. For production optimization, tuning these options can significantly reduce memory usage.

## 4.10 Summary

This chapter covered bufferization—MLIR's transformation from functional tensor IR to imperative memref IR. The key insights:

1. **Two representations, two purposes**: Tensors enable optimization (functional, no aliasing), memrefs enable execution (imperative, matches hardware). Each serves a critical role in the compilation pipeline.

2. **Read-after-write hazards**: Naive bufferization can produce incorrect results when tensors are read after being written. One-Shot Bufferize performs alias analysis and conflict detection to prevent these bugs while maximizing in-place updates.

3. **One-Shot Bufferize**: A sophisticated whole-program transformation that converts tensors to memrefs while analyzing aliasing, inserting allocations, and enabling safe in-place updates. Not a simple type substitution.

4. **Buffer-Results-To-Out-Params**: Converts return values to out-parameters, matching C calling conventions and enabling clean Python bindings. Essential for interoperability.

5. **MemRef layout maps and permutation maps**: MemRefs are descriptors bundling pointers with metadata including layout maps. Permutation maps enable zero-cost transformations (transpose, reshape) by changing indexing logic without moving data—a critical optimization for transformer operations.

6. **The complete pipeline**: Canonicalization → One-Shot Bufferize → Buffer-Results-To-Out-Params → Bufferization-To-MemRef → Progressive lowering. Each pass builds on previous transformations.

7. **Clean Python APIs**: Binding layers hide complexity, automatically allocating output buffers and presenting natural APIs (`C = gemm(A, B)`) while internally calling out-parameter functions.

8. **Optimization story**: Tensor-level optimizations (fusion, rewriting) are simpler and more powerful than memref-level optimizations. Bufferize late in the pipeline, after high-level transformations complete.

9. **Production pattern**: Modern ML compilers universally follow the pattern: high-level tensor IR → optimization → bufferization → low-level memref IR → code generation.

With bufferization mastered, we've completed the foundational infrastructure (Chapters 1-4). We now understand:
- IR generation (Chapter 1)
- Dynamic shapes (Chapter 2)
- Compilation strategies and pass infrastructure (Chapter 3)
- Functional-to-imperative transformation with correctness guarantees (Chapter 4)

We're ready to build increasingly sophisticated operations, starting with neural network primitives in Part II.