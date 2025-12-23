# Chapter 1: Introduction to MLIR

Modern machine learning infrastructure faces a fundamental challenge: **how do you efficiently compile and optimize code that spans multiple levels of abstraction?** This chapter introduces MLIR (Multi-Level Intermediate Representation), a compiler framework designed to address this challenge. We'll explore compiler fundamentals, understand why one IR is insufficient, examine MLIR's solution, and implement our first working example: a JIT-compiled matrix multiply.

---

## 1.1 The Traditional Compiler Pipeline

### What Does a Compiler Do?

At its core, a compiler is a translator. It takes code written in a high-level language and translates it into something your computer can execute. This translation happens in stages, like a factory assembly line:

```
Source Code → [Lexer] → Tokens → [Parser] → AST → [Semantic Analysis] → 
→ [IR Generation] → IR → [Optimization] → Optimized IR → 
→ [Code Generation] → Assembly/Machine Code
```

Let's illustrate with a simple example:

**Source Code:**
```c++
int square(int x) {
    return x * x;
}
```

**Step 1: Lexing**  
The lexer breaks text into meaningful chunks called tokens:
```
[INT] [IDENTIFIER:square] [LPAREN] [INT] [IDENTIFIER:x] [RPAREN]
[LBRACE] [RETURN] [IDENTIFIER:x] [STAR] [IDENTIFIER:x] [SEMICOLON]
[RBRACE]
```

**Step 2: Parsing**  
The parser organizes tokens into an Abstract Syntax Tree (AST):
```
FunctionDecl: square
├── ReturnType: int
├── Parameter: int x
└── Body:
    └── ReturnStmt
        └── BinaryOp: *
            ├── VarRef: x
            └── VarRef: x
```

This tree captures the structure of your code—what it means, not just what characters appear.

**Step 3: Intermediate Representation**  
The AST is converted to IR. For example, LLVM IR:
```llvm
define i32 @square(i32 %x) {
entry:
  %mul = mul nsw i32 %x, %x
  ret i32 %mul
}
```

Let's decode this LLVM IR:

- `define` - Declares a function definition (as opposed to a declaration)
- `i32` - Integer type, 32 bits wide
- `@square` - Function name (the `@` prefix denotes a global symbol, like a function name)
- `%x` - Local variable or register (the `%` prefix denotes a local virtual register)
- `entry:` - A label for a basic block (a sequence of instructions with one entry and one exit)
- `%mul` - Another local register holding the multiplication result
- `mul nsw` - Multiply instruction (nsw = "no signed wrap", a flag indicating overflow behavior)
- `ret` - Return instruction

LLVM IR uses **SSA form** (Static Single Assignment), meaning each "variable" (virtual register) is assigned exactly once. Instead of reusing `%x`, we create new registers like `%mul`. This makes data flow explicit and simplifies optimizations.

The IR is platform-independent but low-level enough to map efficiently to machine instructions. Think of it as a universal assembly language.

### Why Intermediate Representations Matter

Imagine building a translation service for 10 languages. The naive approach requires 10×9 = 90 different translators (each language to every other). A smarter approach uses one common "pivot language"—now you need only 10×2 = 20 translators.

This is exactly what IR does for compilers:

**Benefits of IR:**
1. **Language Independence**: Many source languages (C, C++, Rust, Swift) compile to the same IR
2. **Target Independence**: One IR generates code for many targets (x86, ARM, RISC-V)
3. **Optimization**: Write optimizations once, benefit all languages and targets
4. **Analysis**: Easier to analyze and transform than raw source code

The most famous IR is LLVM IR, which powers compilers for dozens of languages.

### The Structure of IR

Good IR makes everything explicit. Consider addition:

**Source code:**
```c++
x + y
```

**LLVM IR:**
```llvm
%1 = load i32, i32* %x      ; Load value of x
%2 = load i32, i32* %y      ; Load value of y
%3 = add nsw i32 %1, %2     ; Add them (nsw = no signed wrap)
```

**SSA Form (Static Single Assignment)**  
A key property: each variable is assigned exactly once. Instead of reusing variable names, we create new ones:

**Non-SSA (bad):**
```
x = 1
x = x + 2
x = x * 3
```

**SSA (good):**
```
x1 = 1
x2 = x1 + 2
x3 = x2 * 3
```

This makes optimizations easier because you know exactly where each value originates.

---

## 1.2 The Multi-Level Problem

One IR is not enough for modern computing.

### The Rise of Heterogeneous Computing

Consider the journey of a transformer model from high-level Python to silicon:

1. **Framework Level**: PyTorch tensors, attention operations, layer normalization
2. **Graph Level**: Computational graphs with fusion opportunities
3. **Kernel Level**: Matrix multiplies, memory layouts, data movement
4. **Hardware Level**: CPU/GPU instructions, registers, cache hierarchies

When LLVM was created in the early 2000s, computing was simpler:
- Most code ran on CPUs
- CPUs improved through clock speed increases
- One IR served most needs

Today's world is radically different:
- **CPUs** with complex instruction sets
- **GPUs** for parallel computation (CUDA, ROCm)
- **TPUs** (Tensor Processing Units) for machine learning
- **FPGAs** for custom hardware acceleration
- **NPUs** (Neural Processing Units) for AI inference

Each processor has different programming models, optimization opportunities, and constraints.

### The Abstraction Gap

Here's the problem: LLVM IR operates at roughly assembly language level. It's excellent for CPUs but terrible for representing high-level concepts.

**Example: Matrix Multiplication**

At high level, you want to express:
```python
C = A @ B  # Matrix multiplication
```

This operation could be:
- Optimized with tiling and cache blocking
- Parallelized across cores
- Mapped to GPU tensor cores
- Compiled to specialized TPU instructions

But in LLVM IR, you have only loops and memory operations:
```llvm
; Nested loops with thousands of load/store operations
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  ; ... hundreds of lines of low-level code ...
```

**The problem**: By the time you reach LLVM IR, you've lost the high-level structure. You can no longer identify "this is matrix multiplication" and apply matrix-specific optimizations.

### Domain-Specific Needs

Many domains have created specialized languages:

- **Machine Learning**: TensorFlow graphs, PyTorch TorchScript
- **Databases**: SQL query plans and optimization
- **Graphics**: Shader languages (GLSL, HLSL)
- **Scientific Computing**: High-performance Fortran constructs

Each domain wants to:
1. Represent operations naturally at the right abstraction level
2. Apply domain-specific optimizations
3. Eventually compile to efficient machine code

The traditional approach? Each domain builds its complete compiler infrastructure from scratch:
- Reimplementing basic compiler passes
- Reinventing optimization techniques
- Duplicating testing and tooling
- Limited interoperability between systems

There must be a better way.

---

## 1.3 MLIR: A Multi-Level Solution

### The Fragmented Landscape Before MLIR

Before MLIR emerged in 2019, the machine learning compiler landscape was particularly fragmented. Each major framework built its own complete compilation stack:

**TensorFlow** used GraphDef as its internal representation:
- Custom graph IR with fixed operation vocabulary
- Limited extensibility for new operations
- Separate optimization passes reimplemented from scratch
- Difficult to integrate with other frameworks

**PyTorch** developed TorchScript:
- Python-like IR for model serialization
- Ad-hoc optimization strategies
- Separate lowering paths for different backends
- Minimal code sharing with other frameworks

**XLA** (Accelerated Linear Algebra) had HLO (High-Level Optimizer):
- Fixed set of high-level operations
- Good for specific use cases but limited extensibility
- Another isolated compilation stack
- Complex integration with frameworks

Each system reimplemented similar transformations:
- Operator fusion (combining operations to reduce memory traffic)
- Layout optimization (choosing memory layouts for efficiency)
- Device-specific lowering (generating code for GPU, TPU, etc.)
- Shape inference and propagation

This fragmentation meant:
- Thousands of person-hours duplicated across teams
- Innovations in one framework couldn't easily transfer to others
- Each new hardware target required custom integration with every framework
- Compiler optimizations had to be reimplemented for each IR

Google created MLIR to unify this ecosystem. By providing:
1. **Extensible dialects** - each framework can define its own operations
2. **Shared infrastructure** - reusable passes, type system, optimization framework
3. **Interoperability** - different dialects can coexist and transform between each other
4. **Progressive lowering** - systematic path from high-level to hardware-specific code

Today, the ecosystem has consolidated around MLIR:
- **TensorFlow** uses MLIR for compilation and optimization
- **PyTorch 2.0** leverages Torch-MLIR for its compilation backend
- **JAX** uses MLIR (via StableHLO) for its compilation path
- **Serving frameworks** like TensorRT-LLM build on MLIR infrastructure

This unification means optimizations and hardware support developed once benefit the entire ecosystem.

### The Philosophy

MLIR stands for **Multi-Level Intermediate Representation**. The key is "Multi-Level."

Instead of one IR at one abstraction level (like LLVM), MLIR allows multiple IRs at different abstraction levels that coexist and interoperate.

**Traditional approach (LLVM):**
```
High-Level Language → [BIG JUMP] → LLVM IR (low-level) → Machine Code
```

**MLIR approach:**
```
High-Level Language → High-Level IR → Mid-Level IR → Low-Level IR → Machine Code
                       ↓                ↓              ↓
                    (optimize)      (optimize)    (optimize)
```

Each level uses appropriate abstractions for its optimization opportunities.

### The Dialect System

A **dialect** is a collection of operations, types, and attributes forming a cohesive vocabulary—like a namespace for a specific domain or abstraction level. MLIR ships with dozens of built-in dialects:

**High-Level Dialects**:
- `tensor`: Immutable multi-dimensional arrays (functional style)
- `linalg`: Linear algebra operations (matmul, conv, transpose)
- `tosa`: Tensor Operator Set Architecture (NN operations)

**Mid-Level Dialects**:
- `affine`: Polyhedral loop optimization abstractions
- `scf`: Structured Control Flow (for, while, if)
- `vector`: Explicit SIMD vector operations

**Low-Level Dialects**:
- `memref`: Memory references (pointers with shape metadata)
- `arith`: Arithmetic operations (add, mul, cmp)
- `cf`: Control Flow (branches, basic blocks)
- `llvm`: MLIR's representation of LLVM IR

**Utility Dialects**:
- `func`: Function definitions and calls
- `math`: Mathematical functions (exp, log, sqrt)

This hierarchy enables **progressive lowering**: high-level operations transform into mid-level, then low-level, finally reaching LLVM IR or machine code.

### Dialects in This Book

Throughout the book, we'll work primarily with these dialects:

| Dialect | Purpose | Key Operations | Chapters |
|---------|---------|----------------|----------|
| `func` | Function definitions | func.func, func.call, func.return | All |
| `linalg` | Linear algebra | matmul, transpose, fill, generic | 1-11 |
| `memref` | Memory management | alloc, load, store, dealloc | 1-8 |
| `tensor` | Immutable arrays | empty, extract, insert | 2, 4 |
| `arith` | Arithmetic | constant, addf, mulf, cmpf | All |
| `scf` | Control flow loops | for, while, if, yield | 5-7 |
| `math` | Math functions | exp, log, sqrt, tanh | 6-14 |
| `affine` | Polyhedral loops | affine.for, affine.load | 10 |
| `vector` | SIMD operations | vector.load, vector.fma | 10 |
| Custom | Our own dialects | nn.linear, transformer.attention | 8-14 |

### Extensibility: Creating Custom Dialects

Unlike LLVM, MLIR is designed for extensibility. In Chapters 8-9, we'll create a `NeuralNetwork` dialect with operations like:
- `nn.linear`: Fully connected layer
- `nn.layer_norm`: Layer normalization
- `nn.softmax`: Softmax activation

And a `Transformer` dialect with:
- `transformer.attention`: Multi-head attention
- `transformer.ffn`: Feed-forward network
- `transformer.block`: Complete transformer block

These custom operations can carry domain-specific semantics, enabling optimizations that generic compilers cannot perform.

### Reusability Through Infrastructure

MLIR provides infrastructure that all dialects share:

**1. Pass Infrastructure**  
Write transformations once, apply to any dialect:
- Canonicalization (simplification)
- Dead code elimination
- Inlining
- Loop optimizations

**2. Type System**  
Dialects define custom types but share common infrastructure:
- Integer types: `i32`, `i64`
- Floating point: `f32`, `f64`
- Tensors: `tensor<2x3xf64>`
- Custom types: `!mydialect.mytype`

**3. Attribute System**  
Attach compile-time constants and metadata to operations.

**4. Region System**  
Operations can contain nested regions of code (like function bodies).

**5. Interfaces**  
Define common behavior across dialects, enabling generic algorithms. For example, a `ShapeInferenceInterface` can work with any operation that implements it.

### Why This Matters

Consider building a machine learning framework:

**Without MLIR** (traditional approach):
- Build custom IR for ML operations from scratch
- Implement custom optimization passes
- Write custom code generation for each target
- Thousands of person-hours of duplicated work

**With MLIR** (composable approach):
- Use `tensor` and `linalg` dialects for ML operations (already exist)
- Leverage built-in optimization infrastructure
- Lower to `gpu` dialect for GPUs, `llvm` dialect for CPUs
- Reuse LLVM's mature code generation
- Focus on unique value-add, not infrastructure

This is why major projects have adopted MLIR: TensorFlow, PyTorch 2.0, JAX, and serving frameworks like TensorRT-LLM.

---

## 1.4 Lowering: The Heart of MLIR Compilation

The central concept in MLIR is **progressive lowering**—transforming operations from high-level abstractions to low-level implementations through a series of **passes**.

### What is a Pass?

A pass is a transformation that walks the IR and rewrites it. Passes can:
- **Lower** operations (replace high-level ops with low-level ones)
- **Optimize** code (eliminate redundancy, fuse operations)
- **Canonicalize** patterns (simplify to standard forms)

### The Lowering Pipeline

A typical MLIR compilation pipeline for matrix multiply:

```
┌─────────────────────┐
│  linalg.matmul      │  High-level: "multiply these matrices"
│  (declarative)      │
└──────────┬──────────┘
           │ Linalg-to-Loops Pass
           ▼
┌─────────────────────┐
│  scf.for (nested)   │  Mid-level: explicit loops with iterators
│  memref.load        │
│  arith.mulf, addf   │
└──────────┬──────────┘
           │ SCF-to-CF Pass
           ▼
┌─────────────────────┐
│  cf.br, cf.cond_br  │  Low-level: basic blocks and branches
│  llvm.load, llvm.   │
└──────────┬──────────┘
           │ Convert-to-LLVM Pass
           ▼
┌─────────────────────┐
│  llvm.* operations  │  LLVM dialect (ready for LLVM backend)
└──────────┬──────────┘
           │ LLVM Backend
           ▼
┌─────────────────────┐
│  Machine Code       │  x86-64 or ARM assembly
└─────────────────────┘
```

Each pass is **composable**—you can add, remove, or reorder passes to create custom pipelines. This flexibility is crucial for ML workloads where optimization strategies vary by model architecture and hardware target.

### Pass Example: Linalg to Loops

Consider `linalg.matmul`:
```mlir
linalg.matmul ins(%A, %B : memref<8x32xf32>, memref<32x16xf32>)
              outs(%C : memref<8x16xf32>)
```

After the Linalg-to-Loops pass:
```mlir
scf.for %i = 0 to 8 {
  scf.for %j = 0 to 16 {
    scf.for %k = 0 to 32 {
      %a = memref.load %A[%i, %k] : memref<8x32xf32>
      %b = memref.load %B[%k, %j] : memref<32x16xf32>
      %c = memref.load %C[%i, %j] : memref<8x16xf32>
      %prod = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %prod : f32
      memref.store %sum, %C[%i, %j] : memref<8x16xf32>
    }
  }
}
```

The transformation is mechanical but powerful: it exposes the loop structure for subsequent optimizations (tiling, vectorization, parallelization).

---

## 1.5 Backends: AOT vs JIT Compilation

Once we've lowered to LLVM dialect, we have two execution strategies:

### Ahead-of-Time (AOT) Compilation

**Process**:
1. Compile MLIR → LLVM IR → object files
2. Link object files into shared library or executable
3. Load and execute at runtime

**Advantages**:
- No compilation overhead at runtime
- Can perform expensive optimizations
- Suitable for production deployment

**Disadvantages**:
- Requires separate compilation step
- Cannot specialize for runtime values
- Longer development iteration cycle

**Use cases**: Deploying fixed models in production servers, embedded systems, mobile apps.

### Just-in-Time (JIT) Compilation

**Process**:
1. Generate MLIR at runtime
2. Apply passes and lower to LLVM
3. JIT compile to machine code
4. Execute immediately

**Advantages**:
- Immediate feedback during development
- Can specialize for runtime values (shape, batch size)
- Dynamic optimization opportunities

**Disadvantages**:
- Compilation overhead (milliseconds per function)
- Requires compiler toolchain at runtime
- Memory overhead for IR and compiled code

**Use cases**: Research, experimentation, dynamic models, interactive development. This is the model used by PyTorch's `torch.compile()` and JAX's `jit()`.

### Our Approach: JIT Throughout

This book uses JIT compilation exclusively for three reasons:

1. **Immediate feedback**: Compile and test in seconds, not minutes
2. **Pedagogical clarity**: See IR transformations interactively
3. **Python integration**: Natural fit for Python-first ML workflows

Chapter 3 will address JIT's compilation overhead through caching—compile once, reuse many times.

---

## 1.6 Your First MLIR Program: Matrix Multiply

Now that we understand the foundations, let's implement a complete example. We'll build an 8×32 × 32×16 matrix multiply using MLIR's `linalg` dialect, compile it with JIT, and execute it from Python.

### Problem Statement

Implement:
```
C = A @ B

where:
  A: 8 × 32 matrix (float32)
  B: 32 × 16 matrix (float32)  
  C: 8 × 16 matrix (float32)
```

This is GEMM (General Matrix Multiply)—the fundamental operation in neural networks. Every linear layer, attention mechanism, and feed-forward network relies on matrix multiplication.

### The High-Level MLIR IR

Our goal is to generate this MLIR code programmatically:

```mlir
func.func @gemm_8x16x32(%arg0: memref<8x32xf32>,   // Input A
                        %arg1: memref<32x16xf32>,  // Input B
                        %arg2: memref<8x16xf32>) { // Output C (in-place)
  %cst = arith.constant 0.0 : f32                   // Zero constant
  linalg.fill ins(%cst) outs(%arg2)                 // C = zeros
  linalg.matmul ins(%arg0, %arg1) outs(%arg2)       // C += A @ B
  return
}
```

Let's break this down:

**Function signature**: `func.func @gemm_8x16x32`
- Uses the `func` dialect for function definitions
- Takes 3 arguments: A, B, and C (all memrefs—think "pointers to arrays")

**Types**: `memref<8x32xf32>`
- `memref` = memory reference (like a pointer with shape info)
- `8x32` = 2D shape
- `f32` = 32-bit float
- We use memrefs instead of tensors to avoid complexity (Chapter 4 will explain)

**Operations**:
1. `arith.constant 0.0` - Create a zero value
2. `linalg.fill` - Initialize output matrix to zeros
3. `linalg.matmul` - The actual matrix multiply (declarative!)

**Key Insight**: The `linalg.matmul` operation is *declarative*—it specifies **what** to compute, not **how**. There are no explicit loops, no SIMD instructions, no memory layout decisions. MLIR's optimization passes will make those decisions during lowering.

---

## 1.7 Generating IR with the C++ API

MLIR provides a C++ API for programmatic IR construction. Let's walk through the implementation in `ch.1.Fixed-size/src/ir.cpp`.

### Loading Dialects

```cpp
OwningOpRef<ModuleOp> createGemmModule(MLIRContext& context) {
  // Load required dialects
  context.getOrLoadDialect<func::FuncDialect>();    // func.func, func.return
  context.getOrLoadDialect<linalg::LinalgDialect>(); // linalg.matmul
  context.getOrLoadDialect<memref::MemRefDialect>(); // memref types
  context.getOrLoadDialect<arith::ArithDialect>();   // arith.constant
```

**Dialects are vocabularies**. Each dialect provides a set of operations. We must load the ones we need before using their operations.

### Creating the Module and Builder

```cpp
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();  // Source location (not tracked here)
  auto module = ModuleOp::create(loc); // Top-level container
  
  // CRITICAL: Set insertion point
  builder.setInsertionPointToStart(module.getBody());
```

- `OpBuilder` is the tool for constructing operations
- `ModuleOp` is the top-level container
- **The insertion point tells the builder where to add new operations**—forgetting to set it is a common error

### Defining Types

```cpp
  auto f32Type = builder.getF32Type();
  auto matrixA_type = MemRefType::get({8, 32}, f32Type);  // 8×32
  auto matrixB_type = MemRefType::get({32, 16}, f32Type); // 32×16
  auto matrixC_type = MemRefType::get({8, 16}, f32Type);  // 8×16
  
  auto funcType = builder.getFunctionType(
    {matrixA_type, matrixB_type, matrixC_type},  // inputs
    {}                                           // no return (in-place)
  );
```

MLIR types are first-class objects. We create memref types with explicit shapes. The fixed sizes (8, 32, 16) simplify the implementation; Chapter 2 introduces dynamic shapes.

### Creating the Function

```cpp
  auto funcOp = builder.create<func::FuncOp>(loc, "gemm_8x16x32", funcType);
  funcOp.setPublic();  // External linkage
  
  // Create function body
  auto* entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  
  // Get function arguments (the memrefs)
  auto matrixA = entryBlock->getArgument(0);
  auto matrixB = entryBlock->getArgument(1);
  auto matrixC = entryBlock->getArgument(2);
```

- `func::FuncOp` represents a function
- We add an entry block to hold operations
- Arguments are accessed via `getArgument(index)`

### Generating the Matrix Multiply

```cpp
  // Create zero constant for initialization
  auto zeroAttr = builder.getFloatAttr(f32Type, 0.0);
  auto zero = builder.create<arith::ConstantOp>(loc, zeroAttr);
  
  // Initialize output to zeros: C = 0
  builder.create<linalg::FillOp>(loc, zero.getResult(), matrixC);
  
  // Matrix multiply: C += A @ B
  builder.create<linalg::MatmulOp>(loc, 
                                    ValueRange{matrixA, matrixB},  // inputs
                                    ValueRange{matrixC});          // output
  
  // Return (void function)
  builder.create<func::ReturnOp>(loc);
  
  return module;
}
```

Notice the abstraction level: we're not writing loops or specifying memory access patterns. The `linalg.matmul` operation is declarative—it carries semantic meaning ("matrix multiply") that later passes will expand into efficient implementation.

---

## 1.8 The Compilation Pipeline

With IR generated, we apply a series of passes to lower it to executable machine code. The implementation resides in `lowering.cpp`.

### The Pass Sequence

```cpp
void applyOptimizationPasses(ModuleOp module) {
  PassManager pm(module.getContext());
  
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertArithToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  
  if (failed(pm.run(module))) {
    llvm::errs() << "Pass manager failed\n";
  }
}
```

### Understanding Each Pass

**Canonicalization**  
Simplifies patterns like `x + 0 → x`, removes dead code, and normalizes IR to standard forms. This is a general cleanup pass run between major transformations.

**Linalg → Loops**  
Converts `linalg.matmul` into nested `scf.for` loops with explicit memory accesses:
  ```mlir
  // Before:
  linalg.matmul ins(%A, %B) outs(%C)
  
  // After:
  scf.for %i = 0 to 8 {
    scf.for %j = 0 to 16 {
      scf.for %k = 0 to 32 {
        %a = memref.load %A[%i, %k]
        %b = memref.load %B[%k, %j]
        %c = memref.load %C[%i, %j]
        %prod = arith.mulf %a, %b
        %sum = arith.addf %c, %prod
        memref.store %sum, %C[%i, %j]
      }
    }
  }
  ```

The transformation is mechanical: one declarative operation becomes explicit loops. This exposes the computation for subsequent optimizations like tiling, vectorization, and parallelization.

**SCF → CF**  
Converts structured loops (`scf.for`) into basic blocks and branches (`cf.br`). Structured control flow is easier to analyze and optimize; control flow graph representation prepares for LLVM lowering.

**MemRef → LLVM**  
Converts memref types to LLVM pointers and lowers memory operations (`memref.load` → `llvm.load`). Chapter 2 will reveal the complexity hidden here: memrefs with dynamic shapes require descriptor structs with stride and offset information.

**Final Lowering Passes**  
The remaining three passes convert all high-level dialects (func, arith, cf) to LLVM dialect. After this pipeline, we have pure LLVM IR in MLIR's representation—ready for JIT compilation.

---

## 1.9 JIT Compilation and Execution

With lowered LLVM dialect IR, we use MLIR's ExecutionEngine to compile and execute it. The implementation is in `jit.cpp`.

```cpp
void* jitCompileFunction(ModuleOp module) {
  ExecutionEngineOptions options;
  options.transformer = mlir::makeOptimizingTransformer(3, 0, nullptr);
  
  auto engine = mlir::ExecutionEngine::create(module, options);
  if (!engine) {
    llvm::errs() << "Failed to create execution engine\n";
    return nullptr;
  }
  
  auto result = (*engine)->lookup("gemm_8x16x32");
  if (!result) {
    llvm::errs() << "Failed to find function\n";
    return nullptr;
  }
  
  return reinterpret_cast<void*>(*result);
}
```

The `makeOptimizingTransformer` applies LLVM's `-O3` optimizations (instruction selection, register allocation, etc.). We link against the C standard library for malloc and free—memref operations require dynamic memory allocation.

### Invoking the Compiled Function

```cpp
typedef void (*GemmFunc)(float*, float*, float*);

void callGemmJIT(void* funcPtr, float* A, float* B, float* C) {
  auto gemm = reinterpret_cast<GemmFunc>(funcPtr);
  gemm(A, B, C);
}
```

We cast the void pointer to a function pointer matching our MLIR signature. In memory, memrefs are simply float pointers for fixed-size arrays. Chapter 2 will reveal the additional complexity for dynamic shapes.

### Python Integration

Pybind11 exposes our JIT compiler to Python (implementation in `bindings.cpp`):

```cpp
py::array_t<float> gemm(py::array_t<float> A, py::array_t<float> B) {
  // Validate shapes
  if (A.shape(0) != 8 || A.shape(1) != 32)
    throw std::runtime_error("A must be 8x32");
  if (B.shape(0) != 32 || B.shape(1) != 16)
    throw std::runtime_error("B must be 32x16");
  
  // Allocate output
  py::array_t<float> C({8, 16});
  
  // Get raw pointers
  float* ptrA = A.mutable_data();
  float* ptrB = B.mutable_data();
  float* ptrC = C.mutable_data();
  
  // JIT compile and execute
  auto module = createGemmModule(context);
  applyOptimizationPasses(module);
  void* funcPtr = jitCompileFunction(module);
  callGemmJIT(funcPtr, ptrA, ptrB, ptrC);
  return C;
}

PYBIND11_MODULE(ch1_fixed_size, m) {
  m.def("gemm", &gemm, "JIT-compiled GEMM");
}
```

This workflow validates shapes, generates IR, applies passes, JIT compiles, and executes—all triggered by a Python function call. Chapter 3 will optimize this by caching compiled functions.

---

## 1.10 Testing and Validation

We verify correctness by comparing against NumPy's optimized BLAS implementation (from `test_jit.py`):

```python
import numpy as np
import ch1_fixed_size

# Test 1: Ones matrix (easy manual verification)
print("Test 1: Ones matrix")
A = np.ones((8, 32), dtype=np.float32)
B = np.ones((32, 16), dtype=np.float32)
C = ch1_fixed_size.gemm(A, B)

# Each element should be 32.0 (sum of 32 ones)
expected = 32.0
assert np.allclose(C, expected), f"Expected {expected}, got {C[0,0]}"
print(f"✓ Result: all elements = {C[0,0]} (correct!)")

# Test 2: Random matrices (compare with NumPy)
print("\nTest 2: Random matrices")
A = np.random.randn(8, 32).astype(np.float32)
B = np.random.randn(32, 16).astype(np.float32)

C_ours = ch1_fixed_size.gemm(A, B)
C_numpy = A @ B

# Check numerical accuracy
diff = np.abs(C_ours - C_numpy)
max_diff = np.max(diff)
print(f"Max difference vs NumPy: {max_diff}")
assert np.allclose(C_ours, C_numpy, rtol=1e-5), "Mismatch with NumPy!"
print("✓ Matches NumPy (correct!)")
```

The ones matrix test provides easy manual verification: each element should be 32.0 (the sum of 32 multiplications of 1.0 × 1.0). The random matrix test ensures our implementation matches NumPy's battle-tested BLAS routines. Small differences (around 10⁻⁷) are expected due to floating-point rounding.

---

## 1.11 Summary

This chapter introduced MLIR's foundational concepts and implemented a complete JIT-compiled matrix multiply. We covered:

**Conceptual Foundations**:
- The multi-level problem in ML compilation
- LLVM IR as the low-level foundation
- MLIR's dialect system for extensible operation vocabularies
- Progressive lowering through composable passes
- AOT vs JIT compilation strategies

**Practical Implementation**:
- Generating MLIR IR programmatically with the C++ API
- Applying a pass pipeline (Linalg → SCF → CF → LLVM)
- JIT compilation with ExecutionEngine
- Python integration via pybind11
- Testing against NumPy for correctness

**Key Insights**:
- Declarative operations (`linalg.matmul`) separate "what" from "how"
- Passes progressively lower abstraction levels
- Fixed-size memrefs simplify the interface (dynamic shapes add complexity)
- JIT enables immediate feedback but introduces compilation overhead

**Design Choices**:
- MemRefs instead of tensors (avoids bufferization complexity for now)
- Fixed dimensions (8×32×16) for pedagogical clarity
- JIT compilation for rapid iteration

Chapter 2 tackles the next challenge: handling arbitrary matrix sizes with dynamic shapes. This introduces the memref descriptor—a 21-parameter structure that will significantly complicate our implementation.