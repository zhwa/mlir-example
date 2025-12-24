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

This is exactly what IR does for compilers. By introducing an intermediate representation between source languages and target machines, we dramatically reduce complexity while enabling powerful optimizations. When you compile C++ to ARM, the compiler first translates C++ to IR, then IR to ARM instructions. When a Rust developer targets x86, their compiler translates Rust to that same IR, then to x86. This means that optimizations written for the IR automatically benefit every language and every target—you write an optimization once, and it works for all combinations.

IR provides **language independence** because many source languages (C, C++, Rust, Swift, and dozens more) compile to the same intermediate form. It offers **target independence** because that same IR can generate efficient code for many hardware targets (x86, ARM, RISC-V, and more). Beyond simplifying the compiler architecture, IR makes **optimization** practical at scale: writing a dead code elimination pass once in the IR benefits every language targeting every platform. IR also makes **analysis** easier—its explicit structure and constraints (like SSA form) make it simpler to reason about program behavior than raw source code with its implicit semantics and complex control flow.

The most famous IR is LLVM IR, which powers compilers for dozens of languages and enables the optimization passes that make modern compilers competitive with hand-tuned assembly.

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
A key property: each *value* (virtual register) is assigned exactly once. Instead of reusing variable names, we create new ones:

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

**CRITICAL DISTINCTION**: SSA applies to *values* (virtual registers like `%1`, `%2`), not to *memory*. In MLIR, a memref pointer like `%A` is immutable (SSA), but the data it points to is mutable—you can store different values into the same memory location. Think of SSA as "each pointer is assigned once," not "each memory cell is written once." This distinction becomes crucial in Chapters 2-4 when we work with buffers and understand why operations like `memref.store` can write to the same address multiple times.

This makes optimizations easier because you know exactly where each value originates.

---

## 1.2 The Multi-Level Problem

One IR is not enough for modern computing.

### The Rise of Heterogeneous Computing

Consider the journey of a transformer model from high-level Python to silicon. At the framework level, you write code using PyTorch tensors, attention operations, and layer normalization. These high-level abstractions get compiled into computational graphs where the compiler identifies fusion opportunities. The graph then lowers to kernel-level operations: matrix multiplies with specific memory layouts and optimized data movement. Finally, this generates hardware-level instructions for CPU/GPU execution, including register allocation and cache hierarchy management.

When LLVM was created in the early 2000s, computing was simpler. Most code ran on CPUs, which improved performance primarily through clock speed increases. One intermediate representation served most compilation needs, since the variation between hardware targets was limited.

Today's world is radically different. Modern systems combine CPUs with complex instruction sets, GPUs for parallel computation (CUDA, ROCm), TPUs (Tensor Processing Units) optimized for machine learning workloads, FPGAs for custom hardware acceleration, and NPUs (Neural Processing Units) specialized for AI inference. Each processor has different programming models, optimization opportunities, and constraints. A compiler that generates optimal code for one target may generate inefficient code for another.

Each processor has different programming models, optimization opportunities, and constraints.

### The Abstraction Gap

Here's the problem: LLVM IR operates at roughly assembly language level. It's excellent for CPUs but terrible for representing high-level concepts.

**Example: Matrix Multiplication**

At high level, you want to express:
```python
C = A @ B  # Matrix multiplication
```

This simple operation opens a vast optimization space. It could be optimized with tiling and cache blocking to maximize data reuse in CPU caches. It could be parallelized across cores to leverage multicore CPUs. It could be mapped to GPU tensor cores, which provide specialized hardware for matrix operations. For specialized accelerators, it could be compiled to TPU instructions optimized for dense linear algebra. The "what" (matrix multiplication) remains constant, but the "how" (implementation strategy) varies dramatically by target hardware.

But in LLVM IR, you have only loops and memory operations:
```llvm
; Nested loops with thousands of load/store operations
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  ; ... hundreds of lines of low-level code ...
```

**The problem**: By the time you reach LLVM IR, you've lost the high-level structure. You can no longer identify "this is matrix multiplication" and apply matrix-specific optimizations.

### Domain-Specific Needs

Many domains have created specialized languages to work at appropriate abstraction levels. Machine learning uses TensorFlow graphs and PyTorch TorchScript to represent neural network operations. Databases rely on SQL query plans with specialized optimization rules. Graphics uses shader languages like GLSL and HLSL for parallel rendering operations. Scientific computing extends Fortran with high-performance constructs for array operations.

Each domain shares common goals: represent operations naturally at the right abstraction level, apply domain-specific optimizations that generic compilers cannot perform, and eventually compile to efficient machine code for the target hardware. The challenge is that these goals require compiler infrastructure—type systems, optimization frameworks, code generation—that's expensive to build and maintain.

The traditional approach has each domain building its complete compiler infrastructure from scratch. This means reimplementing basic compiler passes that are conceptually identical across domains (like dead code elimination or constant folding), reinventing optimization techniques that have been solved elsewhere, duplicating testing and tooling infrastructure, and ending up with limited interoperability between systems. A machine learning framework can't easily leverage optimizations from a database compiler, even when they're solving similar problems. There must be a better way.

---

## 1.3 MLIR: A Multi-Level Solution

### The Fragmented Landscape Before MLIR

Before MLIR emerged in 2019, the machine learning compiler landscape was particularly fragmented. Each major framework built its own complete compilation stack.

**TensorFlow** used GraphDef as its internal representation. This custom graph IR had a fixed operation vocabulary, which limited extensibility when researchers wanted to add new operations. TensorFlow implemented its own optimization passes from scratch, duplicating work done elsewhere. The system's architecture made it difficult to integrate with other frameworks—TensorFlow operations couldn't easily interoperate with other IRs.

**PyTorch** developed TorchScript, a Python-like IR for model serialization. While this made PyTorch models portable, the optimization strategies were ad-hoc rather than systematic. Different backends (CPU, CUDA, mobile) required separate lowering paths, each implemented independently. The framework shared minimal code with other projects, meaning optimizations developed for PyTorch couldn't benefit other frameworks.

**XLA** (Accelerated Linear Algebra) had HLO (High-Level Optimizer), which worked well for specific use cases but offered limited extensibility. Google developed HLO primarily for TPUs, and adapting it to new operations or hardware required deep system modifications. Like TensorFlow and PyTorch, XLA represented another isolated compilation stack with complex integration challenges.

Each system reimplemented similar transformations. Operator fusion (combining operations to reduce memory traffic) appeared in all three frameworks but with different implementations. Layout optimization (choosing memory layouts for efficiency) was solved independently multiple times. Device-specific lowering (generating code for GPU, TPU, etc.) required separate implementations in each framework. Shape inference and propagation—determining tensor shapes throughout the graph—was duplicated across all systems.

This fragmentation imposed significant costs. Thousands of person-hours were duplicated across teams solving identical problems. Innovations in one framework couldn't easily transfer to others—a clever optimization in TensorFlow couldn't be ported to PyTorch without substantial reengineering. Each new hardware target required custom integration with every framework, multiplying the integration burden. Compiler optimizations had to be reimplemented for each IR, wasting engineering effort that could have been spent on novel research.

Google created MLIR to unify this ecosystem. MLIR provides **extensible dialects**, allowing each framework to define its own operations while sharing infrastructure. It offers **shared infrastructure** including reusable passes, a type system, and an optimization framework that all dialects can leverage. Different dialects can **interoperate**, coexisting in the same program and transforming between each other. Finally, MLIR enables **progressive lowering** through a systematic path from high-level to hardware-specific code, where each transformation is explicit and composable.

Today, the ecosystem has consolidated around MLIR. TensorFlow uses MLIR for compilation and optimization. PyTorch 2.0 leverages Torch-MLIR for its compilation backend. JAX uses MLIR (via StableHLO) for its compilation path. Serving frameworks like TensorRT-LLM build on MLIR infrastructure.
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

A **dialect** collects operations, types, and attributes into a cohesive vocabulary—like a namespace for a specific domain or abstraction level. Dialects organize MLIR's operation vocabulary into logical categories, each serving a specific purpose in the compilation pipeline. MLIR ships with dozens of built-in dialects spanning different abstraction levels.

At the **high level**, dialects like `tensor` provide immutable multi-dimensional arrays with functional semantics, `linalg` offers linear algebra operations (matrix multiply, convolution, transpose), and `tosa` (Tensor Operator Set Architecture) supplies neural network operations. These high-level dialects let you express what computation you want without specifying how to implement it.

The **middle level** contains dialects that add structure to computation. The `affine` dialect provides polyhedral loop optimization abstractions that capture loop bounds and memory access patterns. The `scf` (Structured Control Flow) dialect offers familiar programming constructs like for-loops, while-loops, and if-statements. The `vector` dialect makes SIMD vectorization explicit, letting you operate on vector data types directly.

At the **low level**, dialects get closer to machine semantics. The `memref` dialect represents memory references—essentially pointers with shape metadata—and provides operations for allocation, loading, and storing. The `arith` dialect supplies basic arithmetic operations like addition, multiplication, and comparison. The `cf` (Control Flow) dialect provides unstructured control flow through branches and basic blocks. Finally, the `llvm` dialect represents LLVM IR constructs within MLIR's infrastructure, serving as the bridge to LLVM's code generation.

MLIR also provides **utility dialects** used across abstraction levels. The `func` dialect handles function definitions and calls, appearing in virtually every MLIR program. The `math` dialect provides mathematical functions like exponential, logarithm, and square root.

This hierarchy enables **progressive lowering**: high-level operations transform into mid-level representations, then into low-level operations, finally reaching LLVM IR or machine code. Each level provides appropriate abstractions for its optimization opportunities.

Throughout this book, we'll work with several key dialects. The `func` dialect handles function definitions and calls (used in all chapters). The `linalg` dialect provides linear algebra operations like matrix multiply and convolutions (Chapters 1-11). The `memref` dialect manages memory with operations for allocation, loading, and storing (Chapters 1-8). The `tensor` dialect represents immutable arrays (Chapters 2 and 4). The `arith` dialect supplies arithmetic operations like addition and multiplication (used throughout). The `scf` dialect offers structured control flow with loops and conditionals (Chapters 5-7). Additional dialects like `math` (mathematical functions), `affine` (polyhedral optimization), and `vector` (SIMD operations) appear in later chapters. In Chapters 8-14, we'll even create our own custom dialects.

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

Beyond the dialect system, MLIR provides shared infrastructure that all dialects can leverage, eliminating the need to rebuild common compiler functionality from scratch. This infrastructure forms the foundation that makes MLIR's composability practical.

The **pass infrastructure** lets you write transformations once and apply them to any dialect that exposes the appropriate interfaces. Standard passes like canonicalization (simplification), dead code elimination, inlining, and loop optimizations work across dialects, saving you from reimplementing these fundamental compiler techniques for each new dialect you create or use.

The **type system** provides common infrastructure that all dialects share while allowing each dialect to define its own custom types. Built-in types include integers (`i32`, `i64`), floating-point numbers (`f32`, `f64`), and tensors (`tensor<2x3xf64>`). Dialects can also introduce custom types using the syntax `!mydialect.mytype`, benefiting from MLIR's type checking and printing infrastructure without reimplementing these features.

The **attribute system** lets you attach compile-time constants and metadata to operations. Whether storing the value `0.0` for a constant operation, specifying loop unrolling factors, or annotating operations with optimization hints, attributes provide a uniform mechanism for associating data with IR constructs.

The **region system** allows operations to contain nested blocks of code, enabling representation of structured constructs like function bodies, loop bodies, and conditional branches. This hierarchical structure makes MLIR more powerful than flat instruction lists, capturing program structure explicitly.

Finally, the **interface system** defines common behavior across dialects, enabling generic algorithms that work with any dialect implementing the required interface. For example, a `ShapeInferenceInterface` can work with any operation that implements shape inference methods, regardless of which dialect defines that operation. This polymorphic capability is crucial for building reusable compiler passes.

### Why This Matters

The composability of MLIR's infrastructure dramatically reduces the effort required to build compilers for new domains. Consider the traditional approach to building a machine learning framework: you'd need to build a custom IR for ML operations from scratch, implement custom optimization passes from basic principles, write custom code generation for each target hardware platform, and invest thousands of person-hours in duplicated work that other frameworks have already done.

With MLIR's composable approach, the landscape changes completely. You can use the existing `tensor` and `linalg` dialects for ML operations instead of defining your own from scratch. You leverage the built-in optimization infrastructure rather than reimplementing fundamental compiler transformations. For code generation, you lower to the `gpu` dialect for GPUs and the `llvm` dialect for CPUs, then reuse LLVM's mature code generation machinery that has been refined over decades. This lets you focus on your unique value-add—whether that's a novel operation scheduler, a new fusion strategy, or domain-specific optimizations—rather than rebuilding the entire compiler stack.

This is why major projects have adopted MLIR. TensorFlow uses it for compilation and optimization. PyTorch 2.0 leverages Torch-MLIR for its compilation backend. JAX uses MLIR (via StableHLO) for its compilation path. Serving frameworks like TensorRT-LLM build on MLIR infrastructure. The ecosystem consolidation means that innovations in one project—a new optimization pass, support for new hardware, or improved analysis techniques—can benefit the entire community rather than remaining siloed in a single framework's codebase.

Now that we understand MLIR's multi-level philosophy and dialect system, we need to answer a crucial question: how do high-level operations like `linalg.matmul` actually become executable machine code? This is where MLIR's transformation system comes in.

---

## 1.4 Lowering: The Heart of MLIR Compilation

MLIR's central mechanism is **progressive lowering**—transforming operations from high-level abstractions to low-level implementations through a series of **passes**.

### What is a Pass?

A pass is a transformation that walks the IR and rewrites it. Passes can **lower** operations by replacing high-level operations with lower-level equivalents. They can **optimize** code by eliminating redundancy, fusing operations to reduce memory traffic, or reordering computations for better performance. Passes can also **canonicalize** patterns by simplifying code to standard forms that make subsequent transformations easier.

### The Progressive Lowering Pipeline: MLIR's Abstraction Ladder

MLIR's power comes from maintaining operations at multiple abstraction levels simultaneously. Here's the complete picture of how high-level operations progressively transform into machine code:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ABSTRACTION LADDER                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HIGH LEVEL: "What to compute"                                 │
│  ┌───────────────────────────────────────┐                     │
│  │ Tensor/Linalg Dialect                 │                     │
│  │  • linalg.matmul                      │                     │
│  │  • tensor.extract, tensor.insert      │                     │
│  │  Declarative: describes computation    │                     │
│  └─────────────┬─────────────────────────┘                     │
│                │ Linalg-to-Loops                               │
│                ▼                                                │
│  MID LEVEL: "How to iterate"                                   │
│  ┌───────────────────────────────────────┐                     │
│  │ SCF/Arith Dialect                     │                     │
│  │  • scf.for (structured loops)         │                     │
│  │  • arith.addf, arith.mulf             │                     │
│  │  • memref.load, memref.store          │                     │
│  │  Imperative: explicit control flow    │                     │
│  └─────────────┬─────────────────────────┘                     │
│                │ SCF-to-CF                                      │
│                ▼                                                │
│  LOW LEVEL: "Basic blocks & branches"                          │
│  ┌───────────────────────────────────────┐                     │
│  │ CF/LLVM Dialect                       │                     │
│  │  • cf.br, cf.cond_br                  │                     │
│  │  • llvm.load, llvm.store              │                     │
│  │  Unstructured: just control flow      │                     │
│  └─────────────┬─────────────────────────┘                     │
│                │ LLVM Translation                               │
│                ▼                                                │
│  ┌───────────────────────────────────────┐                     │
│  │ Machine Code (x86, ARM, etc.)         │                     │
│  │  • mov, add, mul (registers)          │                     │
│  │  • Memory addressing, jumps           │                     │
│  └───────────────────────────────────────┘                     │
│                                                                 │
│  Each level enables appropriate optimizations:                 │
│  • High: operation fusion, algebraic simplification            │
│  • Mid: loop optimization, vectorization                       │
│  • Low: register allocation, instruction scheduling            │
└─────────────────────────────────────────────────────────────────┘
```

**Why Progressive?** Each abstraction level provides different optimization opportunities. High-level operations enable semantic optimizations ("these two matrix multiplies can fuse"). Mid-level loops enable cache optimizations ("tile this loop for L1 cache"). Low-level code enables hardware optimizations ("use SIMD instructions"). By lowering progressively, MLIR applies the right optimizations at the right level.

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

## 1.5 Compilation Strategy: JIT for This Book

Once we've lowered MLIR to LLVM dialect, we face a choice: compile ahead-of-time (AOT) to produce executables or libraries, or compile just-in-time (JIT) at runtime. Each approach has tradeoffs involving development speed, runtime performance, and deployment complexity.

This book uses **JIT compilation exclusively**. When you call a function for the first time, MLIR generates the IR, applies optimization passes, compiles to machine code, and executes—all at runtime. This provides immediate feedback during development: you can modify code, re-run your Python script, and see results in seconds rather than waiting for separate compilation and linking steps. For learning MLIR, this rapid iteration is invaluable.

JIT compilation does introduce overhead: each function must be compiled before its first execution, taking milliseconds to seconds depending on complexity. For production deployments, AOT compilation often makes more sense since you compile once during build time and execute many times without recompilation overhead. However, Chapter 3 will show how to cache compiled functions, making JIT viable even for production by compiling once and reusing the cached machine code across multiple executions.

The comprehensive comparison of compilation strategies—AOT vs JIT tradeoffs, use cases, implementation techniques, and caching mechanisms—appears in Chapter 3. For now, understanding that this book uses JIT for its development speed benefits is sufficient.

---

## 1.6 Understanding the MLIR Programming Model

Before we dive into implementing matrix multiply, we need to understand how MLIR programs are constructed. Coming from Python or C++, MLIR's programming model may seem unusual at first. This section introduces the core concepts that underpin all MLIR code you'll write throughout this book.

### Building a Computation Graph

When you write MLIR code programmatically (using the C++ API), you're not writing imperative code that executes immediately. Instead, you're **building a computation graph**—a data structure that represents your program. Think of it like constructing a recipe before cooking: you specify all the steps and ingredients first, then execute the recipe later.

In MLIR, you construct this graph by adding **operations** to it. An operation is a node in the graph that represents a computation or action. Operations produce **values** (the results of computations) and consume values (the inputs). These values flow through the graph along edges, defining the program's data flow.

Here's a crucial insight that surprises many newcomers: **even creating a constant is an operation**. When you need the number 0.0 in your program, you don't just write `0.0` and have it appear. Instead, you add a "constant operation" to the graph. This operation takes no inputs but produces one output: the constant value you specified.

Consider this MLIR text IR:

```mlir
%zero = arith.constant 0.0 : f32
%result = arith.addf %input, %zero : f32
```

This represents two operations in the graph:
1. A constant operation that produces the value `%zero` (which is 0.0)
2. An addition operation that consumes `%input` and `%zero`, producing `%result`

The `arith.constant` operation creates a new node in the computation graph. The value `0.0` itself is stored as an **attribute**—compile-time data attached to the operation. Think of it this way: the operation is the action ("create a constant"), while the attribute is the data ("the constant's value is 0.0").

This distinction between operations (actions in the graph) and attributes (data attached to operations) is fundamental to MLIR. Operations can execute at runtime and produce values that flow through your program. Attributes are compile-time constants that parameterize operations.

### Operations, Values, and Types

Every operation in MLIR has four key components. **Operands** are input values consumed by the operation, similar to function arguments. **Results** are output values produced by the operation—the computed results that flow to subsequent operations. **Attributes** store compile-time constant data like numbers, strings, and types that parameterize the operation. Finally, **regions** are optional nested blocks of code, such as function bodies or loop bodies, that allow operations to contain hierarchical structure.

When you build IR programmatically, you use an `OpBuilder` to create operations. The builder maintains an **insertion point**—a location in the graph where new operations will be added. As you create operations, they're inserted at this point and connected to other operations through their inputs and outputs.

Values in MLIR are strongly typed. Every value has a **type** that describes its data: `f32` for 32-bit floating point, `i64` for 64-bit integers, `memref<8x32xf32>` for a 2D array of floats, and so on. The type system ensures that you don't accidentally add an integer to a floating-point number or pass a matrix where a scalar is expected. Types are verified when you construct the IR, catching errors early.

### The Three-Phase Pattern

Nearly every MLIR project follows the same three-phase structure:

**Phase 1: IR Generation** - Build the computation graph using high-level operations. In our matrix multiply example, this means creating operations like `linalg.matmul` that declaratively specify "multiply these matrices." You're not specifying *how* to multiply them (nested loops? SIMD instructions? parallel threads?), just *what* to compute.

**Phase 2: Transformation and Lowering** - Apply optimization passes that transform the graph. These passes progressively lower high-level operations into lower-level implementations. A pass might convert `linalg.matmul` into nested loops, then another pass converts those loops into conditional branches, and finally another pass converts everything into LLVM IR. Each pass operates on the graph, rewriting operations and transforming the structure.

**Phase 3: Compilation and Execution** - Translate the final low-level IR to machine code and execute it. For JIT compilation (what we use in this book), this happens at runtime. The IR gets compiled to executable machine instructions stored in memory, and we get back a function pointer we can call from C++ or Python.

Understanding this pattern helps you reason about where code lives and when it executes. The IR generation code (Phase 1) runs in your program to *build* the computation graph. The optimization passes (Phase 2) run to *transform* the graph. The generated code (Phase 3) runs to *execute* the computation you described. These are three distinct execution phases with different responsibilities.

With this foundation, we're ready to implement our first complete example: a JIT-compiled matrix multiply. As you read the code, remember: we're building a graph of operations that will be optimized and compiled later, not writing code that executes immediately.

---

## 1.7 Your First MLIR Program: Matrix Multiply

Now that we understand the foundations, let's implement a complete example. We'll build an 8×32 × 32×16 matrix multiply using MLIR's `linalg` dialect, compile it with JIT, and execute it from Python.

### Problem Statement

We're implementing matrix multiplication: C = A @ B, where A is an 8×32 matrix, B is 32×16, and C is 8×16 (all float32). This is GEMM (General Matrix Multiply)—the fundamental operation in neural networks. Every linear layer, attention mechanism, and feed-forward network relies on matrix multiplication.

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

Let's break this down operation by operation. The function signature `func.func @gemm_8x16x32` uses the `func` dialect for function definitions. It takes three arguments: A, B, and C, all represented as memrefs (memory references—think "pointers with shape information"). The type `memref<8x32xf32>` describes a 2D array with shape 8×32 and element type f32 (32-bit float).

We use memrefs instead of tensors here to keep the example simple. Memrefs directly map to memory buffers and require no additional transformation. Tensors, while more optimization-friendly, need a bufferization pass to convert them to memrefs before execution. Chapter 2 will introduce tensors, and Chapter 4 will dive deep into bufferization. For this introductory example, memrefs suffice.

**The Output Buffer Pattern**: Notice that our function takes the output matrix C as an *input parameter* rather than returning a new matrix. This "output buffer pattern" is common in systems programming: the caller allocates memory and passes it in, and the function writes results directly to that buffer. This avoids hidden allocations and gives the caller control over memory management. Python programmers coming from `result = a @ b` semantics might find this unfamiliar, but it's standard in C++ and critical for performance. Chapter 4 will show how this pattern connects to MLIR's bufferization system.

The function body contains three operations. First, `arith.constant 0.0` creates a constant zero value (from the `arith` dialect, which we'll see more of in Chapter 5). Second, `linalg.fill` initializes the output matrix C to all zeros. Third, `linalg.matmul` performs the actual multiplication.

**[CRITICAL - MUST UNDERSTAND]**: The `linalg.fill` operation is NOT optional—it's *mandatory*. Here's why: `linalg.matmul` uses *accumulation semantics*, meaning it performs `C += A @ B`, not `C = A @ B`. If you omit the fill operation, matmul will add results to whatever garbage values happen to be in memory, producing incorrect results. This is similar to how BLAS routines like `cblas_sgemm` require alpha/beta parameters to control accumulation. Always initialize output buffers before accumulation operations. We'll see this pattern repeatedly with linalg operations throughout the book.

The key insight here is that `linalg.matmul` is declarative. It specifies *what* to compute (multiply matrices A and B, accumulate into C) without specifying *how*. There are no explicit loops, no SIMD instructions, no memory layout decisions. MLIR's optimization passes will make those decisions during the lowering phase, choosing the best implementation for the target hardware.

---

## 1.8 Generating IR with the C++ API

MLIR provides a C++ API for programmatic IR construction. Let's walk through the implementation in `ch.1.Fixed-size/src/ir.cpp`, understanding how we build the computation graph piece by piece.

### Loading Dialects

```cpp
OwningOpRef<ModuleOp> createGemmModule(MLIRContext& context) {
  // Load required dialects
  context.getOrLoadDialect<func::FuncDialect>();    // func.func, func.return
  context.getOrLoadDialect<linalg::LinalgDialect>(); // linalg.matmul
  context.getOrLoadDialect<memref::MemRefDialect>(); // memref types
  context.getOrLoadDialect<arith::ArithDialect>();   // arith.constant
```

Before using any dialect's operations, we must load that dialect into the context. The context stores all MLIR state—types, attributes, and loaded dialects. Loading a dialect registers its operations, making them available for use. Each dialect provides a vocabulary of operations: `func` for functions, `linalg` for linear algebra, `memref` for memory operations, and `arith` for arithmetic.

### Creating the Module and Builder

```cpp
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();  // Source location (not tracked here)
  auto module = ModuleOp::create(loc); // Top-level container
  
  // CRITICAL: Set insertion point
  builder.setInsertionPointToStart(module.getBody());
```

The `OpBuilder` is our tool for constructing operations. Every operation we create goes through this builder. The builder maintains an insertion point—the location where new operations will be added. Setting the insertion point to the module's body means operations we create will be added as top-level entities in the module (like function definitions).

The `ModuleOp` is MLIR's top-level container. A module holds functions, global variables, and other top-level declarations. Every MLIR program is organized as a module containing one or more functions. The `loc` parameter represents source location information for debugging, but since we're generating IR programmatically rather than parsing source files, we use `getUnknownLoc()`.

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

Types in MLIR are first-class objects that describe the structure of data. We create memref types by specifying their shape (dimensions) and element type. The `MemRefType::get({8, 32}, f32Type)` creates a type representing an 8×32 matrix of 32-bit floats. The fixed sizes simplify this first example—Chapter 2 will show how to handle dynamic shapes where dimensions aren't known until runtime.

The function type is created using `getFunctionType`, which takes two lists: input types and output types. Our function takes three memref inputs and returns nothing (the output is written in-place to the third argument). This pattern of passing output buffers as arguments is common in systems programming and will be explored more deeply in Chapter 4 when we discuss calling conventions.

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

Now we create the function operation itself. The `func::FuncOp` represents a function definition—it's the MLIR equivalent of a C++ function. We provide a name ("gemm_8x16x32") and the function type we just defined. Calling `setPublic()` marks the function as externally visible, which we need since we'll call it from outside the MLIR module (from our Python code).

Every function needs a body—a sequence of operations to execute. In MLIR, function bodies are organized as basic blocks. A basic block is a straight-line sequence of operations with one entry point and one exit point (no branches in the middle). We call `addEntryBlock()` to create the first block and set our builder's insertion point there, so subsequent operations we create will be added to this block.

The function's arguments (the three memrefs) are available through the entry block. Each argument is a value that we can use as an input to operations. We retrieve them with `getArgument(index)` and store them in variables for convenient reference.

### Generating the Matrix Multiply Operations

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

Now we add the operations that actually perform the computation. Remember from section 1.6 that even constants are operations in MLIR. To get the value 0.0, we create a constant operation. The `builder.getFloatAttr(f32Type, 0.0)` creates an attribute (compile-time data) holding the value 0.0, and `builder.create<arith::ConstantOp>` creates the operation that produces this constant as a runtime value. We call `zero.getResult()` to get the value produced by this operation, which we can then use as an input to other operations.

The `linalg::FillOp` initializes the output matrix C to all zeros. Matrix multiplication is an accumulation operation (C += A @ B), so we need to start with zero. The fill operation takes two arguments: the value to fill with (our zero) and the memref to fill (matrixC).

Finally, we create the `linalg::MatmulOp`. This is the high-level operation that represents our matrix multiplication. Notice we don't specify any implementation details—no loops, no memory access patterns, no vectorization. The operation carries high-level semantic meaning ("multiply these two matrices"), and later compilation passes will decide how to implement it efficiently. This declarative approach is central to MLIR's design philosophy.

We end the function with a `func::ReturnOp`. Since our function has void return type, the return takes no arguments. With all operations created, we return the complete module.

Notice the abstraction level throughout this code: we're building a computation graph using high-level operations. We're not writing loops or specifying how the GPU schedules work. That's the power of MLIR—start at a high level, let the compiler handle the low-level details.

---

## 1.9 Debugging Toolbox: When Things Go Wrong

Before diving into the compilation pipeline, let's cover essential debugging tools. MLIR code will fail—type mismatches, dialect loading errors, pass failures. Knowing how to inspect and debug IR is critical for productive development.

### Saving IR to Files

The most important debugging technique is **inspecting the IR** at each compilation stage. In your C++ code (like `ir.cpp` from Chapter 1), add IR dumps:

```cpp
// After building your module, before compilation
module.dump();  // Prints IR to stderr - see it in console

// Or save to file for detailed inspection
std::string irString;
llvm::raw_string_ostream os(irString);
module.print(os);
std::ofstream file("debug_output.mlir");
file << os.str();
file.close();
std::cout << "IR saved to debug_output.mlir\n";
```

This creates a `.mlir` file you can examine in a text editor. You'll see exactly what operations were generated, their types, and the structure of the IR.

**Quick Debug**: The simplest approach is `module.dump()` right after IR construction. This prints to your console immediately—no file handling needed.

### Using mlir-opt: The Swiss Army Knife

`mlir-opt` is MLIR's command-line tool for applying passes and inspecting transformations. It's invaluable for debugging:

```bash
# View your IR (checks syntax)
mlir-opt debug_output.mlir

# Apply specific passes to see transformations
mlir-opt debug_output.mlir --convert-linalg-to-loops

# Chain multiple passes
mlir-opt debug_output.mlir \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --canonicalize

# Get help on available passes
mlir-opt --help
```

**Why use mlir-opt?** When your JIT compilation fails mysteriously, `mlir-opt` lets you apply passes one at a time and see which one breaks. If pass 3 fails but passes 1-2 succeed, you know exactly where to investigate.

### Common Errors and Solutions

**Error: "Dialect not loaded"**
```
Error: operation 'linalg.matmul' is unknown
```
**Solution**: Load the dialect before using its operations:
```cpp
context.loadDialect<linalg::LinalgDialect>();
```

**Error: "Type mismatch"**
```
Error: operand type 'memref<8x32xf32>' doesn't match expected 'memref<8x16xf32>'
```
**Solution**: Check your dimensions. Matrix multiply requires compatible shapes (M×K @ K×N → M×N).

**Error: "Verification failed"**
```
Error: 'linalg.matmul' op requires two input operands
```
**Solution**: Operations have requirements (argument count, types, attributes). Check the dialect documentation.

### Debugging Workflow

When something breaks:

1. **Save IR immediately** before the failing pass: `module.dump()` or save to file
2. **Run mlir-opt** on saved IR with the failing pass only
3. **Read error message** carefully—MLIR errors pinpoint exact operation and reason
4. **Check operation definition** in dialect documentation
5. **Verify types** of all operands match expectations
6. **Test incrementally**—add operations one at a time, verify after each

**Pro tip**: MLIR's error messages are excellent. They tell you exactly what's wrong ("expected type X, got type Y") and which operation failed. Read them carefully before diving into code.

With these debugging tools, you can diagnose and fix issues quickly. Now let's see the full compilation pipeline that transforms our high-level IR into machine code.

---

## 1.10 The Compilation Pipeline: Progressive Lowering

With the IR generated, we need to transform it from high-level operations into executable machine code. This happens through a series of transformation passes, each lowering the IR one step closer to the machine. The implementation resides in `lowering.cpp`.

### The Pass Manager and Pass Sequence

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

The `PassManager` orchestrates the transformation pipeline. Each pass is a transformation that walks the IR and rewrites it according to specific rules. We add passes to the manager in the order they should run, and then `pm.run(module)` executes them sequentially. Each pass transforms the IR in place, modifying the module. Later passes see the results of earlier passes, creating a progressive lowering pipeline.

### Understanding Progressive Lowering: Step by Step

Let's trace how our matrix multiply transforms through each pass. Understanding this progression is key to working effectively with MLIR.

**Pass 1: Canonicalization**

The canonicalizer applies algebraic simplifications and normalizes patterns. For our matrix multiply, it might simplify constant expressions, eliminate dead code, or normalize operation orders. For example, if we had `x + 0`, it would simplify to `x`. This is a cleanup pass that makes subsequent transformations easier by ensuring the IR is in a standard form. Canonicalization is cheap and effective, so it often runs multiple times throughout compilation.

**Pass 2: Linalg to Loops**

This is where high-level semantics meet implementation. The `convertLinalgToLoops` pass replaces our declarative `linalg.matmul` operation with explicit nested loops using the SCF (Structured Control Flow) dialect. Here's conceptually what happens:

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

Now we have explicit loops iterating over matrix dimensions. The outer loops (i, j) iterate over output positions in C, while the inner loop (k) accumulates the dot product. Inside the innermost loop, we see the fundamental operations: load elements from A and B, multiply them, add to the accumulator, and store back to C. The SCF dialect provides structured loops with well-defined semantics—we'll explore it thoroughly in Chapter 5.

This transformation exposes the computation's structure, making it accessible to subsequent optimizations. Notice how much detail is now explicit: memory access patterns, loop bounds, arithmetic operations. Later passes can apply optimizations like loop tiling (breaking large loops into cache-friendly blocks), vectorization (processing multiple elements per iteration with SIMD), and parallelization (running iterations concurrently).

**Pass 3: SCF to Control Flow**

The `SCF-to-CF` pass converts structured loops into unstructured control flow—basic blocks and branches. It transforms each `scf.for` operation into a header block (checks loop condition), a body block (executes loop iteration), and branch operations connecting them. This lowers from structured to unstructured control flow.

Why this transformation? Structured control flow makes optimization easier because the compiler can reason about loop structure, invariants, and dependencies. But CPUs ultimately execute conditional and unconditional branches, not high-level loop constructs. This pass bridges that gap, converting loops into the basic blocks and branches that the CF (Control Flow) dialect represents. We'll explore these control flow primitives in Chapter 5.

**Pass 4: MemRef to LLVM**

This pass handles a deceptively complex task: converting memref types to LLVM pointer types and transforming memory operations (`memref.load`, `memref.store`) into LLVM load and store instructions. For our fixed-size memrefs, the transformation is straightforward: `memref<8x32xf32>` becomes a pointer to float with some address arithmetic for indexing.

But there's hidden complexity that Chapter 2 will reveal. Memrefs can have dynamic dimensions (unknown until runtime), non-contiguous layouts (strided or transposed), and complex addressing schemes. The lowering pass handles all this by generating code that computes addresses from base pointers, offsets, and strides. For now, our fixed-size example avoids this complexity, but keep in mind that memref lowering can generate significant code for complex cases.

**Passes 5-7: Final Dialect Lowering**

The final three passes convert remaining high-level dialects to LLVM dialect. The `ConvertFuncToLLVM` pass converts function definitions and calls. The `ConvertArithToLLVM` pass converts arithmetic operations like `arith.mulf` to `llvm.fmul`. The `ConvertControlFlowToLLVM` pass converts branches and control flow. After these passes complete, every operation in our IR is from the LLVM dialect.

The result is LLVM dialect IR—MLIR operations that directly correspond to LLVM IR constructs. We haven't left MLIR yet; we're still working with MLIR's representation. But the operations are now LLVM operations represented in MLIR's infrastructure. This enables one final translation step to actual LLVM IR for compilation.

---

## 1.10 JIT Compilation and Execution

With our IR fully lowered to LLVM dialect, we can now compile it to executable machine code and run it. MLIR provides the `ExecutionEngine` class that wraps LLVM's JIT compiler, making this process straightforward. The implementation resides in `jit.cpp`.

### Creating the Execution Engine

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

The `ExecutionEngine::create` method performs several steps internally. First, it translates the MLIR LLVM dialect to actual LLVM IR (a format LLVM's optimizer and code generator understand). Second, it applies LLVM optimizations through the transformer we configured—the `makeOptimizingTransformer(3, 0, nullptr)` creates an optimization pipeline equivalent to `-O3` compiler flags. This includes instruction selection, register allocation, instruction scheduling, and architecture-specific optimizations like vectorization.

Third, the execution engine JIT-compiles the optimized LLVM IR to native machine code for the host architecture (x86_64, ARM, etc.). This generated code is stored in memory and is immediately executable. Finally, we use `lookup` to find our function by name, getting back a pointer to the compiled code.

The transformer is crucial for performance. Without optimization (transformer level 0), we'd get naive code with redundant loads, no vectorization, and poor register usage. With `-O3` (level 3), LLVM applies aggressive optimizations that can match hand-written assembly for many computations. Chapter 3 will explore the execution engine in more depth, including caching strategies to avoid recompiling the same code repeatedly.

### Invoking the Compiled Function from C++

```cpp
typedef void (*GemmFunc)(float*, float*, float*);

void callGemmJIT(void* funcPtr, float* A, float* B, float* C) {
  auto gemm = reinterpret_cast<GemmFunc>(funcPtr);
  gemm(A, B, C);
}
```

Once we have the function pointer from JIT compilation, calling it is straightforward. We cast the void pointer to a function pointer type matching our signature: three float pointers (one for each matrix) and no return value. The memrefs in our MLIR code compile down to simple pointers for fixed-size arrays—the dimensions are baked into the generated code as loop bounds.

In memory, the arrays are just contiguous sequences of floats in row-major order. When we pass NumPy arrays from Python, NumPy ensures the data is contiguous, so we can safely pass the underlying memory pointers to compiled code. Chapter 2 will reveal additional complexity when we introduce dynamic shapes: memrefs become descriptor structs containing size and stride information, requiring more complex calling conventions.

### Python Integration with Pybind11

The final piece is exposing our JIT compiler to Python so we can easily test and use it. Pybind11 provides seamless C++/Python integration. The implementation in `bindings.cpp` wraps our JIT compiler in a Python-friendly interface.

**Understanding the Python ↔ C++ Boundary**: This is where Python's high-level NumPy arrays cross into C++'s low-level memory world. Python code calls `gemm(A, B)` with NumPy arrays. Pybind11 extracts raw memory pointers from these arrays. We pass these pointers to JIT-compiled machine code (which expects raw float pointers). The compiled code writes results directly into memory. Python sees the modified NumPy array. Understanding this boundary is crucial for performance—copying data across this boundary is expensive, so we avoid it by passing pointers.

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

### Testing and Validation

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

This chapter introduced MLIR's foundational concepts and implemented a complete JIT-compiled matrix multiply. We explored how the multi-level problem emerged from heterogeneous computing, why LLVM IR alone proved insufficient, and how MLIR's dialect system provides extensible operation vocabularies at different abstraction levels. Through progressive lowering and composable passes, MLIR transforms high-level operations into executable machine code while maintaining optimization opportunities at each stage.

We began by understanding the traditional compiler pipeline and why one intermediate representation isn't sufficient for modern heterogeneous computing. The ML compiler landscape before MLIR was fragmented, with each framework (TensorFlow, PyTorch, XLA) building its own complete compilation stack. MLIR unified this ecosystem by providing extensible dialects—vocabularies of operations at different abstraction levels that can coexist and transform between each other.

The dialect system is central to MLIR's design. Each dialect provides operations appropriate for its abstraction level: high-level operations like `linalg.matmul` declare what to compute, mid-level operations like `scf.for` specify structured loops, and low-level operations like `llvm.load` directly map to machine instructions. Progressive lowering through passes transforms operations from one level to the next, maintaining optimization opportunities at each stage.

We learned a crucial programming model concept: in MLIR, you're building a computation graph, not writing imperative code. Operations are nodes in this graph, and even creating a constant requires an operation. Attributes store compile-time data (like the value 0.0), while operations execute at runtime and produce values. Understanding this distinction—operations versus attributes, graph construction versus execution—is fundamental to working with MLIR effectively.

Our matrix multiply implementation demonstrated the three-phase pattern that underlies nearly every MLIR project. Phase 1 generates IR using high-level operations (`linalg.matmul`). Phase 2 applies transformation passes that progressively lower the IR (Linalg → SCF → CF → LLVM). Phase 3 compiles to machine code and executes. Each phase has distinct responsibilities: building the graph, transforming it, and executing the result.

We chose specific design simplifications for this introductory example. Using memrefs instead of tensors avoided the complexity of bufferization (covered in Chapter 4). Fixed dimensions (8×32×16) kept the interface simple—dynamic shapes introduce memref descriptors that complicate calling conventions, as we'll see in Chapter 2. JIT compilation provided immediate feedback, though Chapter 3 will address its compilation overhead through caching.

The testing strategy validated correctness by comparing against NumPy's battle-tested implementations. Small floating-point differences (around 10⁻⁷) are expected due to different operation ordering, but the results match within acceptable tolerance. This testing pattern—comparing against reference implementations—will continue throughout the book.

Chapter 2 tackles the next challenge: arbitrary matrix sizes with dynamic dimensions. The `?` notation in types like `memref<?x?xf32>` represents dimensions unknown until runtime. This flexibility comes at a cost: memrefs become complex descriptor structures with multiple fields, and the calling convention between Python and compiled code becomes significantly more involved. But this complexity is necessary for production systems where input sizes vary.