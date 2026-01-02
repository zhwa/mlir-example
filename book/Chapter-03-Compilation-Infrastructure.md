# Chapter 3: Compilation Infrastructure: AOT, JIT, and the Pass Pipeline

In Chapters 1 and 2, we generated MLIR IR for matrix multiplication operations and compiled it to executable code using MLIR's `ExecutionEngine`. We used passes like `createConvertLinalgToLoopsPass()` and `createSCFToControlFlowPass()` to progressively lower our IR, treating compilation as a black box—feed it IR, get back a callable function. This simplicity was intentional for learning, but now we need to understand what's happening inside that black box.

This chapter addresses fundamental questions about compilation infrastructure: **What is the difference between ahead-of-time (AOT) and just-in-time (JIT) compilation? Why does MLIR provide an ExecutionEngine, and when should you use it? How does MLIR's pass infrastructure orchestrate transformations? And critically for production systems, how do you compile MLIR to standalone executables or libraries?**

Understanding these concepts is essential because the compilation strategy you choose has profound implications for deployment, performance, and flexibility. Modern ML systems like PyTorch 2.0, TensorRT, and XLA all make specific choices about AOT vs JIT, and understanding these trade-offs will help you architect better systems.

## 3.1 AOT vs JIT: Two Compilation Philosophies

Compilers have two fundamentally different approaches to generating executable code: compile everything before running anything (ahead-of-time), or compile code on-demand as the program runs (just-in-time). Each philosophy has deep implications for performance, flexibility, and deployment.

### Ahead-of-Time (AOT) Compilation

**AOT compilation** is the traditional compilation model exemplified by C, C++, Rust, and Go compilers. The workflow is strictly separated into two phases.

During build time, which happens once, the compiler reads source files, parses and validates them, optimizes the intermediate representation, generates machine code, links everything into executables or libraries, and writes binary files to disk. At runtime, which happens many times, the system simply loads the precompiled binary from disk and executes immediately with no compilation overhead.

The key characteristic: **all compilation happens before the program runs**. When a user launches your application, they're executing pre-generated machine code. No compiler runs during execution.

AOT compilation offers several compelling advantages. The program starts instantly and runs at full speed from the first instruction with zero compilation overhead at runtime. Every execution has identical performance characteristics, providing predictable performance that's crucial for production systems. The compiler can tune code for specific CPU features like AVX-512 or ARM NEON, optimizing for the target hardware. Deployment is simplified because you distribute simple binaries without requiring the compiler or source code. This minimal startup time is critical for services that must respond immediately, such as web servers and embedded systems.

However, AOT compilation has important limitations. It cannot adjust to runtime information like input shapes, sparsity patterns, or hardware differences, lacking runtime adaptability. Supporting multiple platforms requires separate binaries for architectures like x86_64 and ARM64, increasing binary size and distribution complexity. Any code change requires rebuilding and redistributing binaries, which can slow development iteration.

### Just-in-Time (JIT) Compilation

**JIT compilation** delays code generation until runtime. The workflow interleaves compilation with execution in a single unified phase. The system starts with a high-level representation—bytecode, IR, or source code. As code is needed, it compiles to machine code on-demand, caches the compiled code for reuse, executes it, and repeats this process for new code paths. Modern JIT systems include Java's HotSpot VM, JavaScript engines (V8, SpiderMonkey), PyPy, LuaJIT, and increasingly, ML frameworks like PyTorch 2.0 with torch.compile and JAX.

JIT compilation offers powerful capabilities for dynamic systems. **Runtime adaptability** allows the compiler to specialize code based on actual data—input shapes in neural networks, branch frequencies discovered during execution, and type information observed at runtime. **Dynamic optimization** enables profiling code during execution and recompiling hot paths with aggressive optimizations informed by real usage patterns. JIT systems can ship a **single deployment artifact**: high-level IR that compiles to native code on whatever platform it runs on, eliminating the need for platform-specific binaries. **Incremental compilation** means only code that actually executes pays compilation cost—unused code paths never get compiled, saving time and memory.

However, JIT compilation imposes significant costs. **Compilation overhead** means the first execution of code pays the compilation cost, which can be 10-100ms for small functions or seconds for large modules—unacceptable for latency-sensitive applications. **Memory overhead** requires keeping the entire compiler infrastructure in memory throughout execution, consuming 50-200 MB or more. **Unpredictable performance** means the first run is slow while warming up, subsequent runs are fast—this variability complicates performance analysis and capacity planning. Finally, **complex implementation** makes JIT compilers significantly more complex than AOT compilers, requiring runtime compilation infrastructure, code cache management, and profile-guided optimization systems.

### The ML Systems Landscape

Machine learning compilation sits in an interesting position. Consider how different systems choose.

AOT-dominant systems prioritize predictable latency for production serving. TensorFlow/XLA compiles computation graphs to optimized binaries offline. TensorRT takes a trained model, compiles it to GPU kernels, and saves the result to a file for deployment. ONNX Runtime loads serialized models, applies graph optimizations, and compiles kernels ahead of serving. TVM compiles models for edge devices like mobile phones and IoT hardware where no compiler can run at deployment time. Why AOT? When a user queries a deployed model, the first inference must be as fast as the thousandth inference—compilation overhead is unacceptable. The solution: compile once during model deployment, serve forever.

JIT-friendly systems prioritize flexibility for research and experimentation. PyTorch eager mode interprets operations dynamically without compilation. PyTorch 2.0's torch.compile JIT-compiles computation graphs from eager execution traces. JAX JIT-compiles function transformations like jit, grad, and vmap at first invocation. Triton JIT-compiles GPU kernels from Python-like syntax. Why JIT? Researchers change model architectures constantly—recompiling and redeploying for every tweak kills productivity. JIT compilation enables a fast workflow: write code, run immediately, iterate quickly. The compilation overhead (a few seconds on first run) is acceptable compared to hour-long model training times.

Many production systems use a hybrid approach that combines both strategies. During development, they use JIT for fast iteration; during deployment, they switch to AOT for predictable performance. PyTorch exemplifies this pattern: researchers use eager mode (no compilation) or torch.compile (JIT) during development, then export models to TorchScript or ONNX for AOT compilation before deployment. This gives the best of both worlds—flexibility during research, performance in production.

## 3.2 MLIR's ExecutionEngine: A Prototyping Tool, Not a Production Solution

Throughout Chapters 1 and 2, we used MLIR's `ExecutionEngine` to compile and execute our matrix multiplication. It worked beautifully—pass in MLIR IR, get back a callable function pointer. But it's crucial to understand: **ExecutionEngine is a convenience tool for prototyping, not how production MLIR systems compile code**.

### What ExecutionEngine Provides

`ExecutionEngine` is a thin wrapper around LLVM's LLJIT (Lazy LLVM JIT) that makes JIT compilation accessible with a simple API. It handles the entire pipeline: translating MLIR to LLVM IR, optimizing the code, compiling to machine code in memory, and providing function pointers you can call directly. This is enormously convenient for learning and testing—you get a runnable function in milliseconds with just a few lines of code. Section 3.5 explains these internal steps and the API in detail.

However, convenience comes with trade-offs.

### When ExecutionEngine is Appropriate

ExecutionEngine shines in specific scenarios:

**Prototyping and testing**: When developing MLIR transformations, you want immediate feedback. Write a pass, compile with ExecutionEngine, verify correctness. The edit-compile-test cycle is seconds instead of minutes.

**Dynamic workloads**: Applications where code generation patterns change frequently. For example, a REPL (read-eval-print loop) for a domain-specific language, where each user input generates new code.

**Research experiments**: When comparing optimization strategies, measuring performance of different lowering paths, or validating correctness of transformations. The JIT overhead is negligible compared to model training times.

**Small-scale serving**: For internal tools or services with low query volume where a few milliseconds of warmup latency is acceptable.

### When ExecutionEngine is NOT Appropriate

ExecutionEngine is inappropriate for production systems for several reasons:

**Unpredictable latency**: The first invocation of a function pays compilation cost. For a matrix multiplication, this might be 10-100ms. That's unacceptable for serving systems where every millisecond matters. Users expect sub-millisecond response times for simple queries.

**Memory overhead**: ExecutionEngine keeps the entire compiler infrastructure in memory—LLVM's optimizer, code generator, and all supporting data structures. For a simple matrix multiplication, this can be 50-200 MB. Multiply by hundreds of worker processes, and you've wasted gigabytes of RAM on compilation machinery that sits idle after warmup.

**Security concerns**: Allowing code generation at runtime is a security risk. If an attacker can influence what code gets JIT-compiled (even indirectly through inputs), they might be able to generate and execute malicious code. AOT compilation eliminates this attack surface—the deployed binary contains only pre-approved code.

**Platform limitations**: Some deployment targets (iOS apps, certain embedded systems, secure enclaves) **prohibit JIT compilation entirely** for security reasons. ExecutionEngine simply cannot work on these platforms.

**Reproducibility**: JIT-compiled code might vary between runs (different CPU features detected, different optimization heuristics applied). For systems that require bit-exact reproducibility (regulatory compliance, debugging), this variability is problematic.

### What Production MLIR Systems Actually Do

Real ML serving systems using MLIR follow an AOT workflow. **TensorFlow's MLIR pipeline** demonstrates this approach: it generates MLIR IR from the TensorFlow computation graph, applies optimization passes to improve performance, lowers the optimized IR to LLVM IR, compiles to object files (.o), links those object files into a shared library (.so), deploys the library to production servers, and finally the servers simply load the library and call functions with no compilation overhead.

Triton compiler, despite being "JIT" in spirit, follows a similar AOT pattern: it parses Triton Python code to IR, lowers to LLVM IR, generates PTX (NVIDIA) or AMDGPU code, passes the code to the GPU driver for final compilation, caches the compiled kernels on disk, and reuses those cached kernels across runs. Even systems with "JIT" in their name ultimately do AOT compilation—they just do it lazily on first use, then cache the results.

### The Right Tool for the Job

Think of ExecutionEngine as **scaffolding**—temporary infrastructure that supports development but isn't part of the final product. You wouldn't ship a building with scaffolding still attached, and you shouldn't ship a production service with ExecutionEngine embedded.

**Connection to Chapter 1**: Remember when we first used ExecutionEngine in Chapter 1 to compile our matrix multiplication? We treated it as a simple "compile and run" black box. Now you understand what's really happening: ExecutionEngine wraps LLVM's JIT, translates MLIR dialects to LLVM IR, optimizes the code, compiles to machine code in memory, and gives us a function pointer. That "magic" was LLVM's battle-tested JIT infrastructure working behind the scenes.

For this book, we use ExecutionEngine for three pragmatic reasons. It simplifies learning by letting us focus on IR rather than build systems. It enables rapid experimentation—we can change code and test immediately without rebuild cycles. It makes Python integration trivial, avoiding the complexity of linking and loading compiled libraries.

## 3.3 The Pass Infrastructure: Orchestrating Transformations

At the heart of MLIR's compilation model is the **pass infrastructure**—a framework for organizing, sequencing, and applying transformations to IR. Every optimization, lowering, and analysis we've performed runs as a "pass" within this infrastructure. Understanding how passes work is essential for both using MLIR effectively and extending it with custom transformations.

### What is a Pass?

A **pass** is a self-contained transformation or analysis that operates on MLIR IR. Think of a pass as a single, focused task in the compilation pipeline.

Transformation passes modify the IR in specific ways. Canonicalization simplifies IR to canonical forms, such as reducing `x + 0` to just `x`. Inlining replaces function calls with function bodies. Loop unrolling replicates loop bodies to eliminate loop overhead. Constant folding evaluates constant expressions at compile time. Dead code elimination removes operations that don't affect the result.

Analysis passes gather information without modification. Liveness analysis determines which values are live at each program point. Alias analysis determines whether two memrefs can point to the same memory. Call graph construction identifies which functions call which other functions.

Lowering passes convert between abstraction levels. The Linalg-to-Loops pass transforms high-level operations like `linalg.matmul` into nested `scf.for` loops. The SCF-to-CF pass converts structured control flow to basic blocks and branches. The MemRef-to-LLVM pass lowers memref operations to LLVM pointer operations.

Each pass has a narrow responsibility. This modularity is powerful: you can understand, test, and debug each pass independently. Composing passes creates complex transformations from simple building blocks.

### The PassManager: The Conductor

The **PassManager** orchestrates pass execution. It's responsible for scheduling passes in the correct order, managing pass dependencies (some passes require others to run first), providing context (access to MLIRContext, analyses, etc.), handling errors (propagating failures when a pass fails), and enabling parallelism (running passes on different operations concurrently when safe).

Creating a pass pipeline looks like this:

```cpp
PassManager pm(context);

// Add passes in sequence
pm.addPass(createCanonicalizerPass());           // Simplify IR
pm.addPass(createConvertLinalgToLoopsPass());    // Linalg → Loops
pm.addPass(createSCFToControlFlowPass());        // SCF → CF
pm.addPass(createFinalizeMemRefToLLVMConversionPass());    // MemRef → LLVM

// Run all passes on the module
if (failed(pm.run(module))) {
  // Handle compilation failure
}
```

The PassManager ensures passes execute in order, manages memory efficiently, and can even run passes in parallel when safe (multiple passes on different functions simultaneously).

### Pass Granularity: Operating on Specific Operations

Passes don't just operate on entire modules—they operate on **specific operation types**. This granularity is crucial for modularity and performance.

A ModulePass operates on an entire `ModuleOp`, suitable when transformations need a whole-program view, such as interprocedural optimizations or dead function elimination. Module passes cannot be parallelized across modules since each module is independent.

A FunctionPass operates on individual functions, making it appropriate for most function-level optimizations like loop transformations and local optimizations. Function passes can be parallelized: the PassManager runs the same pass on multiple functions simultaneously, utilizing available CPU cores.

An OperationPass can be either op-agnostic or op-specific. Op-agnostic passes work on any operation type, making them suitable for generic transformations like canonicalization and common subexpression elimination. They're the most flexible but require careful implementation since they can't assume specific operation structure. Op-specific passes target particular operation types, such as a pass that optimizes `linalg.matmul` specifically. These are type-safe—the compiler enforces that you're operating on the right operation type.

This granularity matters for performance. In Chapter 1, when we ran passes on a single function, the PassManager could optimize aggressively. In a real application with hundreds of functions, the PassManager runs function passes in parallel, utilizing all CPU cores.

### Pass Dependencies and Ordering

Passes often have dependencies—one pass requires another to run first. Consider lowering `linalg.matmul`:

```
linalg.matmul  (high level)
      ↓ requires ConvertLinalgToLoopsPass
scf.for loops  (structured control flow)
      ↓ requires ConvertSCFToCFPass
cf.br branches (unstructured control flow)
      ↓ requires ConvertControlFlowToLLVMPass
llvm.br        (LLVM IR)
```

If you run passes out of order, compilation fails. The SCF-to-CF pass can't convert operations that don't exist yet. The PassManager doesn't automatically infer dependencies—**you must specify the correct order**.

Some passes have subtle dependencies. Canonicalization, for example, should run **after** major transformations to clean up the IR they produce. Lowering passes create verbose IR with redundant operations; canonicalization simplifies it. The typical pattern runs a major transformation (like loop tiling), then canonicalization to clean up the resulting mess, followed by the next transformation. This interleaving of transformations and cleanup passes ensures the IR remains in a simplified, canonical form throughout the compilation pipeline.

### Pass Verification and Debugging

MLIR's pass infrastructure includes **verification** after each pass. By default, after every pass executes, MLIR verifies the entire IR is well-formed. This verification checks that all operations have valid types, SSA dominance properties hold (definitions dominate uses), and dialect-specific invariants are maintained.

If verification fails, MLIR reports which pass broke the IR. This is invaluable for debugging—without verification, a pass could corrupt the IR, and you'd only discover the problem later when code generation mysteriously fails.

You can control verification:

```cpp
PassManager pm(context);
pm.enableVerifier(true);  // Default: verify after each pass
```

For debugging, you can print IR after each pass:

```cpp
pm.enableIRPrinting(
  /*shouldPrintBeforePass=*/false,
  /*shouldPrintAfterPass=*/true,
  /*shouldPrintAfterOnlyOnChange=*/true
);
```

This shows exactly how each pass transforms the IR—essential when developing custom passes.

### Custom Passes: Extending the Infrastructure

You can write your own passes. A minimal pass looks like:

```cpp
struct MyCustomPass : public PassWrapper<MyCustomPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();  // Current function
    
    // Walk through all operations in the function
    func.walk([&](Operation *op) {
      // Analyze or transform each operation
    });
  }
};
```

This pass runs on every function in the module. You can access and modify operations, create new operations, or gather analysis information. The PassManager handles all the infrastructure—scheduling, parallelism, verification.

Writing passes is how you extend MLIR with domain-specific optimizations. Want to fuse attention operations in transformers? Write a pass. Want to optimize memory layouts for a specific accelerator? Write a pass. The infrastructure provides the scaffolding; you provide the transformation logic.

## 3.4 MLIR's Compilation Workflow: Progressive Lowering

Now that we understand passes, let's examine MLIR's compilation model: **progressive lowering**. This is the philosophy that distinguishes MLIR from traditional compilers. Instead of one giant transformation from source to machine code, MLIR applies many small transformations, each lowering the IR by one level of abstraction.

### The Abstraction Ladder

MLIR IR exists at multiple levels simultaneously:

**Level 5 - Domain-specific (highest abstraction)** contains custom dialects for specific domains such as Transformer operations and database queries. Operations like `transformer.attention` and `graph.conv2d` exist at this level. The semantics focus purely on "what computation to perform," providing mathematical specifications without implementation details.

At Level 4, structured compute is represented through the Linalg dialect, with operations like `linalg.matmul`, `linalg.conv`, and `linalg.reduce` that express structured linear algebra using iteration spaces, index maps, and declarative operations.

Level 3 introduces loops and structured control flow via the SCF dialect, with operations like `scf.for`, `scf.while`, and `scf.if` that provide control flow with explicit structure including loop bounds and loop bodies. The semantics represent imperative computation with structured control.

At Level 2, we find basic blocks and branches in the CF dialect, with unstructured control flow operations like `cf.br` and `cf.cond_br` that resemble assembly-level gotos.

Level 1 represents LLVM IR through the LLVM dialect, with operations like `llvm.load`, `llvm.store`, and `llvm.fadd` that are close to machine instructions, operating on a register-based virtual machine model.

Finally, Level 0 is machine code itself—the lowest abstraction—comprising actual CPU instructions for architectures like x86_64, ARM64, and RISC-V, with semantics defined by what the silicon does.

MLIR's progressive lowering walks down this ladder one step at a time. Each pass lowers operations by one level, maintaining well-formed IR at every step.

### Why Progressive Lowering?

Why not compile directly from high-level operations to machine code in one pass? Several reasons:

**Modularity**: Each lowering step is simple and self-contained. Converting `linalg.matmul` → loops requires understanding Linalg semantics. Converting loops → basic blocks requires understanding loop structure. Separating these concerns makes each pass simpler to implement, test, and maintain.

**Composability**: You can insert transformations at any level. Want to tile loops before converting to basic blocks? Insert a loop tiling pass between Linalg and SCF lowering. Want to fuse operations? Do it at the Linalg level where structure is explicit.

**Reusability**: The SCF → CF lowering works for **any** structured control flow, whether it came from Linalg, Tensor ops, or custom dialects. Write once, use everywhere. This is MLIR's superpower—each dialect doesn't need its own lowering to machine code, just lowering to an intermediate dialect that already has lowering paths defined.

**Debugging**: When something goes wrong, you can identify **which lowering step** failed. Did `linalg.matmul` lower to incorrect loops? Check the Linalg → Loops pass. Do loops have incorrect bounds? Check the loop construction logic. Progressive lowering provides intermediate checkpoints for debugging.

### The Typical Compilation Pipeline

**Connection to Chapter 1**: In Chapter 1, we saw progressive lowering in action when our `linalg.matmul` operation transformed through multiple passes: Linalg → SCF → CF → LLVM. At the time, we focused on the "what" (which passes to call). Now we understand the "why" (maintaining well-formed IR at each abstraction level) and the "how" (the PassManager orchestrating transformations).

For the matrix multiplication in Chapters 1-2, here's the complete lowering sequence:

**Phase 1: Canonicalization**
```mlir
// Clean up redundant operations, fold constants
func.func @gemm(...) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst) outs(%C)
  linalg.matmul ins(%A, %B) outs(%C)
}
```

**Phase 2: Linalg → Loops**
```mlir
// Structured operations → explicit loops
func.func @gemm(...) {
  %M = memref.dim %A, 0
  %N = memref.dim %B, 1
  %K = memref.dim %A, 1
  scf.for %i = 0 to %M {
    scf.for %j = 0 to %N {
      scf.for %k = 0 to %K {
        %a = memref.load %A[%i, %k]
        %b = memref.load %B[%k, %j]
        // ... multiply-add operations
      }
    }
  }
}
```

**Phase 3: SCF → CF**
```mlir
// Structured loops → basic blocks and branches
func.func @gemm(...) {
  %M = memref.dim %A, 0
  cf.br ^loop_i_header
^loop_i_header:
  %i = ...
  %cond = arith.cmpi ult, %i, %M
  cf.cond_br %cond, ^loop_i_body, ^exit
^loop_i_body:
  // Nested loop structure as basic blocks
  %i_next = arith.addi %i, %c1
  cf.br ^loop_i_header
^exit:
  return
}
```

**Phase 4: MemRef → LLVM**
```mlir
// Memref operations → LLVM pointer arithmetic
func.func @gemm(...) {
  %desc = llvm.extractvalue %A[...] // Extract descriptor fields
  %ptr = llvm.extractvalue %desc[1]  // Get data pointer
  %stride = llvm.extractvalue %desc[5] // Get stride
  %offset = llvm.mul %i, %stride
  %addr = llvm.getelementptr %ptr[%offset]
  %val = llvm.load %addr
  // ...
}
```

**Phase 5: LLVM IR → Machine Code**
```asm
; x86_64 assembly (conceptual)
gemm:
  mov rax, [rdi + 8]    ; Load data pointer
  mov rcx, [rdi + 40]   ; Load stride
  imul rdx, rcx         ; Compute offset
  movss xmm0, [rax + rdx*4]  ; Load float
  ; ... multiply-add instructions
  ret
```

Each phase maintains IR correctness. You can stop at any phase, inspect the IR, and resume. This is fundamentally different from traditional compilers where intermediate stages aren't well-defined.

### Mixing Abstraction Levels

One unique MLIR feature: **IR can contain multiple abstraction levels simultaneously**. A single function might have high-level `linalg.matmul` operations alongside mid-level `scf.for` loops and low-level `llvm.load` instructions, all coexisting in the same function.

This happens naturally during progressive lowering—you lower some operations but not others. Consider a function with matrix multiplication and a simple addition:

```mlir
func.func @mixed() {
  %result = linalg.matmul ins(%A, %B) outs(%C)  // Not lowered yet
  %sum = llvm.fadd %x, %y : f32                 // Already lowered
  return
}
```

This is valid MLIR! The PassManager will eventually lower everything, but during compilation, mixed-level IR is normal and expected. This flexibility enables powerful transformations—lower some operations aggressively while keeping others high-level for analysis.

## 3.5 AOT Workflow: From MLIR to Object Files

Now we arrive at production reality: how do you actually deploy MLIR-compiled code? For serving systems, embedded devices, and performance-critical applications, the answer is **ahead-of-time compilation to object files**. This section explains the complete AOT workflow.

### The AOT Pipeline: Step by Step

**Step 1: Generate MLIR IR** (same as JIT):
```cpp
MLIRContext context;
auto module = createGemmModule(context);
```

**Step 2: Apply optimization passes** (same as JIT):
```cpp
PassManager pm(context);
pm.addPass(createCanonicalizerPass());
pm.addPass(createConvertLinalgToLoopsPass());
// ... more passes
pm.run(module);
```

**Step 3: Translate to LLVM IR**:
```cpp
// Register translation from MLIR dialects to LLVM IR
registerBuiltinDialectTranslation(context);
registerLLVMDialectTranslation(context);

// Convert MLIR to LLVM IR
llvm::LLVMContext llvmContext;
auto llvmModule = translateModuleToLLVMIR(*module, llvmContext);
```

This converts MLIR's LLVM dialect operations to actual LLVM IR. After this step, you have a standard LLVM module that any LLVM tool can process. **This is what production systems do instead of ExecutionEngine**—they compile to object files that can be deployed without runtime compilation overhead.

**Step 4: Configure target machine**:
```cpp
// Specify what hardware you're compiling for
llvm::Triple triple("x86_64-unknown-linux-gnu");  // Or "aarch64-apple-darwin", etc.

std::string error;
const llvm::Target* target = llvm::TargetRegistry::lookupTarget(
    triple.getTriple(), error);

// Configure CPU features and optimization level
llvm::TargetOptions options;
llvm::TargetMachine* targetMachine = target->createTargetMachine(
    triple.getTriple(),
    "generic",        // CPU name (or specific: "haswell", "skylake", "arm-v8")
    "",               // CPU features (e.g., "+avx2,+fma")
    options,
    llvm::Reloc::PIC_ // Position-independent code
);
```

The target machine specification determines what instructions the compiler can use. Targeting "skylake" enables AVX-512; targeting "generic" produces code that runs on any x86_64 CPU.

**Step 5: Run LLVM optimization passes**:
```cpp
llvm::legacy::PassManager llvmPM;

// Add optimization passes (inlining, vectorization, etc.)
llvm::PassManagerBuilder builder;
builder.OptLevel = 3;  // -O3 equivalent
builder.SizeLevel = 0;
builder.Inliner = llvm::createFunctionInliningPass();
builder.populateModulePassManager(llvmPM);

// Optimize the LLVM IR
llvmPM.run(*llvmModule);
```

This applies LLVM's optimization suite—the same optimizations that Clang uses when compiling C++. This is where aggressive optimizations happen: loop vectorization, auto-parallelization, instruction scheduling.

**Step 6: Emit object file**:
```cpp
// Open output file
std::error_code EC;
llvm::raw_fd_ostream dest("gemm.o", EC, llvm::sys::fs::OF_None);

// Configure code generation
llvm::legacy::PassManager codegen;
targetMachine->addPassesToEmitFile(
    codegen,
    dest,
    nullptr,
    llvm::CodeGenFileType::CGFT_ObjectFile  // Produce .o file
);

// Generate machine code and write to file
codegen.run(*llvmModule);
dest.flush();
```

This produces a **relocatable object file** (`.o` on Unix, `.obj` on Windows)—the same format that C++ compilers produce. The object file contains machine code for all functions, a symbol table listing function names and global variables, relocation information for the linker to resolve cross-file references, and debugging information if enabled during compilation.

**Step 7: Link into library or executable**:
```bash
# Link object file into shared library
clang -shared gemm.o -o libgemm.so

# Or link into executable with main()
clang gemm.o main.o -o my_program
```

The linker resolves references between object files, applies relocations, and produces a final executable or library. This is standard compilation—nothing MLIR-specific at this point.

### Loading and Calling AOT-Compiled Code

From Python (or any language), you load the compiled library and call functions:

```python
import ctypes
import numpy as np

# Load shared library
lib = ctypes.CDLL('./libgemm.so')

# Define function signature (21 parameters for 3 memrefs)
lib.gemm.argtypes = [
    # First memref (A): allocated, aligned, offset, size0, size1, stride0, stride1
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    # Second memref (B): same structure
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    # Third memref (C): same structure
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
]

# Call the function
A = np.random.randn(8, 32).astype(np.float32)
B = np.random.randn(32, 16).astype(np.float32)
C = np.zeros((8, 16), dtype=np.float32)

lib.gemm(
    A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    0,  # offset
    A.shape[0], A.shape[1],  # sizes
    A.strides[0] // 4, A.strides[1] // 4,  # strides (in elements, not bytes)
    # ... repeat for B and C
)
```

This looks verbose, but it's **exactly what ExecutionEngine does internally**. The difference: no compilation at runtime, no ExecutionEngine overhead. The library loaded in microseconds; function calls are pure native code.

### Cross-Compilation: Building for Different Platforms

AOT compilation enables **cross-compilation**—build on one platform, run on another:

```cpp
// Build on x86_64 Linux, target ARM64 Android
llvm::Triple triple("aarch64-linux-android");

// Use ARM-specific optimizations
llvm::TargetMachine* tm = target->createTargetMachine(
    triple.getTriple(),
    "cortex-a76",  // Specific ARM CPU
    "+neon",       // Enable NEON SIMD
    options,
    llvm::Reloc::PIC_
);
```

This generates ARM machine code on your x86_64 development machine. Ship the resulting `.so` to Android devices—it runs natively, no recompilation needed. This is how mobile ML frameworks work: compile models on servers, deploy to billions of devices.

### Comparison: AOT vs ExecutionEngine

Let's make the comparison concrete.

**ExecutionEngine (JIT)** follows a simple workflow: pass IR to ExecutionEngine.create() and get back a function pointer. The compilation time is 10-100ms on the first call. Memory overhead is substantial at 50-200 MB for the compiler infrastructure. The output is code in process memory that's ephemeral—it disappears when the process exits. Deployment requires shipping either source code or IR along with the compiler. Platform support is limited to environments where JIT compilation is allowed, excluding iOS, many embedded systems, and secure environments.

**AOT (object files)** has a more complex build-time workflow: translate IR to LLVM, optimize, emit object files (.o), link into libraries (.so), but then runtime is trivial. Compilation time at runtime is 0ms since all compilation happened upfront. Memory overhead is minimal—only the compiled code itself, typically measured in kilobytes. The output is a relocatable object file (.o) or shared library (.so) that persists on disk. Deployment ships only the compiled binaries with no compiler needed. Platform support is universal, including iOS, embedded systems, and secure environments that prohibit runtime code generation.

For production, AOT wins decisively. For experimentation and rapid development, ExecutionEngine is more convenient.

## 3.6 JIT Workflow: ExecutionEngine in Depth and Caching Strategies

While production systems favor AOT, JIT compilation has legitimate uses in development, experimentation, and certain dynamic workloads. Let's understand how MLIR's ExecutionEngine works internally and how caching mitigates JIT overhead.

### ExecutionEngine Internals

When you call `ExecutionEngine::create(module, options)`, here's what happens under the hood:

**Phase 1: Dialect translation registration**:
The ExecutionEngine needs to know how to translate MLIR dialects to LLVM IR. You register translations:
```cpp
registerBuiltinDialectTranslation(context);
registerLLVMDialectTranslation(context);
```
These register conversion functions for each dialect. When the engine encounters a `memref.load`, it looks up the registered translator and converts it to LLVM's `load` instruction.

**Phase 2: Translation to LLVM IR**:
The entire MLIR module is converted to an LLVM module:
```cpp
llvm::LLVMContext llvmContext;
auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
```
This produces standard LLVM IR—the same IR that Clang produces from C++. From this point, we're in pure LLVM territory.

**Phase 3: Optimization**:
If you provided an optimization transformer, it runs:
```cpp
auto transformer = makeOptimizingTransformer(
    /*optLevel=*/3,
    /*sizeLevel=*/0,
    /*targetMachine=*/nullptr
);
transformer(llvmModule);
```
This applies LLVM's optimization passes: inlining, vectorization, dead code elimination, constant folding, etc. This is where the compilation time is spent—aggressive optimization takes time.

**Phase 4: JIT compilation—MLIR ExecutionEngine and LLVM ORC**:
In our code examples throughout this book, we use **MLIR's ExecutionEngine**, not raw LLVM JIT directly. It's important to understand the relationship between these layers:

**The layering architecture**:
```
┌─────────────────────────────────────┐
│  MLIR ExecutionEngine               │  ← What we use in chapters 1-3
│  (mlir::ExecutionEngine)            │     Convenience wrapper
├─────────────────────────────────────┤
│  LLVM ORC JIT (llvm::orc::LLJIT)    │  ← LLVM's JIT infrastructure
│  Lazy compilation, symbol tables    │     Core compilation engine
├─────────────────────────────────────┤
│  Machine code generation            │  ← Actual x86/ARM codegen
│  Memory management                  │     OS interaction
└─────────────────────────────────────┘
```

**What MLIR's ExecutionEngine provides**:
MLIR's `ExecutionEngine` is a thin wrapper that handles MLIR-specific concerns:

```cpp
// High-level MLIR API (what we use)
auto engine = ExecutionEngine::create(module, options);
auto funcPtr = engine->lookup("gemm");
```

Internally, ExecutionEngine performs several critical operations. It **registers MLIR dialect translations**, automatically setting up translators for MLIR's dialects (MemRef, SCF, Arith, etc.) to LLVM IR. It **manages MLIR to LLVM conversion** by calling `translateModuleToLLVMIR()` with the correct context. It **wraps LLVM ORC LLJIT**, creating and managing an `llvm::orc::LLJIT` instance. It **simplifies the optimization pipeline** by providing `makeOptimizingTransformer()` for standard LLVM passes. It **handles symbol export**, making MLIR function names visible to the JIT's symbol table. Finally, it provides **memory lifetime management**, owning the LLVM context, JIT engine, and compiled code.

**Under the hood—what LLVM ORC does**:
When ExecutionEngine calls into LLVM's ORC (On-Request Compilation) framework:

```cpp
// What ExecutionEngine does internally (simplified)
auto jit = llvm::orc::LLJITBuilder().create();
jit->addIRModule(llvm::orc::ThreadSafeModule(
    std::move(llvmModule), std::move(llvmContext)
));
```

LLVM's **LLJIT** (Lazy LLVM JIT) then performs several operations: it compiles the LLVM IR to machine code **in memory**, allocates executable memory pages with OS permissions set to read+execute, generates position-independent code suitable for dynamic loading, builds a symbol table mapping function names to memory addresses, and manages the code cache and memory layout.

No files are written to disk. The generated code lives in the process's address space, in executable memory pages. This is fundamentally different from AOT compilation, where code lives in `.o` or `.so` files.

**Why MLIR provides ExecutionEngine**: You could directly use LLVM's ORC JIT, but you'd need to manually translate MLIR dialects to LLVM IR, register all dialect translations, handle MLIR-specific ABI conventions like memref descriptors, set up optimization transformers correctly, and manage multiple LLVM contexts. ExecutionEngine encapsulates all this boilerplate, letting you focus on generating and executing MLIR IR. For learning and prototyping, this convenience is invaluable.

**The practical implication**:
When you see `ExecutionEngine` in our code, remember: it's a convenience wrapper around LLVM ORC. The actual JIT compilation is done by LLVM's battle-tested ORC infrastructure—the same infrastructure used by Clang's JIT, Julia's compiler, and other production systems. MLIR just makes it easier to use from MLIR IR.

**Phase 5: Symbol lookup**:
To call a function, we look up its address:
```cpp
auto symbolAddr = jit->lookup("gemm");
auto* funcPtr = symbolAddr->getAddress();
```
The JIT maintains a symbol table mapping function names to memory addresses. Looking up "gemm" returns the address of the compiled function—you cast this to a function pointer and call it directly.

**Phase 6: Execution**:
Calling the function executes native machine code directly:
```cpp
auto* gemmFunc = reinterpret_cast<GemmFnPtr>(funcPtr);
gemmFunc(args...);  // Runs at native speed
```
There's no interpretation, no bytecode, no overhead—this is compiled x86_64/ARM64 instructions executing on bare metal.

### The Caching Strategy: Compile Once, Use Many Times

**Connection to Chapter 1**: In Chapter 1, we mentioned that "Chapter 3 will explore caching strategies to avoid recompiling the same code repeatedly." Here's that promise fulfilled. The key insight is that with dynamic shapes, compiled code is shape-agnostic—compile once, reuse for all matrix sizes.

The key insight for practical JIT: **cache the compiled function**. Since we use dynamic shapes (`memref<?x?xf32>`), the compiled code is shape-agnostic—it works for **any** matrix dimensions. Compile once, reuse forever.

From the ch.3 code, the caching pattern:

```cpp
// Global cache
struct GlobalJITCache {
  ExecutionEngine* engine = nullptr;
  GemmFnPtr funcPtr = nullptr;
  bool isCompiled = false;
} gGemmJIT;

void executeGemm(float* A, float* B, float* C, ...) {
  // Check cache
  if (!gGemmJIT.isCompiled) {
    // First call: compile (noticeable overhead)
    auto [engine, funcPtr] = compileGemmFunction();
    gGemmJIT.engine = engine;
    gGemmJIT.funcPtr = funcPtr;
    gGemmJIT.isCompiled = true;
  }
  
  // All calls: execute cached function (fast)
  gGemmJIT.funcPtr(A, B, C, ...);
}
```

**Why caching matters for performance**: The first call pays compilation cost—this can be noticeable for even simple functions and substantial for complex modules. This "warmup" phase creates unpredictable latency: the first invocation is much slower than subsequent ones. For production serving systems where consistent latency matters, this unpredictability is problematic. But for development, testing, and workloads where the same function is called many times, caching makes JIT overhead negligible.

**Key insight**: With dynamic shapes, the compiled code is reusable across different input sizes. Compile once for `memref<?x?xf32>`, use for any matrix dimensions. This amortization is why JIT can be practical for research and prototyping despite the initial compilation overhead.

### When Caching Breaks: Recompilation Triggers

Caching assumes the compiled code remains valid. When does it need to recompile?

**Different operation semantics**: If the *type* of computation changes (from matrix multiply to convolution), you need different compiled code. Cache by operation type.

**Different data types**: A function for `f32` won't work for `f64` or `i32`. Cache by data type.

**Different optimization flags**: Compiling with `-O0` vs `-O3` produces different code. Cache by optimization level.

**Dynamic shapes**: These **don't** require recompilation! That's the whole point of dynamic shapes. The same compiled code works for 8×32 and 1024×2048 matrices.

**Static shapes (if used)**: If you hardcode dimensions (`memref<8x32xf32>`), recompiling for different dimensions is necessary. But we avoid this by using dynamic shapes.

### Advanced Caching: Persistent Cache to Disk

Some systems cache compiled code **to disk** to avoid recompilation across program restarts:

```cpp
// Pseudocode for persistent caching
std::string cacheFile = "gemm_cache.bin";

if (fileExists(cacheFile)) {
  // Load cached compiled code from disk
  auto cachedCode = loadFromFile(cacheFile);
  gGemmJIT.funcPtr = reinterpret_cast<GemmFnPtr>(cachedCode);
} else {
  // Compile and save to disk
  auto [engine, funcPtr] = compileGemmFunction();
  saveToFile(cacheFile, funcPtr);
  gGemmJIT.funcPtr = funcPtr;
}
```

PyTorch 2.0 (torch.compile) uses this strategy—cache compiled models to disk, reload them instantly on subsequent runs. The user experience: first run is slow (compiling), subsequent runs are fast (loading cached code).

This hybrid approach combines JIT's flexibility (compile based on runtime information) with AOT's performance (no recompilation on restart). It's a sweet spot for development workflows.

## 3.7 Performance Considerations and Trade-offs

We've seen two compilation strategies—let's synthesize the trade-offs and performance implications.

### Compilation Time vs Execution Time

The fundamental trade-off: **time spent compiling vs time spent executing**.

Consider **Scenario 1: Run once**. A program compiles and executes a function **once**, then exits. AOT compilation requires substantial upfront build time before any execution. JIT compilation delays compilation until the function is actually called. **JIT is faster** for single-execution workloads since it avoids compiling code that might never run and has lower baseline compilation overhead.

In **Scenario 2: Run many times**, the function executes repeatedly in a loop or across many invocations. With AOT, you pay the compilation cost once upfront, then every execution is fast. With JIT, you pay compilation cost on the first call, then subsequent calls are fast. **Both converge to similar performance** once the JIT overhead is amortized over many calls.

**Scenario 3: Production serving** considers a server handling millions of requests over days or weeks. When compilation cost is amortized over this many executions, both approaches have negligible per-request overhead. **Winner: Tie** in aggregate throughput.

But scenario 3 has a critical difference: **latency distribution**. With AOT, all requests have consistent latency. With JIT, the first request pays the compilation penalty—a significant outlier. For user-facing services, this outlier is unacceptable. Serving systems care about p99 latency (99th percentile), and JIT's first-request penalty ruins this metric.

### Memory Footprint

**ExecutionEngine overhead**: LLVM's JIT infrastructure consumes 50-150 MB of memory, compiled code takes 10-500 KB per function, and symbol tables and metadata require 5-20 MB. For a single process, this is manageable. But serving systems run hundreds of worker processes. If each worker loads ExecutionEngine, the memory waste adds up quickly: 100 workers × 100 MB equals 10 GB of wasted RAM. AOT-compiled libraries have **zero compiler overhead**—the binary contains only compiled code without LLVM infrastructure. The memory footprint is just the code itself, measured in kilobytes.

### Code Size and Bloat

JIT compilation generates code for the **current platform only**. AOT compilation for multiple platforms requires multiple binaries. For example, a hypothetical ML library might produce binaries of 500 KB for x86_64 Linux, 480 KB for ARM64 Linux, 510 KB for x86_64 macOS, and 490 KB for ARM64 macOS, totaling approximately 2 MB across all platforms.

But you distribute only one binary per platform—users download 500 KB, not 2 MB. And modern app stores handle this automatically (iOS delivers ARM64 to iPhones, x86_64 to M1 Mac emulation).

For JIT, you distribute **IR** (MLIR or LLVM IR) and compile on the target. IR is typically more compact than machine code, so you save download size. But users pay compilation cost on their devices—wasting their CPU, battery, and time.

### Security Implications

JIT compilation is a **security risk** in untrusted environments:

**Attack vector**: If an attacker can influence what code gets JIT-compiled (through inputs, configuration files, network payloads), they might be able to generate and execute malicious code.

**Mitigation**: AOT compilation eliminates this attack surface. The binary contains only pre-approved code. No code generation occurs at runtime—attackers cannot inject arbitrary machine code.

This is why iOS **prohibits JIT compilation** (except in Safari, which has special entitlements). Apps must be fully AOT-compiled. Android allows JIT for apps with special permissions, but secure applications avoid it.

### Development vs Production: The Right Tool for Each Phase

The industry has converged on a best practice that separates concerns by phase.

For development and research, use JIT compilation tools like ExecutionEngine, torch.compile, or JAX jit. Prioritize fast iteration, flexibility, and immediate feedback, accepting the trade-offs of compilation overhead, memory usage, and unpredictable latency.

For production deployment, use AOT compilation: compile to object files and link to libraries. Prioritize predictable latency, minimal memory footprint, security, and fast startup time, accepting the trade-offs of longer build times, platform-specific binaries, and less runtime flexibility.

This separation is clean and effective. Researchers don't worry about deployment complexity; production engineers don't fight with compilation overhead.

## 3.8 Summary

This chapter covered the infrastructure underlying MLIR compilation—knowledge essential for building production systems:

1. **AOT vs JIT compilation**: Two philosophies with different trade-offs. AOT (ahead-of-time) provides predictable performance and zero runtime overhead but sacrifices flexibility. JIT (just-in-time) offers runtime adaptability at the cost of compilation overhead and complexity.

2. **MLIR's ExecutionEngine**: A convenient wrapper around LLVM's LLJIT for prototyping and testing. Excellent for development, inappropriate for production. Production systems compile to object files and link into libraries—no runtime code generation.

3. **Pass infrastructure**: MLIR organizes transformations as composable passes managed by the PassManager. Passes operate at different granularities (module, function, operation) and can run in parallel. Understanding passes is key to both using and extending MLIR.

4. **Progressive lowering**: MLIR's compilation model walks down an abstraction ladder one step at a time. High-level operations (Linalg) → structured control flow (SCF) → basic blocks (CF) → LLVM IR → machine code. Each step maintains well-formed IR, enabling debugging and intermediate optimizations.

5. **AOT workflow**: The production compilation path: generate MLIR IR → apply passes → translate to LLVM IR → optimize → emit object files → link to libraries. This produces standalone binaries with zero runtime compilation overhead.

6. **JIT workflow with caching**: ExecutionEngine translates IR to machine code in memory. Combined with dynamic shapes, we compile once and reuse for all invocations. Caching amortizes compilation cost across many calls, making JIT practical for workloads with warmup tolerance.

7. **Performance trade-offs**: JIT adds 10-100ms warmup latency and 50-200 MB memory overhead. For production serving with millions of requests, AOT wins decisively. For research with thousands of training iterations, JIT's flexibility wins.

With compilation infrastructure understood, we're ready to tackle the final foundational piece: **bufferization** (Chapter 4), which bridges high-level tensor operations and low-level memory management. After Chapter 4, we'll have all the foundations needed to build increasingly sophisticated ML systems—from simple element-wise operations (Chapter 5) through transformers (Chapters 11-13) to production serving engines (Chapter 16).