# Chapter 3: Compilation Infrastructure: AOT, JIT, and the Pass Pipeline

In the first two chapters, we generated MLIR IR and compiled it to executable code using MLIR's `ExecutionEngine`. We treated compilation as a black box—feed it IR, get back a callable function. This simplicity was intentional for learning, but now we need to understand what's happening inside that black box.

This chapter addresses fundamental questions about compilation infrastructure: **What is the difference between ahead-of-time (AOT) and just-in-time (JIT) compilation? Why does MLIR provide an ExecutionEngine, and when should you use it? How does MLIR's pass infrastructure orchestrate transformations? And critically for production systems, how do you compile MLIR to standalone executables or libraries?**

Understanding these concepts is essential because the compilation strategy you choose has profound implications for deployment, performance, and flexibility. Modern ML systems like PyTorch 2.0, TensorRT, and XLA all make specific choices about AOT vs JIT, and understanding these trade-offs will help you architect better systems.

## 3.1 AOT vs JIT: Two Compilation Philosophies

Compilers have two fundamentally different approaches to generating executable code: compile everything before running anything (ahead-of-time), or compile code on-demand as the program runs (just-in-time). Each philosophy has deep implications for performance, flexibility, and deployment.

### Ahead-of-Time (AOT) Compilation

**AOT compilation** is the traditional compilation model exemplified by C, C++, Rust, and Go compilers. The workflow is strictly separated into two phases:

**Build time** (happens once):
1. Read source files
2. Parse and validate
3. Optimize intermediate representation
4. Generate machine code
5. Link into executables or libraries
6. Write binary files to disk

**Runtime** (happens many times):
1. Load precompiled binary from disk
2. Execute immediately—no compilation overhead

The key characteristic: **all compilation happens before the program runs**. When a user launches your application, they're executing pre-generated machine code. No compiler runs during execution.

**Advantages**:
- **Zero compilation overhead at runtime**: The program starts instantly and runs at full speed from the first instruction
- **Predictable performance**: Every execution has identical performance characteristics
- **Optimized for target hardware**: The compiler can tune code for specific CPU features (AVX-512, ARM NEON, etc.)
- **Simplified deployment**: Distribute simple binaries—no compiler or source code needed
- **Startup time**: Critical for services that must respond immediately (web servers, embedded systems)

**Disadvantages**:
- **No runtime adaptability**: Cannot adjust to runtime information (input shapes, sparsity patterns, hardware differences)
- **Binary size**: Supporting multiple platforms requires separate binaries (x86_64, ARM64, etc.)
- **Recompilation required**: Any code change requires rebuilding and redistributing binaries

### Just-in-Time (JIT) Compilation

**JIT compilation** delays code generation until runtime. The workflow interleaves compilation with execution:

**Runtime** (single unified phase):
1. Start with high-level representation (bytecode, IR, source)
2. As code is needed: compile to machine code on-demand
3. Cache compiled code for reuse
4. Execute compiled code
5. Repeat for new code paths

Modern JIT systems include Java's HotSpot VM, JavaScript engines (V8, SpiderMonkey), PyPy, LuaJIT, and increasingly, ML frameworks (PyTorch 2.0 with torch.compile, JAX).

**Advantages**:
- **Runtime adaptability**: Can specialize code based on actual data (input shapes, branch frequencies, type information)
- **Dynamic optimization**: Profile code during execution and recompile hot paths with aggressive optimizations
- **Single deployment artifact**: Ship high-level IR that compiles to native code on any platform
- **Incremental compilation**: Only compile code that actually executes

**Disadvantages**:
- **Compilation overhead**: First execution of code pays compilation cost (10-100ms for small functions, seconds for large modules)
- **Memory overhead**: Must keep compiler infrastructure in memory
- **Unpredictable performance**: First run is slow (warming up), subsequent runs are fast
- **Complex implementation**: JIT compilers are significantly more complex than AOT compilers

### The ML Systems Landscape

Machine learning compilation sits in an interesting position. Consider how different systems choose:

**AOT-dominant systems**:
- **TensorFlow/XLA**: Compiles computation graphs to optimized binaries offline
- **TensorRT**: Takes a trained model, compiles to GPU kernels, saves to file for deployment
- **ONNX Runtime**: Loads serialized models, applies graph optimizations, compiles kernels ahead of serving
- **TVM**: Compiles models for edge devices (mobile, IoT) where no compiler can run

Why AOT? Production serving prioritizes **predictable latency**. When a user queries a deployed model, the first inference must be as fast as the thousandth inference. Compilation overhead is unacceptable. The solution: compile once during model deployment, serve forever.

**JIT-friendly systems**:
- **PyTorch eager mode**: Interprets operations dynamically, no compilation
- **PyTorch 2.0 (torch.compile)**: JIT-compiles computation graphs from eager execution traces
- **JAX**: JIT-compiles function transformations (jit, grad, vmap) at first invocation
- **Triton**: JIT-compiles GPU kernels from Python-like syntax

Why JIT? Research and experimentation prioritize **flexibility**. Researchers change model architectures constantly. Recompiling and redeploying for every tweak kills productivity. JIT compilation enables: write code, run immediately, iterate fast. The compilation overhead (a few seconds on first run) is acceptable compared to hour-long model training times.

**The hybrid approach**:
Many production systems use **both**:
1. **Development**: JIT for fast iteration
2. **Deployment**: AOT for predictable performance

PyTorch exemplifies this: researchers use eager mode (no compilation) or torch.compile (JIT) during development, then export models to TorchScript or ONNX for AOT compilation before deployment. This gives the best of both worlds—flexibility during research, performance in production.

## 3.2 MLIR's ExecutionEngine: A Prototyping Tool, Not a Production Solution

Throughout Chapters 1 and 2, we used MLIR's `ExecutionEngine` to compile and execute our matrix multiplication. It worked beautifully—pass in MLIR IR, get back a callable function pointer. But it's crucial to understand: **ExecutionEngine is a convenience tool for prototyping, not how production MLIR systems compile code**.

### What ExecutionEngine Provides

`ExecutionEngine` is a thin wrapper around LLVM's LLJIT (Lazy LLVM JIT), providing a simple interface for JIT compilation:

```cpp
// High-level API (what we've used)
auto engine = ExecutionEngine::create(module, options);
auto funcPtr = engine->lookup("gemm");
(*funcPtr)(args...);  // Call the compiled function
```

Under the hood, ExecutionEngine:
1. **Translates MLIR → LLVM IR**: Converts MLIR's dialects to LLVM IR using registered translations
2. **Optimizes LLVM IR**: Applies LLVM's optimization passes (inlining, constant folding, vectorization, etc.)
3. **Generates machine code**: Compiles LLVM IR to native instructions for the current CPU
4. **Keeps code in memory**: No files written to disk—code lives in process memory
5. **Provides function lookup**: Maps function names to callable function pointers
6. **Manages memory**: Handles code cache, data sections, and dynamic linking

This is enormously convenient for learning and testing—you get a runnable function in milliseconds with just a few lines of code. But convenience comes with trade-offs.

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

Real ML serving systems using MLIR follow an AOT workflow:

**TensorFlow's MLIR pipeline**:
1. Generate MLIR IR from TensorFlow graph
2. Apply optimization passes
3. Lower to LLVM IR
4. Compile to object files (.o)
5. Link into shared library (.so)
6. Deploy library to servers
7. Servers load library and call functions—no compilation

**Triton compiler** (despite being "JIT" in spirit):
1. Parse Triton Python code to IR
2. Lower to LLVM IR
3. Generate PTX (NVIDIA) or AMDGPU code
4. Pass to GPU driver for final compilation
5. Cache compiled kernels on disk
6. Reuse cached kernels across runs

Even systems with "JIT" in their name ultimately do AOT compilation—they just do it lazily on first use, then cache the results.

### The Right Tool for the Job

Think of ExecutionEngine as **scaffolding**—temporary infrastructure that supports development but isn't part of the final product. You wouldn't ship a building with scaffolding still attached, and you shouldn't ship a production service with ExecutionEngine embedded.

For this book, we use ExecutionEngine because:
1. It simplifies learning—focus on IR, not build systems
2. Enables rapid experimentation—change code, test immediately
3. Makes Python integration trivial—no linking or loading complexity

But as we progress toward production-grade systems (Chapters 14-16 on GPT serving), we'll discuss how real deployments work: compile MLIR to object files, link to create libraries, deploy statically compiled binaries.

## 3.3 The Pass Infrastructure: Orchestrating Transformations

At the heart of MLIR's compilation model is the **pass infrastructure**—a framework for organizing, sequencing, and applying transformations to IR. Every optimization, lowering, and analysis we've performed runs as a "pass" within this infrastructure. Understanding how passes work is essential for both using MLIR effectively and extending it with custom transformations.

### What is a Pass?

A **pass** is a self-contained transformation or analysis that operates on MLIR IR. Think of a pass as a single, focused task in the compilation pipeline:

**Transformation passes** modify the IR:
- **Canonicalization**: Simplifies IR to canonical forms (e.g., `x + 0` → `x`)
- **Inlining**: Replaces function calls with function bodies
- **Loop unrolling**: Replicates loop bodies to eliminate loop overhead
- **Constant folding**: Evaluates constant expressions at compile time
- **Dead code elimination**: Removes operations that don't affect the result

**Analysis passes** gather information without modification:
- **Liveness analysis**: Which values are live at each program point?
- **Alias analysis**: Can two memrefs point to the same memory?
- **Call graph construction**: What functions call which other functions?

**Lowering passes** convert between abstraction levels:
- **Linalg → Loops**: Transform `linalg.matmul` into nested `scf.for` loops
- **SCF → CF**: Convert structured control flow to basic blocks and branches
- **MemRef → LLVM**: Lower memref operations to LLVM pointer operations

Each pass has a narrow responsibility. This modularity is powerful: you can understand, test, and debug each pass independently. Composing passes creates complex transformations from simple building blocks.

### The PassManager: The Conductor

The **PassManager** orchestrates pass execution. It's responsible for:

1. **Scheduling passes** in the correct order
2. **Managing pass dependencies** (some passes require others to run first)
3. **Providing context** (access to MLIRContext, analyses, etc.)
4. **Handling errors** (if a pass fails, propagate the failure)
5. **Enabling parallelism** (run passes on different operations concurrently)

Creating a pass pipeline looks like this:

```cpp
PassManager pm(context);

// Add passes in sequence
pm.addPass(createCanonicalizerPass());           // Simplify IR
pm.addPass(createConvertLinalgToLoopsPass());    // Linalg → Loops
pm.addPass(createConvertSCFToCFPass());          // SCF → CF
pm.addPass(createFinalizeMemRefToLLVMPass());    // MemRef → LLVM

// Run all passes on the module
if (failed(pm.run(module))) {
  // Handle compilation failure
}
```

The PassManager ensures passes execute in order, manages memory efficiently, and can even run passes in parallel when safe (multiple passes on different functions simultaneously).

### Pass Granularity: Operating on Specific Operations

Passes don't just operate on entire modules—they operate on **specific operation types**. This granularity is crucial for modularity and performance.

**ModulePass**: Operates on an entire `ModuleOp`
- Use when transformation needs whole-program view
- Example: interprocedural optimizations, dead function elimination
- Cannot be parallelized across modules (each module is independent)

**FunctionPass**: Operates on individual functions
- Use for most function-level optimizations
- Example: loop transformations, local optimizations
- Can be parallelized: PassManager runs the same pass on multiple functions simultaneously

**OperationPass** (op-agnostic): Operates on any operation type
- Use for generic transformations that work on any IR
- Example: canonicalization, CSE (common subexpression elimination)
- Most flexible but requires careful implementation (don't assume operation structure)

**OperationPass** (op-specific): Operates on specific operation types
- Use when transformation only makes sense for certain ops
- Example: a pass that optimizes `linalg.matmul` specifically
- Type-safe: compiler enforces you're operating on the right op type

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

Some passes have subtle dependencies. Canonicalization, for example, should run **after** major transformations to clean up the IR they produce. Lowering passes create verbose IR with redundant operations; canonicalization simplifies it. The pattern is:

```
1. Major transformation (e.g., loop tiling)
2. Canonicalization (clean up mess)
3. Next transformation
```

### Pass Verification and Debugging

MLIR's pass infrastructure includes **verification** after each pass. By default, after every pass executes, MLIR verifies the entire IR is well-formed:
- All operations have valid types
- SSA dominance properties hold
- Dialect-specific invariants are maintained

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

**Level 5 - Domain-specific (highest abstraction)**:
- Custom dialects for specific domains (Transformer ops, database queries, etc.)
- Operations like `transformer.attention`, `graph.conv2d`
- Semantics: "what computation to perform" (mathematical specification)

**Level 4 - Structured compute**:
- Linalg dialect: `linalg.matmul`, `linalg.conv`, `linalg.reduce`
- Operations express structured linear algebra
- Semantics: iteration spaces, index maps, declarative operations

**Level 3 - Loops and structured control flow**:
- SCF dialect: `scf.for`, `scf.while`, `scf.if`
- Control flow with explicit structure (loop bounds, loop bodies)
- Semantics: imperative computation with structured control

**Level 2 - Basic blocks and branches**:
- CF dialect: `cf.br`, `cf.cond_br`
- Unstructured control flow (gotos)
- Semantics: assembly-level control flow

**Level 1 - LLVM IR**:
- LLVM dialect: `llvm.load`, `llvm.store`, `llvm.fadd`
- Close to machine instructions
- Semantics: register-based virtual machine

**Level 0 - Machine code** (lowest abstraction):
- x86_64, ARM64, RISC-V, etc.
- Actual CPU instructions
- Semantics: what the silicon does

MLIR's progressive lowering walks down this ladder one step at a time. Each pass lowers operations by one level, maintaining well-formed IR at every step.

### Why Progressive Lowering?

Why not compile directly from high-level operations to machine code in one pass? Several reasons:

**Modularity**: Each lowering step is simple and self-contained. Converting `linalg.matmul` → loops requires understanding Linalg semantics. Converting loops → basic blocks requires understanding loop structure. Separating these concerns makes each pass simpler to implement, test, and maintain.

**Composability**: You can insert transformations at any level. Want to tile loops before converting to basic blocks? Insert a loop tiling pass between Linalg and SCF lowering. Want to fuse operations? Do it at the Linalg level where structure is explicit.

**Reusability**: The SCF → CF lowering works for **any** structured control flow, whether it came from Linalg, Tensor ops, or custom dialects. Write once, use everywhere. This is MLIR's superpower—each dialect doesn't need its own lowering to machine code, just lowering to an intermediate dialect that already has lowering paths defined.

**Debugging**: When something goes wrong, you can identify **which lowering step** failed. Did `linalg.matmul` lower to incorrect loops? Check the Linalg → Loops pass. Do loops have incorrect bounds? Check the loop construction logic. Progressive lowering provides intermediate checkpoints for debugging.

### The Typical Compilation Pipeline

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

One unique MLIR feature: **IR can contain multiple abstraction levels simultaneously**. A single function might have:
- High-level `linalg.matmul` operations
- Mid-level `scf.for` loops
- Low-level `llvm.load` instructions

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

This converts MLIR's LLVM dialect operations to actual LLVM IR. After this step, you have a standard LLVM module that any LLVM tool can process.

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

This produces a **relocatable object file** (`.o` on Unix, `.obj` on Windows)—the same format that C++ compilers produce. The object file contains:
- Machine code for all functions
- Symbol table (function names, global variables)
- Relocation information (for the linker)
- Debugging information (if enabled)

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

Let's make the comparison concrete:

**ExecutionEngine (JIT)**:
- Workflow: IR → ExecutionEngine.create() → function pointer
- Time: 10-100ms compilation on first call
- Memory: 50-200 MB for compiler infrastructure
- Output: Code in process memory (ephemeral)
- Deployment: Ship source code or IR + compiler
- Platforms: Only where JIT is allowed

**AOT (object files)**:
- Workflow: IR → translate to LLVM → optimize → emit .o → link → .so
- Time: 0ms at runtime (all compilation upfront)
- Memory: Only the compiled code (kilobytes)
- Output: Relocatable object file (.o) or library (.so)
- Deployment: Ship compiled binaries (no compiler needed)
- Platforms: Any platform (including iOS, embedded, secure environments)

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

Internally, ExecutionEngine:
1. **Registers MLIR dialect translations**: Automatically sets up translators for MLIR's dialects (MemRef, SCF, Arith, etc.) to LLVM IR
2. **Manages MLIR → LLVM conversion**: Calls `translateModuleToLLVMIR()` with correct context
3. **Wraps LLVM ORC LLJIT**: Creates and manages an `llvm::orc::LLJIT` instance
4. **Simplifies optimization pipeline**: Provides `makeOptimizingTransformer()` for standard LLVM passes
5. **Handles symbol export**: Makes MLIR function names visible to the JIT's symbol table
6. **Memory lifetime management**: Owns the LLVM context, JIT engine, and compiled code

**Under the hood—what LLVM ORC does**:
When ExecutionEngine calls into LLVM's ORC (On-Request Compilation) framework:

```cpp
// What ExecutionEngine does internally (simplified)
auto jit = llvm::orc::LLJITBuilder().create();
jit->addIRModule(llvm::orc::ThreadSafeModule(
    std::move(llvmModule), std::move(llvmContext)
));
```

LLVM's **LLJIT** (Lazy LLVM JIT) then:
- Compiles the LLVM IR to machine code **in memory**
- Allocates executable memory pages (with OS permissions: read+execute)
- Generates position-independent code suitable for dynamic loading
- Builds a symbol table mapping function names to addresses
- Manages code cache and memory layout

No files are written to disk. The generated code lives in the process's address space, in executable memory pages. This is fundamentally different from AOT compilation, where code lives in `.o` or `.so` files.

**Why MLIR provides ExecutionEngine**:
You could directly use LLVM's ORC JIT, but you'd need to:
1. Manually translate MLIR dialects to LLVM IR
2. Register all dialect translations
3. Handle MLIR-specific ABI conventions (memref descriptors, etc.)
4. Set up optimization transformers correctly
5. Manage multiple LLVM contexts

ExecutionEngine encapsulates all this boilerplate, letting you focus on generating and executing MLIR IR. For learning and prototyping, this convenience is invaluable.

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
    // First call: compile (expensive, ~10-100ms)
    auto [engine, funcPtr] = compileGemmFunction();
    gGemmJIT.engine = engine;
    gGemmJIT.funcPtr = funcPtr;
    gGemmJIT.isCompiled = true;
  }
  
  // All calls: execute cached function (fast, <1µs)
  gGemmJIT.funcPtr(A, B, C, ...);
}
```

**First call**: Pays compilation cost (10-100ms for a simple function, seconds for complex modules). This is the "warmup" phase.

**Subsequent calls**: No compilation, just direct function invocation. Performance is identical to AOT-compiled code.

**Amortization**: If your program calls the function 1000 times, the compilation cost is amortized across all calls. Total overhead: 100ms ÷ 1000 = 0.1ms per call—negligible.

### When Caching Breaks: Recompilation Triggers

Caching assumes the compiled code remains valid. When does it need to recompile?

**Different operation semantics**: If the *type* of computation changes (from matrix multiply to convolution), you need different compiled code. Cache by operation type.

**Different data types**: A function for `f32` won't work for `f64` or `i32`. Cache by data type.

**Different optimization flags**: Compiling with `-O0` vs `-O3` produces different code. Cache by optimization level.

**Dynamic shapes**: These **don't** require recompilation! That's the whole point of dynamic shapes. The same compiled code works for 8×32 and 1024×2048 matrices.

**Static shapes (if used)**: If you hardcode dimensions (`memref<8x32xf32>`), recompiling for different dimensions is necessary. But we avoid this by using dynamic shapes.

### Performance Measurement: JIT Overhead in Practice

Let's quantify JIT overhead with realistic numbers:

**Compilation time** (measured on x86_64, 3.5 GHz CPU):
- Simple function (matrix add): 10-20ms
- Matrix multiplication (our example): 30-50ms
- Complex graph (10+ operations): 100-500ms
- Large model (transformer layer): 1-5 seconds

**Execution time** (for matrix multiplication 1024×1024):
- First call (with compilation): 35ms + 5ms = 40ms total
- Subsequent calls: 5ms each

**Amortization**:
- After 8 calls: 40ms + 7×5ms = 75ms total → 9.4ms per call (average)
- After 100 calls: 40ms + 99×5ms = 535ms total → 5.35ms per call (average)
- After 1000 calls: 40ms + 999×5ms = 5035ms total → 5.035ms per call (average)

For a serving system handling 1000 requests per second, the compilation cost becomes negligible after one second of runtime. But that first second—with unpredictable latency—is why production systems avoid JIT.

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

**Scenario 1: Run once**
- Program compiles and executes a function **once**, then exits
- AOT: Compile ahead (1 second), execute (1ms) → Total: 1001ms
- JIT: Compile on first call (50ms), execute (1ms) → Total: 51ms
- **Winner: JIT** (20x faster for single execution)

**Scenario 2: Run 100 times**
- Function executes **100 times** in a loop
- AOT: Compile ahead (1 second), execute 100×1ms → Total: 1100ms
- JIT: Compile on first call (50ms), execute 100×1ms → Total: 150ms
- **Winner: JIT** (7x faster due to amortization)

**Scenario 3: Production serving**
- Server handles **1 million requests** over days/weeks
- AOT: Compile ahead (1 second once), execute 1M×1ms → Total: 1001 seconds
- JIT: Compile on first request (50ms once), execute 1M×1ms → Total: 1000.05 seconds
- **Winner: Tie** (compilation cost is negligible at this scale)

But scenario 3 has a critical difference: **latency distribution**. With AOT, all requests take 1ms. With JIT, the first request takes 51ms—a 50x outlier. For user-facing services, this outlier is unacceptable. Serving systems care about p99 latency (99th percentile), and JIT's first-request penalty ruins this metric.

### Memory Footprint

**ExecutionEngine overhead**:
- LLVM's JIT infrastructure: 50-150 MB
- Compiled code: 10-500 KB per function
- Symbol tables and metadata: 5-20 MB

For a single process, this is manageable. But serving systems run hundreds of worker processes. If each worker loads ExecutionEngine:
- 100 workers × 100 MB = 10 GB of wasted RAM

AOT-compiled libraries have **zero compiler overhead**—the binary contains only compiled code, no LLVM infrastructure. Memory footprint: just the code itself (kilobytes).

### Code Size and Bloat

JIT compilation generates code for the **current platform only**. AOT compilation for multiple platforms requires multiple binaries. For example, a hypothetical ML library might produce:

- x86_64 Linux: 500 KB
- ARM64 Linux: 480 KB
- x86_64 macOS: 510 KB
- ARM64 macOS: 490 KB
- Total: ~2 MB

But you distribute only one binary per platform—users download 500 KB, not 2 MB. And modern app stores handle this automatically (iOS delivers ARM64 to iPhones, x86_64 to M1 Mac emulation).

For JIT, you distribute **IR** (MLIR or LLVM IR) and compile on the target. IR is typically more compact than machine code, so you save download size. But users pay compilation cost on their devices—wasting their CPU, battery, and time.

### Security Implications

JIT compilation is a **security risk** in untrusted environments:

**Attack vector**: If an attacker can influence what code gets JIT-compiled (through inputs, configuration files, network payloads), they might be able to generate and execute malicious code.

**Mitigation**: AOT compilation eliminates this attack surface. The binary contains only pre-approved code. No code generation occurs at runtime—attackers cannot inject arbitrary machine code.

This is why iOS **prohibits JIT compilation** (except in Safari, which has special entitlements). Apps must be fully AOT-compiled. Android allows JIT for apps with special permissions, but secure applications avoid it.

### Development vs Production: The Right Tool for Each Phase

The industry has converged on a best practice:

**Development and research**:
- Use JIT (ExecutionEngine, torch.compile, JAX jit)
- Prioritize: fast iteration, flexibility, immediate feedback
- Accept: compilation overhead, memory usage, unpredictable latency

**Production deployment**:
- Use AOT (compile to object files, link to libraries)
- Prioritize: predictable latency, minimal memory, security, startup time
- Accept: longer build times, platform-specific binaries, less runtime flexibility

This separation of concerns is clean and effective. Researchers don't worry about deployment complexity; production engineers don't fight with compilation overhead.

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
