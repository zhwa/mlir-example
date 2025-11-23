# MLIR Coding Pattern: A Practical Guide

This document explains the basic implementation pattern for building MLIR projects, based on our matrix multiplication example.

## Understanding Ownership: Reference vs Copy vs Pointer

**Why does the code sometimes pass by reference (`MLIRContext&`), sometimes by value (`ModuleOp`), and sometimes return `OwningOpRef<ModuleOp>`?**

This is about **ownership semantics** - who is responsible for keeping an object alive.

### The Two Fundamental Types

#### 1. `MLIRContext` - Always Passed by Reference

```cpp
// In ir.cpp
OwningOpRef<ModuleOp> createGemmModule(MLIRContext& context) {
    // ↑ Reference: We DON'T own the context
}

// In jit.cpp
void executeGemm(float* A, float* B, float* C) {
    MLIRContext context;  // ← We create and OWN the context
    // ↑ Stack allocation: automatically destroyed when function exits
    
    auto module = createGemmModule(context);  // Pass reference
}
```

**Why reference?**
- `MLIRContext` is **heavy** - it stores all dialects, types, attributes, and unique constants
- **Expensive to copy** - you never want to copy a context
- **Long-lived** - usually exists for the entire compilation session
- **Non-owning access** - functions that use it don't need to manage its lifetime

**Rule:** Always pass `MLIRContext` by reference (`&`) or pointer (`*`), never by value!

#### 2. `ModuleOp` - Lightweight Handle (Cheap to Copy)

```cpp
// In lowering.cpp
LogicalResult applyOptimizationPasses(ModuleOp module) {
    // ↑ By value: This is just a pointer-sized handle, cheap to copy
    MLIRContext* context = module.getContext();  // Can get back to context
}
```

**What is `ModuleOp`?**
- It's a **lightweight handle** (essentially a pointer under the hood)
- Copying it is cheap (just copies the pointer)
- The actual module data lives in the `MLIRContext`
- **Non-owning** - doesn't control the module's lifetime

**Think of it like:**
```cpp
// ModuleOp is similar to:
class ModuleOp {
    Operation* ptr;  // Just a pointer!
    // Copying this is cheap
};
```

**Why pass by value?**
- Cheap to copy (pointer-sized)
- Convenient: no need for `&` or `*` syntax
- Non-owning: the context manages the actual data

#### 3. `OwningOpRef<ModuleOp>` - Ownership Wrapper

```cpp
// In ir.cpp - RETURNS ownership
OwningOpRef<ModuleOp> createGemmModule(MLIRContext& context) {
    auto module = ModuleOp::create(loc);
    return module;  // ← Transfers ownership to caller
}

// In jit.cpp - TAKES ownership
auto mlirModule = createGemmModule(context);
// ↑ mlirModule now owns the module
// When mlirModule goes out of scope, the module is destroyed
```

**What is `OwningOpRef<T>`?**
- A **smart pointer** (like `std::unique_ptr`) for MLIR operations
- **Owns** the operation - responsible for destroying it
- Move-only (can't copy, only transfer ownership)
- Ensures operations get cleaned up properly

**Why use it?**
- **Lifetime management** - makes ownership explicit
- **RAII** - automatically cleans up when going out of scope
- **Safety** - prevents memory leaks

### Ownership Pattern Summary

```cpp
// === Pattern 1: Context (heavy, never copy) ===
MLIRContext context;                    // Create on stack (or heap)
doSomething(context);                   // Pass by reference
void doSomething(MLIRContext& ctx) {}   // Receive by reference

// === Pattern 2: ModuleOp (lightweight handle, cheap copy) ===
ModuleOp module = ...;                  // Lightweight copy
processModule(module);                  // Pass by value (cheap!)
void processModule(ModuleOp mod) {}     // Receive by value

// === Pattern 3: Ownership transfer (use smart pointer) ===
OwningOpRef<ModuleOp> createModule() {  // Return ownership
    return ModuleOp::create(...);       
}
auto mod = createModule();              // Take ownership
// mod destroyed here when out of scope

// === Pattern 4: Dereference to get handle ===
OwningOpRef<ModuleOp> ownedModule = ...;
applyPasses(*ownedModule);              // ← Dereference with *
//            ↑ Gets the ModuleOp handle (non-owning)
```

### Real Examples from the Project

```cpp
// ir.cpp - Creates and transfers ownership
OwningOpRef<ModuleOp> createGemmModule(MLIRContext& context) {
    //                                              ↑ Reference: don't own context
    auto module = ModuleOp::create(loc);
    return module;  // Transfer ownership to caller
}

// lowering.cpp - Receives non-owning handle
LogicalResult applyOptimizationPasses(ModuleOp module) {
    //                                         ↑ By value: cheap copy of handle
    MLIRContext* context = module.getContext();  // Get pointer back to context
    PassManager pm(context);  // Pass pointer
    return pm.run(module);    // Pass handle by value
}

// jit.cpp - Owns the module
void executeGemm(float* A, float* B, float* C) {
    MLIRContext context;  // Own the context (stack allocated)
    
    auto mlirModule = createGemmModule(context);  // Take ownership
    //   ↑ Type: OwningOpRef<ModuleOp>
    
    applyOptimizationPasses(*mlirModule);  // Pass non-owning handle
    //                       ↑ Dereference to get ModuleOp
    
    // Both context and mlirModule automatically destroyed here
}
```

### Key Takeaways

| Type | Size | Ownership | How to Pass |
|------|------|-----------|-------------|
| `MLIRContext` | Large | Owner manages | By reference `&` or pointer `*` |
| `ModuleOp` | Small (pointer) | Non-owning | By value (cheap copy) |
| `OwningOpRef<ModuleOp>` | Small | Owns the module | Move (transfer ownership) |

**Golden Rules:**
1. **Never copy `MLIRContext`** - always pass by reference/pointer
2. **`ModuleOp` is cheap to copy** - pass by value is fine
3. **Use `OwningOpRef`** when you need to manage lifetime
4. **Dereference `OwningOpRef` with `*`** to get the handle for passing to functions

---

## Core MLIR Data Types: The Building Blocks

**Understanding these types is essential for working with MLIR, especially when adding features like bufferization.**

### The SSA Value System

MLIR uses **Static Single Assignment (SSA)** form - every value is defined exactly once and used many times.

#### `Value` - The Fundamental Data Carrier

```cpp
// In ir.cpp
Value argA = entryBlock->getArgument(0);  // Get function argument
Value zero = builder.create<arith::ConstantOp>(loc, zeroAttr).getResult();
//    ↑ Represents an SSA value (like %0, %1, %arg0 in MLIR text)
```

**What is `Value`?**
- Represents a **single SSA value** in the IR
- Like a variable in traditional code, but **immutable** (defined once)
- Can be: function arguments, operation results, block arguments
- Cheap to copy (it's just a pointer-sized handle)

**Where you see it:**
```mlir
// In MLIR text form:
%0 = arith.constant 0.0 : f32          // %0 is a Value
%1 = linalg.fill ins(%0) outs(%arg2)   // %arg2 is a Value (function arg)
```

```cpp
// In C++ API:
Value constant = builder.create<arith::ConstantOp>(...).getResult();
//    ↑ Gets the Value produced by the operation
Value funcArg = block->getArgument(0);
//    ↑ Gets a function/block argument as a Value
```

#### `ValueRange` - Collection of Values

```cpp
// In ir.cpp
builder.create<linalg::FillOp>(
    loc,
    ValueRange{zeroConstant.getResult()},  // Input values
    ValueRange{argC}                       // Output values
);

builder.create<linalg::MatmulOp>(
    loc,
    TypeRange{},
    ValueRange{argA, argB},   // Multiple inputs
    ValueRange{argC}          // Output
);
```

**What is `ValueRange`?**
- Represents a **range/array of Values**
- Can be constructed from: `{value1, value2, ...}`, vector, or array
- Used when operations need multiple inputs/outputs
- **Important:** Many ops have separate input and output value ranges

**Common patterns:**
```cpp
ValueRange inputs = {val1, val2, val3};      // Multiple values
ValueRange single = {singleValue};           // Single value (still needs braces!)
ValueRange empty = {};                       // No values
```

### The Type System

#### `Type` - Describes Data Layout

```cpp
// In ir.cpp
auto f32Type = builder.getF32Type();                    // Scalar float32
auto memrefType = MemRefType::get({8, 32}, f32Type);    // 2D array of float32
auto tensorType = RankedTensorType::get({8, 32}, f32Type); // Tensor (for bufferization)
```

**Common MLIR Types:**

| Type | Example | Usage |
|------|---------|-------|
| **Scalar types** | `f32`, `f64`, `i32`, `i64` | Floating point and integers |
| **MemRefType** | `memref<8x32xf32>` | Mutable multi-dimensional arrays (pointers) |
| **RankedTensorType** | `tensor<8x32xf32>` | Immutable tensors (values) - need bufferization |
| **UnrankedTensorType** | `tensor<*xf32>` | Tensor with unknown rank |
| **FunctionType** | `(memref<8x32xf32>) -> ()` | Function signatures |
| **IndexType** | `index` | Architecture-dependent integer (for array indices) |

**Getting types from Values:**
```cpp
Value v = ...;
Type t = v.getType();  // What type is this value?

if (auto memrefTy = dyn_cast<MemRefType>(t)) {
    // It's a memref! Can access shape, element type, etc.
    ArrayRef<int64_t> shape = memrefTy.getShape();
    Type elemType = memrefTy.getElementType();
}
```

#### `TypeRange` - Collection of Types

```cpp
builder.create<linalg::MatmulOp>(
    loc,
    TypeRange{},              // No result types (memref version is in-place)
    ValueRange{argA, argB},
    ValueRange{argC}
);

// For ops that produce results:
auto funcType = builder.getFunctionType(
    TypeRange{type1, type2},  // Argument types
    TypeRange{returnType}     // Return types
);
```

**Why separate Value and Type?**
- **Value** = the runtime data/computation
- **Type** = the compile-time description of that data
- One describes "what flows through the program", the other "what shape it has"

### Structural Components

#### `Operation` - The Base Class

```cpp
Operation* op = builder.create<arith::ConstantOp>(loc, attr);
//        ↑ All ops inherit from Operation

// Query operation properties:
StringRef name = op->getName().getStringRef();  // "arith.constant"
unsigned numResults = op->getNumResults();
Value result = op->getResult(0);
```

**What is `Operation`?**
- **Base class** for all MLIR operations (like `linalg.matmul`, `arith.addf`, etc.)
- Contains: location, operands, results, attributes, regions
- Most specific op types (like `MatmulOp`) are lightweight wrappers around `Operation*`

#### `Block` - Basic Block of Code

```cpp
Block* entryBlock = funcOp.addEntryBlock();  // Create entry block for function
builder.setInsertionPointToStart(entryBlock);

// Get block arguments (like function parameters, but for blocks)
Value arg0 = entryBlock->getArgument(0);
Value arg1 = entryBlock->getArgument(1);
```

**What is `Block`?**
- A **linear sequence of operations** (basic block in compiler terms)
- Has **arguments** (values that are passed when jumping to this block)
- Has **terminator** operation at the end (return, branch, etc.)
- Functions contain blocks, loops contain blocks, etc.

```mlir
// MLIR text form showing blocks:
func.func @example(%arg0: f32) -> f32 {
  // This is the entry block
  %0 = arith.addf %arg0, %arg0 : f32
  return %0 : f32
}

// Loop with multiple blocks:
scf.for %i = %c0 to %c10 step %c1 {
  ^bb0:  // Block label
    // Loop body operations
    scf.yield
}
```

#### `Region` - Container for Blocks

```cpp
Region& bodyRegion = funcOp.getBody();  // Function's body region
Block& firstBlock = bodyRegion.front();
```

**What is `Region`?**
- A **container for one or more Blocks**
- Functions have regions, loops have regions, conditionals have regions
- Represents a **scope** in the program
- Not commonly manipulated directly (usually work with blocks or operations)

### Attributes - Compile-Time Constants

#### `Attribute` - Constant Data

```cpp
// In ir.cpp
auto zeroAttr = builder.getFloatAttr(f32Type, 0.0);  // Floating point constant
auto intAttr = builder.getI64IntegerAttr(42);        // Integer constant
auto strAttr = builder.getStringAttr("hello");       // String constant

// Dense arrays (for tensor initialization):
auto denseAttr = DenseElementsAttr::get(tensorType, ArrayRef<float>{1.0, 2.0, 3.0});
```

**What is `Attribute`?**
- **Compile-time constant data** attached to operations
- Examples: numeric constants, strings, array literals
- Stored in the context (uniqued - only one copy of each distinct value)
- **Not** runtime values (that's what `Value` is for)

**Common attributes:**
- `FloatAttr` - floating point constant
- `IntegerAttr` - integer constant
- `StringAttr` - string constant
- `DenseElementsAttr` - array/tensor of constants
- `ArrayAttr` - array of other attributes
- `TypeAttr` - holds a type as an attribute

### Location - Debug Information

```cpp
auto loc = builder.getUnknownLoc();  // No source info
auto fileLoc = builder.getFileLineColLoc(
    builder.getIdentifier("file.mlir"), 10, 5
);
```

**What is `Location`?**
- Tracks **source code location** for debugging
- Every operation has a location
- Can be: unknown, file/line/column, fused (multiple locations), etc.
- Helps with error reporting

### Quick Reference: When to Use What

| What You Need | Type to Use | Example |
|---------------|-------------|---------|
| Single SSA value | `Value` | `Value v = op.getResult(0);` |
| Multiple values | `ValueRange` | `ValueRange{v1, v2, v3}` |
| Data type info | `Type` | `auto t = builder.getF32Type();` |
| Multiple types | `TypeRange` | `TypeRange{t1, t2}` |
| Function argument | `Value` | `block->getArgument(i)` |
| Operation result | `Value` | `op.getResult(i)` |
| Constant data | `Attribute` | `builder.getFloatAttr(...)` |
| Basic block | `Block*` | `funcOp.addEntryBlock()` |
| Any operation | `Operation*` | Base class pointer |

### Types You'll Need for Bufferization

When you move to tensor-based code with bufferization, you'll use:

```cpp
// Tensor types (immutable values)
auto tensorType = RankedTensorType::get({8, 32}, f32Type);
auto unknownTensor = UnrankedTensorType::get(f32Type);  // Shape unknown at compile time

// Dynamic dimensions (for flexible shapes!)
auto dynamicTensor = RankedTensorType::get(
    {ShapedType::kDynamic, ShapedType::kDynamic},  // Both dimensions dynamic
    f32Type
);

// Operations will return tensors (values) instead of using memrefs (pointers)
auto matmul = builder.create<linalg::MatmulOp>(
    loc,
    TypeRange{resultTensorType},   // ← Returns a tensor!
    ValueRange{tensorA, tensorB},
    ValueRange{outputTensor}
);
Value resultTensor = matmul.getResult(0);  // Get the result as a Value
```

**Key difference:**
- **Memref version:** Takes mutable memrefs, modifies them in-place, returns nothing
- **Tensor version:** Takes immutable tensors, returns new tensor, requires bufferization pass

---

## The Big Picture: Three-Phase Pipeline

```
Phase 1: IR Generation  →  Phase 2: Optimization & Lowering  →  Phase 3: Execution
   (ir.cpp)                        (lowering.cpp)                    (jit.cpp)
```

Every MLIR project follows this pattern:
1. **Generate** high-level IR describing *what* to compute
2. **Lower** it progressively to machine-level IR describing *how* to compute
3. **Execute** by JIT-compiling or AOT-compiling to native code

---

## Phase 1: IR Generation (ir.cpp)

**Goal:** Build high-level MLIR operations that declare your computation declaratively.

### Key Data Types

| Type | Purpose | Example |
|------|---------|---------|
| `MLIRContext` | Container for all MLIR state (dialects, types, attributes) | `MLIRContext context;` |
| `OpBuilder` | Tool for constructing operations | `OpBuilder builder(&context);` |
| `ModuleOp` | Top-level container for functions | `ModuleOp::create(loc)` |
| `FuncOp` | Function declaration | `builder.create<func::FuncOp>(...)` |
| `Type` | Data types (f32, memref, tensor) | `builder.getF32Type()` |
| `Value` | SSA value (result of an operation) | `Value result = op.getResult()` |

### Implementation Pattern

```cpp
OwningOpRef<ModuleOp> createMyModule(MLIRContext& context) {
  // 1. Load required dialects
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<linalg::LinalgDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  
  // 2. Create builder and module
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  
  // 3. Set insertion point (where new ops will be added)
  builder.setInsertionPointToStart(module.getBody());
  
  // 4. Define types for your computation
  auto f32Type = builder.getF32Type();
  auto memrefType = MemRefType::get({8, 32}, f32Type);
  
  // 5. Create function with signature
  auto funcType = builder.getFunctionType(
    {memrefType, memrefType},  // inputs
    {}                         // outputs
  );
  auto funcOp = builder.create<func::FuncOp>(loc, "my_func", funcType);
  
  // 6. Create function body
  auto* entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  
  // 7. Get function arguments
  Value arg0 = entryBlock->getArgument(0);
  Value arg1 = entryBlock->getArgument(1);
  
  // 8. Build operations (the actual computation)
  auto zeroAttr = builder.getFloatAttr(f32Type, 0.0);
  auto zero = builder.create<arith::ConstantOp>(loc, zeroAttr);
  builder.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{arg0});
  
  // 9. Return
  builder.create<func::ReturnOp>(loc);
  
  return module;
}
```

### Key Concepts

- **Dialects**: Collections of related operations (func, linalg, arith, etc.)
  - Load them with `context.getOrLoadDialect<DialectName>()`
  
- **OpBuilder**: The swiss army knife for building IR
  - Always need to set insertion point before creating ops
  - `builder.create<OpType>(location, args...)` to create operations
  
- **Location**: Source location for debugging (use `builder.getUnknownLoc()` if you don't track source)

- **SSA Form**: Every value is defined once and used many times
  - Operations produce `Value`s
  - Use these values as inputs to other operations

---

## Phase 2: Optimization & Lowering (lowering.cpp)

**Goal:** Transform high-level operations into low-level LLVM IR through progressive lowering.

### Key Data Types

| Type | Purpose | Example |
|------|---------|---------|
| `PassManager` | Orchestrates transformation passes | `PassManager pm(context);` |
| `Pass` | Individual transformation | `createCanonicalizerPass()` |
| `ModuleOp` | Module to transform (in-place) | Input/output of pipeline |

### Implementation Pattern

```cpp
LogicalResult applyOptimizationPasses(ModuleOp module) {
  MLIRContext* context = module.getContext();
  PassManager pm(context);
  
  // Add passes in order (they run sequentially)
  // Each pass transforms the IR one step closer to LLVM
  
  // 1. Cleanup/optimization
  pm.addPass(createCanonicalizerPass());
  
  // 2. High-level → mid-level (linalg → loops)
  pm.addPass(createConvertLinalgToLoopsPass());
  
  // 3. Mid-level → low-level (structured control flow → branches)
  pm.addPass(createConvertSCFToCFPass());
  
  // 4. Memory abstractions → LLVM (memref → pointers)
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  
  // 5. Everything else → LLVM dialect
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  
  // 6. Cleanup
  pm.addPass(createReconcileUnrealizedCastsPass());
  
  // Run the pipeline
  return pm.run(module);
}
```

### Progressive Lowering Levels

```
High-level:   linalg.matmul
    ↓ createConvertLinalgToLoopsPass()
Mid-level:    scf.for loops with body computations
    ↓ createConvertSCFToCFPass()
Control-flow: cf.br (basic blocks and branches)
    ↓ createFinalizeMemRefToLLVMConversionPass()
Low-level:    llvm.load, llvm.fadd, llvm.store
    ↓ translateModuleToLLVMIR()
LLVM IR:      %1 = load float*, %2 = fadd float, ...
```

### Key Concepts

- **PassManager**: Runs passes in sequence, each modifying the IR in-place
- **Progressive Lowering**: Each pass handles one abstraction level
  - Don't try to go from linalg directly to LLVM in one pass!
  - Use intermediate representations (loops, control flow, etc.)
- **Return LogicalResult**: `success()` or `failure()` to indicate pass success

---

## Phase 3: Execution (jit.cpp)

**Goal:** Translate MLIR to LLVM IR, JIT-compile to native code, and execute.

### Key Data Types

| Type | Purpose | Example |
|------|---------|---------|
| `llvm::LLVMContext` | LLVM's IR context (separate from MLIR!) | `auto ctx = std::make_unique<llvm::LLVMContext>()` |
| `llvm::Module` | LLVM IR module | Result of `translateModuleToLLVMIR()` |
| `llvm::orc::LLJIT` | JIT compiler | `llvm::orc::LLJITBuilder().create()` |
| `ExecutorAddr` | Function pointer in JIT memory | Result of `JIT->lookup("func_name")` |

### Implementation Pattern

```cpp
void executeMyFunction(float* input, float* output) {
  // 1. Initialize LLVM native target (once per process)
  static bool initialized = false;
  if (!initialized) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    initialized = true;
  }
  
  // 2. Generate MLIR and apply optimization passes
  MLIRContext context;
  context.loadDialect<func::FuncDialect, LLVM::LLVMDialect>();
  
  auto mlirModule = createMyModule(context);
  if (failed(applyOptimizationPasses(*mlirModule))) {
    return; // Handle error
  }
  
  // 3. Register dialect translations (for MLIR → LLVM IR)
  registerBuiltinDialectTranslation(*mlirModule->getContext());
  registerLLVMDialectTranslation(*mlirModule->getContext());
  
  // 4. Translate MLIR → LLVM IR
  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = translateModuleToLLVMIR(*mlirModule, *llvmContext);
  
  // 5. Apply LLVM optimizations (optional but recommended)
  auto optPipeline = makeOptimizingTransformer(
    /*optLevel=*/3,      // -O3 optimization
    /*sizeLevel=*/0,     // Don't optimize for size
    /*targetMachine=*/nullptr
  );
  optPipeline(llvmModule.get());
  
  // 6. Create JIT compiler
  auto JIT = llvm::orc::LLJITBuilder().create();
  if (!JIT) {
    return; // Handle error
  }
  
  // 7. Add LLVM IR to JIT
  auto TSM = llvm::orc::ThreadSafeModule(
    std::move(llvmModule),
    std::move(llvmContext)
  );
  (*JIT)->addIRModule(std::move(TSM));
  
  // 8. Look up function by name
  auto Sym = (*JIT)->lookup("my_func");
  if (!Sym) {
    return; // Handle error
  }
  
  // 9. Cast to function pointer and call
  using FnPtr = void(*)(float*, float*);
  auto* func = Sym->toPtr<FnPtr>();
  func(input, output);
}
```

### Key Concepts

- **Two separate contexts**:
  - `MLIRContext` for MLIR operations
  - `llvm::LLVMContext` for LLVM IR
  - They're independent! Data flows from MLIR → LLVM via translation

- **Translation registration**: 
  - Must register dialect translations before calling `translateModuleToLLVMIR()`
  - This tells MLIR how to convert each dialect to LLVM IR

- **ThreadSafeModule**: 
  - Bundles LLVM module with its context
  - Required by LLJIT API for thread safety

- **Function signatures**:
  - Match the MLIR function signature exactly
  - **Memrefs expand!** `memref<8x32xf32>` becomes 7 arguments:
    - `(float* allocated, float* aligned, int64_t offset, int64_t size0, int64_t size1, int64_t stride0, int64_t stride1)`

### Deep Dive: Memref Descriptor Expansion

**This is one of the most confusing aspects for MLIR beginners!**

When you write a function with memref arguments in high-level MLIR:
```mlir
func.func @gemm_8x16x32(%arg0: memref<8x32xf32>,  // 1 argument
                        %arg1: memref<32x16xf32>, // 1 argument  
                        %arg2: memref<8x16xf32>)  // 1 argument
```

After lowering through the `MemRefToLLVM` passes, each memref becomes **7 separate arguments**:

```cpp
// What you get in LLVM IR / C function signature
void gemm_8x16x32(
    // First memref (A: 8x32)
    float* allocated,    // Pointer to allocated memory
    float* aligned,      // Pointer to aligned data (usually same as allocated)
    int64_t offset,      // Base offset into buffer (usually 0)
    int64_t size0,       // Size of dimension 0 (rows = 8)
    int64_t size1,       // Size of dimension 1 (cols = 32)
    int64_t stride0,     // Stride for dimension 0 (= 32, skip full row)
    int64_t stride1,     // Stride for dimension 1 (= 1, contiguous columns)
    
    // Second memref (B: 32x16) - 7 more arguments
    float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,
    
    // Third memref (C: 8x16) - 7 more arguments
    float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t
);
// Total: 3 memrefs × 7 fields = 21 arguments!
```

**Why does this happen?**

1. **Memref is a high-level abstraction** - it represents multi-dimensional arrays with layout information
2. **LLVM IR doesn't understand memrefs** - it only knows pointers and integers
3. **The lowering passes expand memrefs** into their concrete representation (the memref descriptor)

**Which passes do this?**
```cpp
// In lowering.cpp
pm.addPass(memref::createExpandStridedMetadataPass());  // Exposes stride/size metadata
pm.addPass(createFinalizeMemRefToLLVMConversionPass()); // Converts to LLVM types
```

**For row-major (C-style) arrays:**
- `stride[1] = 1` (elements contiguous within a row)
- `stride[0] = num_columns` (skip entire row to get to next row)
- `offset = 0` (start at beginning)
- `allocated = aligned` (no special alignment in our case)

**Example call from C++:**
```cpp
// Calling the JIT-compiled function with actual NumPy arrays
gemm_func(
    A, A, 0, 8, 32, 32, 1,      // memref<8x32xf32>
    //│  │  │  │  │   │   └─ stride[1] = 1 (contiguous columns)
    //│  │  │  │  │   └───── stride[0] = 32 (skip 32 to next row)
    //│  │  │  │  └────────── size[1] = 32 columns
    //│  │  │  └───────────── size[0] = 8 rows
    //│  │  └──────────────── offset = 0
    //│  └─────────────────── aligned pointer = A
    //└────────────────────── allocated pointer = A
    
    B, B, 0, 32, 16, 16, 1,     // memref<32x16xf32>
    C, C, 0, 8, 16, 16, 1       // memref<8x16xf32>
);
```

**Key takeaway:** When moving to dynamic shapes, you'll need to pass the actual runtime dimensions in these size fields!

---

## Common Patterns & Gotchas

### Pattern: Memref vs Tensor

```cpp
// Use memref for in-place operations (mutable)
auto memrefType = MemRefType::get({8, 32}, f32Type);

// Use tensor for functional style (immutable, requires bufferization)
auto tensorType = RankedTensorType::get({8, 32}, f32Type);
```

**Recommendation**: Start with memrefs - they're simpler and don't require bufferization passes.

### Pattern: Creating Constants

```cpp
// For scalar constants
auto constAttr = builder.getFloatAttr(f32Type, 42.0);
auto constOp = builder.create<arith::ConstantOp>(loc, constAttr);
Value constValue = constOp.getResult();

// For dense tensors/memrefs
auto denseAttr = DenseElementsAttr::get(...);
auto constOp = builder.create<arith::ConstantOp>(loc, denseAttr);
```

### Pattern: Error Handling

```cpp
// MLIR uses LogicalResult (not exceptions)
LogicalResult result = someOperation();
if (failed(result)) {
  llvm::errs() << "Operation failed\n";
  return failure();
}

// LLVM uses Expected<T> (sum type: value or error)
auto JIT = llvm::orc::LLJITBuilder().create();
if (!JIT) {
  llvm::errs() << "Error: " << JIT.takeError() << "\n";
  return;
}
// Use *JIT to get the value
```

### Gotcha: Insertion Point

```cpp
OpBuilder builder(&context);
// ALWAYS set insertion point before creating ops!
builder.setInsertionPointToStart(block);
// Now you can create ops
auto op = builder.create<SomeOp>(...);
```

### Gotcha: Dialect Loading

```cpp
// Must load dialects before using their types/operations
context.getOrLoadDialect<arith::ArithDialect>();
// Now you can use arith.constant, arith.addf, etc.
```

### Gotcha: LLVM Initialization

```cpp
// Must initialize LLVM targets before JIT compilation
llvm::InitializeNativeTarget();
llvm::InitializeNativeTargetAsmPrinter();
llvm::InitializeNativeTargetAsmParser();
// Only needs to happen once per process
```

---

## Quick Reference: Essential Includes

```cpp
// === IR Generation ===
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

// === Lowering ===
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/LinalgToLoops/LinalgToLoops.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"

// === Execution ===
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/TargetSelect.h"
```

---

## Summary: The Complete Workflow

```
1. Create MLIRContext
   ↓
2. Load dialects you'll use
   ↓
3. Use OpBuilder to generate high-level IR
   ↓
4. Apply PassManager with lowering passes
   ↓
5. Translate MLIR → LLVM IR
   ↓
6. Create LLJIT and add IR module
   ↓
7. Lookup function symbol
   ↓
8. Cast to function pointer and call
```

**Key takeaway**: MLIR is all about *progressive transformation*. Start high-level (declarative), gradually lower through intermediate representations, end with executable machine code.

Each phase can be developed and tested independently:
- Test IR generation: Print the module with `module.dump()`
- Test lowering: Print before/after each pass
- Test execution: Compare results against reference implementation (NumPy, etc.)

This modular approach makes debugging much easier than monolithic compilers!
