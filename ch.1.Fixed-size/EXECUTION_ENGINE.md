# ExecutionEngine and LLJIT

This project has been updated to use **mlir::ExecutionEngine** (MLIR's official JIT API) instead of the lower-level LLJIT.

## What Changed

### Before (LLJIT - manual control)
```cpp
// Manual LLVM IR translation
auto llvmModule = translateModuleToLLVMIR(*mlirModule, *llvmContext);

// Manual optimization
auto optPipeline = makeOptimizingTransformer(3, 0, nullptr);
optPipeline(llvmModule.get());

// Manual LLJIT creation
auto JIT = llvm::orc::LLJITBuilder().create();
auto TSM = llvm::orc::ThreadSafeModule(std::move(llvmModule), std::move(llvmContext));
(*JIT)->addIRModule(std::move(TSM));

// Manual lookup
auto Sym = (*JIT)->lookup("gemm_8x16x32");
auto* gemm_func = Sym->toPtr<FnPtr>();
```

**~100 lines of boilerplate code**

### After (ExecutionEngine - official API)
```cpp
// Register translations (required once)
registerBuiltinDialectTranslation(*mlirModule->getContext());
registerLLVMDialectTranslation(*mlirModule->getContext());

// Create ExecutionEngine with options
mlir::ExecutionEngineOptions options;
options.transformer = mlir::makeOptimizingTransformer(3, 0, nullptr);
auto maybeEngine = mlir::ExecutionEngine::create(*mlirModule, options);
auto engine = std::move(*maybeEngine);

// Lookup and call
auto expectedFPtr = engine->lookup("gemm_8x16x32");
auto* gemm_func = reinterpret_cast<FnPtr>(*expectedFPtr);
```

**~40 lines - much simpler!**

## Key Differences

| Aspect | LLJIT (old) | ExecutionEngine (new) |
|--------|-------------|----------------------|
| **API Level** | Low-level LLVM ORC | High-level MLIR wrapper |
| **LLVM IR Translation** | Manual (`translateModuleToLLVMIR`) | Automatic (internal) |
| **Setup Complexity** | ~100 lines | ~40 lines |
| **Error Handling** | Manual at each step | Simplified with Expected<> |
| **Optimization** | Manual pipeline application | Configured via options |
| **Recommended By** | LLVM project (for custom needs) | MLIR project (standard path) |

## Critical Requirements for ExecutionEngine

1. **Register Dialect Translations** (before creating ExecutionEngine):
   ```cpp
   registerBuiltinDialectTranslation(*mlirModule->getContext());
   registerLLVMDialectTranslation(*mlirModule->getContext());
   ```
   Without this, you'll get: `error: cannot be converted to LLVM IR: missing LLVMTranslationDialectInterface`

2. **Use ExecutionEngineOptions** for configuration:
   ```cpp
   mlir::ExecutionEngineOptions options;
   options.transformer = mlir::makeOptimizingTransformer(optLevel, sizeLevel, nullptr);
   ```

3. **Use lookup()** (not invoke() or lookupPacked()) for raw function pointers:
   ```cpp
   auto expectedFPtr = engine->lookup("function_name");
   auto* fn = reinterpret_cast<FnPtr>(*expectedFPtr);
   ```

## Benefits

✅ **Simpler code** - 60% less boilerplate  
✅ **Official MLIR API** - recommended path forward  
✅ **Automatic IR translation** - handles conversion internally  
✅ **Better error messages** - integrated with MLIR diagnostics  
✅ **Future-proof** - maintained by MLIR team  

## Performance

Identical to LLJIT - ExecutionEngine is just a wrapper around LLJIT with convenience features. Zero overhead.

## Migration Guide (for other chapters)

1. Replace includes:
   - Remove: `#include "mlir/Target/LLVMIR/Export.h"`
   - Remove: `#include "llvm/ExecutionEngine/Orc/LLJIT.h"`
   - Add: `#include "mlir/ExecutionEngine/ExecutionEngine.h"`

2. Add translation registration (before ExecutionEngine::create):
   ```cpp
   registerBuiltinDialectTranslation(*mlirModule->getContext());
   registerLLVMDialectTranslation(*mlirModule->getContext());
   ```

3. Replace LLJIT setup with ExecutionEngine:
   ```cpp
   mlir::ExecutionEngineOptions options;
   options.transformer = mlir::makeOptimizingTransformer(3, 0, nullptr);
   auto maybeEngine = mlir::ExecutionEngine::create(*mlirModule, options);
   auto engine = std::move(*maybeEngine);
   ```

4. Replace lookup:
   ```cpp
   auto expectedFPtr = engine->lookup("function_name");
   auto* fn = reinterpret_cast<FnPtr>(*expectedFPtr);
   ```

That's it! Much simpler than LLJIT.

## Important: Lifetime Management and Destruction Order

When using ExecutionEngine, **be careful about object lifetime and destruction order**, especially with LLVM resources.

### Safe Patterns ✅

**1. Local Scope (Chapters 1, 2, 5, 6)**
```cpp
void executeGemm(...) {
    // Create ExecutionEngine in local scope
    auto engine = std::move(*maybeEngine);
    auto* fn = reinterpret_cast<FnPtr>(*expectedFPtr);
    fn(...);
    // ExecutionEngine destroyed here - safe because LLVM is still active
}
```
✅ **Safe**: Destroyed during normal execution while Python/LLVM resources are valid.

**2. Class Member (Chapter 7)**
```cpp
class JITCompiler {
    struct Impl {
        std::vector<std::unique_ptr<mlir::ExecutionEngine>> engines;
    };
    std::unique_ptr<Impl> pImpl;
public:
    ~JITCompiler() { pImpl.reset(); }  // Destroyed during Python cleanup
};
```
✅ **Safe**: Python (via pybind11) owns the object and destroys it before LLVM's static destruction.

### Dangerous Pattern ❌ → Fixed ✅

**3. Global Static with unique_ptr (Chapters 3, 4 - original)**
```cpp
// ❌ DANGEROUS - causes segfault!
struct GlobalJITCache {
    std::unique_ptr<mlir::ExecutionEngine> engine;
} gGemmJIT;  // Global static variable
```

**Problem**: Destruction order during program exit:
1. Python exits and cleans up modules
2. LLVM performs static destruction of global resources
3. **Then** global `gGemmJIT` destructor runs
4. Tries to destroy `ExecutionEngine` → accesses already-freed LLVM memory → **SEGFAULT!** 💥

**Fix: Use raw pointer instead**
```cpp
// ✅ SAFE - intentional memory leak on exit
struct GlobalJITCache {
    mlir::ExecutionEngine* engine = nullptr;  // Raw pointer, no automatic destruction
    GemmFnPtr funcPtr = nullptr;
    bool isCompiled = false;
} gGemmJIT;

// In compilation function
auto engine = std::move(*maybeEngine);
gGemmJIT.engine = engine.release();  // Transfer ownership, no automatic cleanup
```

✅ **Safe**: No destructor calls during static destruction. Memory leak on exit is harmless (OS reclaims all process memory).

### Why This Happens

The issue stems from **static initialization/destruction order** (the "static initialization order fiasco"):

| Phase | Order | Status |
|-------|-------|--------|
| Program Start | → LLVM globals initialize | ✅ |
| Program Start | → Your globals initialize | ✅ |
| Program Exit | ← Python cleanup | ✅ |
| Program Exit | ← LLVM globals destruct | ✅ |
| Program Exit | ← Your globals destruct | ❌ LLVM already gone! |

When your global's destructor runs, LLVM's global state is already destroyed, causing crashes.

### Best Practices

1. **Prefer local scope** - Destroy ExecutionEngine during normal execution
2. **Use class members** - Let Python manage lifetime via pybind11
3. **Avoid global unique_ptr** - If you must use globals, use raw pointers
4. **Document intentional leaks** - Explain why memory isn't freed on exit

### Summary Table

| Storage Type | Chapters | Destruction Timing | Result |
|-------------|----------|-------------------|---------|
| Local `unique_ptr` | 1, 2, 5, 6 | During function execution | ✅ Safe |
| Class member `unique_ptr` | 7, 8 | Python cleanup (early) | ✅ Safe |
| Global `unique_ptr` | 3, 4 (old) | Static destruction (late) | ❌ Segfault |
| Global raw pointer | 3, 4 (fixed) | Never destroyed | ✅ Safe |

**Key Lesson**: When working with LLVM/MLIR, avoid global objects with complex destructors. Use local scope or Python-managed lifetimes instead.