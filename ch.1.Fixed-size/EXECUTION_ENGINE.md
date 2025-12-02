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