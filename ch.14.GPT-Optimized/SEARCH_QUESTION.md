# How to programmatically build Transform Dialect sequences in C++ (LLVM 21)?

## Context

I'm trying to build MLIR Transform Dialect sequences **programmatically in C++** (not textual `.mlir` files) using LLVM 21. I want to construct transform operations that apply Linalg optimization patterns (generalization, fusion, CSE) to a payload module.

**Environment:**
- LLVM version: 21.1.0
- OS: Ubuntu (WSL2)
- Libraries linked: `MLIRTransformDialect`, `MLIRLinalgTransformOps`, `MLIRTransformDialectTransforms`
- Headers: `mlir/Dialect/Transform/IR/TransformDialect.h`, `mlir/Dialect/Transform/IR/TransformOps.h`, `mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h`

## Goal

Programmatically build a transform sequence equivalent to this textual IR:

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.apply_patterns to %root {
      transform.apply_patterns.linalg.generalize
      transform.apply_patterns.cse
    }
    transform.yield
  }
}
```

Then apply it to a payload `ModuleOp` using the transform interpreter.

## Problem 1: NamedSequenceOp Builder Signature

### What I tried:

```cpp
MLIRContext *ctx = module.getContext();
OpBuilder builder(ctx);
Location loc = builder.getUnknownLoc();

auto transformModule = builder.create<ModuleOp>(loc);
builder.setInsertionPointToStart(transformModule.getBody());

auto rootType = transform::AnyOpType::get(ctx);

builder.create<transform::NamedSequenceOp>(
    loc, 
    "__transform_main",
    rootType,    // Entry block argument type
    TypeRange{}, // No return values
    [&](OpBuilder &b, Location loc, Type root, TypeRange results) {
        Value rootHandle = b.getBlock()->getArgument(0);
        // ... build transforms here ...
        b.create<transform::YieldOp>(loc);
    });
```

### Error:

```
error: no matching function for call to 'build'
note: candidate function not viable: no known conversion from 'const char[17]' to '::mlir::TypeRange' for 3rd argument
note: candidate function not viable: requires at most 5 arguments, but 6 were provided
note: candidate function not viable: requires 7 arguments, but 6 were provided
note: candidate function not viable: requires 8 arguments, but 6 were provided
```

### Question 1A:
What is the correct signature for `NamedSequenceOp::build()` in LLVM 21? The error suggests multiple overloads, but none match. From `TransformOps.h.inc` line 7370-7385, I see signatures like:

```cpp
static void build(::mlir::OpBuilder &odsBuilder, 
                  ::mlir::OperationState &odsState, 
                  StringRef symName,
                  TypeAttr function_type,  // Not Type or TypeRange!
                  ...);
```

Do I need to create a `TypeAttr` wrapping a `FunctionType` instead of passing `Type` directly?

### Question 1B:
How do I use the lambda-based `SequenceBodyBuilderFn` parameter correctly? What's the expected signature of the lambda?

## Problem 2: SequenceOp Alternative Approach

### What I tried:

```cpp
auto sequenceOp = builder.create<transform::SequenceOp>(
    loc,
    TypeRange{},  // No results
    transform::FailurePropagationMode::Propagate,
    module,       // The payload ModuleOp
    [&](OpBuilder &b, Location loc, Value rootHandle) {
        // ... build transforms ...
        b.create<transform::YieldOp>(loc);
    });
```

### Error:

```
error: no matching function for call to 'build'
note: candidate function not viable: no known conversion from 'mlir::ModuleOp' to '::mlir::Value' for 5th argument
note: candidate function not viable: no known conversion from 'mlir::ModuleOp' to '::mlir::Type' for 5th argument
```

### Question 2:
`SequenceOp` expects a `Value` (transform handle) or `Type`, not a direct `ModuleOp`. How do I create a handle to my payload module? Is there a `transform::AnyOpHandle` creation function?

## Problem 3: ApplyPatternsOp with Nested Patterns

### What I tried:

```cpp
auto applyPattern = b.create<transform::ApplyPatternsOp>(
    loc, TypeRange{}, rootHandle);
{
  OpBuilder::InsertionGuard pGuard(b);
  Block *patternsBlock = b.createBlock(&applyPattern.getPatterns());
  b.setInsertionPointToStart(patternsBlock);
  
  // Insert pattern ops
  b.create<linalg::transform::ApplyGeneralizePatternsOp>(loc);
}
```

### Questions 3A:
Is this the correct way to build the nested `patterns` region for `ApplyPatternsOp`?

### Question 3B:
Do the pattern ops inside (`ApplyGeneralizePatternsOp`, `ApplyCSEPatternsOp`) need any terminator, or does the parent region handle that?

## Problem 4: Transform Interpreter Execution

### What I tried:

```cpp
transform::TransformOptions options;
if (failed(transform::applyTransformNamedSequence(
        payloadModule, transformModule, "__transform_main", options))) {
  // error
}
```

This compiles, but only if I can successfully create the `NamedSequenceOp` first.

### Alternative attempt:

```cpp
transform::TransformState state(module.getRegion(), transformSeq);
transform::applyTransforms(module, state, transformSeq, options);
```

### Error:

```
error: no matching function for call to 'applyTransforms'
note: candidate function requires 4 arguments, but 4 were provided (parameter mismatch on 2nd argument)
```

### Question 4:
What's the correct way to execute a transform sequence on a payload module in LLVM 21?

## What I've Tried

1. **Reading headers**: Examined `TransformOps.h.inc`, `TransformDialect.h`, `LinalgTransformOps.h`
2. **TableGen definitions**: Checked `TransformOps.td` for operation definitions
3. **Online examples**: Found examples but they're for older LLVM versions (18-19) or use textual IR
4. **Documentation**: `book/draft/docs/Dialects/Transform.md` shows textual IR, not C++ construction

## What Works

- ✅ Transform Dialect loads successfully: `context.loadDialect<transform::TransformDialect>()`
- ✅ Libraries link correctly
- ✅ Can create transform types: `transform::AnyOpType::get(ctx)`
- ✅ Textual IR approach works (parsing `.mlir` files)

## What I Need

**Working C++ code example for LLVM 21 showing:**
1. How to create a `NamedSequenceOp` or `SequenceOp` programmatically
2. Correct builder signatures and parameter types
3. How to create transform handles to payload operations
4. How to build nested `ApplyPatternsOp` regions with pattern ops inside
5. How to execute the transform sequence via interpreter

**Or:**
- Pointer to LLVM 21 test cases that demonstrate programmatic transform sequence construction in C++
- Updated documentation for the C++ API (not textual IR)

## Why Not Use Textual IR?

I understand that writing transforms in `.mlir` files and parsing them is the standard approach. However, for educational purposes, I want to demonstrate programmatic construction to teach students:
1. How transform operations are structured
2. How to build MLIR IR programmatically
3. How transform handles and types work
4. How to compose transformations dynamically based on IR analysis

## Minimal Reproducible Example

```cpp
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

LogicalResult buildTransformSequence(ModuleOp payloadModule) {
  MLIRContext *ctx = payloadModule.getContext();
  OpBuilder builder(ctx);
  Location loc = builder.getUnknownLoc();

  // How to create this programmatically?
  // transform.named_sequence @__transform_main(%root: !transform.any_op) {
  //   transform.apply_patterns to %root {
  //     transform.apply_patterns.linalg.generalize
  //   }
  //   transform.yield
  // }

  // Attempt 1: NamedSequenceOp (doesn't compile)
  auto transformModule = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(transformModule.getBody());
  
  auto rootType = transform::AnyOpType::get(ctx);
  auto seq = builder.create<transform::NamedSequenceOp>(
      loc, "__transform_main",
      rootType,    // ❌ Type mismatch
      TypeRange{},
      [&](OpBuilder &b, Location loc, Type root, TypeRange results) {
        // Build body...
      });

  return success();
}
```

**Compilation error**: No matching function for `NamedSequenceOp::build()`.

---

**Any help with correct C++ API usage for programmatic Transform Dialect sequence construction in LLVM 21 would be greatly appreciated!**

**Platform where I can search:**
- LLVM Discourse forums
- Stack Overflow with tags: [mlir], [llvm], [transform-dialect]
- GitHub issues in llvm/llvm-project
- MLIR chat channels
