# Chapter 9: Custom Dialect with TableGen

**Learn-by-Doing Tutorial: From Python Prototypes to Production-Grade MLIR Dialects**

This chapter demonstrates how to define a production-grade custom MLIR dialect using **TableGen** (ODS - Operation Definition Specification), the standard approach for MLIR dialect development in C++.

This README compares Chapter 8 (Python + libffi prototype) with Chapter 9 (C++ + TableGen production) to help you understand when and how to graduate from rapid prototyping to industrial-strength compiler engineering.

## What You'll Learn

- **TableGen/ODS**: Define operations declaratively using `.td` files
- **Dialect Registration**: C++ boilerplate for custom dialects
- **Lowering Patterns**: Convert custom ops to standard MLIR dialects
- **Pass Infrastructure**: Implement transformation passes
- **Production Patterns**: Best practices for MLIR dialect development

## NN Dialect Operations

The `nn` (Neural Network) dialect provides high-level memref-based operations:

| Operation | Syntax | Description |
|-----------|--------|-------------|
| `nn.add` | `nn.add %a, %b, %out : memref<...>, memref<...>, memref<...>` | Element-wise addition |
| `nn.mul` | `nn.mul %a, %b, %out : memref<...>, memref<...>, memref<...>` | Element-wise multiplication |
| `nn.matmul` | `nn.matmul %a, %b, %out : memref<...>, memref<...>, memref<...>` | Matrix multiplication |
| `nn.relu` | `nn.relu %x, %out : memref<...>, memref<...>` | ReLU activation |

**Note**: All operations use **output-parameter style** (memref-based) to avoid tensor bufferization complexity.

## Project Structure

```
ch.9.TableGen-dialect/
├── include/NN/
│   ├── NNDialect.td       # Dialect definition (TableGen)
│   ├── NNOps.td           # Operation definitions (TableGen)
│   ├── NNDialect.h        # C++ dialect header
│   ├── NNOps.h            # C++ ops header
│   └── NNToStandard.h     # Lowering pass header
├── lib/
│   ├── NN/
│   │   ├── NNDialect.cpp  # Dialect implementation
│   │   └── NNOps.cpp      # Op implementations
│   └── Conversion/
│       └── NNToStandard.cpp # Lowering patterns
├── python/
│   └── bindings.cpp       # Python bindings (pybind11)
├── CMakeLists.txt         # Build configuration with TableGen
└── test_jit.py            # Test suite
```

## TableGen Code Generation

When you build, TableGen generates C++ code from `.td` files:

```bash
# TableGen generates:
include/NN/NNOps.h.inc           # Op class declarations
include/NN/NNOps.cpp.inc         # Op class definitions
include/NN/NNOpsDialect.h.inc    # Dialect declarations
include/NN/NNOpsDialect.cpp.inc  # Dialect definitions
```

## Defining Operations with TableGen

Example from `NNOps.td`:

```tablegen
def NN_AddOp : NN_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "element-wise addition";
  let description = [{
    Performs element-wise addition of two tensors.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
```

This generates ~200 lines of C++ boilerplate automatically!

## Lowering Pattern Example

From `NNToStandard.cpp`:

```cpp
struct NNAddOpLowering : public OpRewritePattern<NNAddOp> {
  LogicalResult matchAndRewrite(NNAddOp op, PatternRewriter &rewriter) const {
    // Convert: nn.add -> linalg.generic with arith.addf
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(...);
    auto addOp = rewriter.create<linalg::GenericOp>(
        ...,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });
    rewriter.replaceOp(op, addOp.getResult(0));
    return success();
  }
};
```

## Building

The project uses TableGen to generate code during build:

```bash
cd build/x64-release
cmake --build . --target ch9 -j 8
```

CMake automatically:
1. Runs `mlir-tblgen` on `.td` files
2. Generates `.inc` files
3. Compiles C++ sources
4. Links Python module

## Usage

```python
import ch9
import numpy as np

# Write MLIR using NN dialect
mlir_code = """
module {
  func.func @compute(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, 
                      %arg2: tensor<4xf32>) {
    %0 = nn.add %arg0, %arg1 : tensor<4xf32>
    %1 = nn.mul %0, %arg1 : tensor<4xf32>
    %2 = nn.relu %1 : tensor<4xf32>
    linalg.generic {...} ins(%2 : tensor<4xf32>) outs(%arg2 : tensor<4xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    }
    return
  }
}
"""

# Execute (automatic lowering and compilation)
a = np.array([1, 2, 3, 4], dtype=np.float32)
b = np.array([2, 3, 4, 5], dtype=np.float32)
result = ch9.execute(mlir_code, "compute", [a, b], (4,))
```

## Compilation Pipeline

1. **Parse**: MLIR text with NN dialect → IR
2. **Lower NN → Standard**: `nn.add` → `linalg.generic` + `arith.addf`
3. **Bufferization**: `tensor` → `memref`
4. **Lower to LLVM**: Standard dialects → LLVM dialect
5. **JIT**: LLVM IR → native code → execute

## Testing

```bash
cd ch.9.TableGen-dialect
python3 test_jit.py
```

Expected output:
```
Chapter 9: Custom Dialect with TableGen
========================================

### Test 1: NN Add ###
✓ [1. 2. 3. 4.] + [5. 6. 7. 8.] = [ 6.  8. 10. 12.]

### Test 2: NN Mul ###
✓ [2. 3. 4. 5.] * [10. 10. 10. 10.] = [20. 30. 40. 50.]

### Test 3: NN MatMul ###
✓ MatMul: (2, 3) @ (3, 4) = (2, 4)
  Result: [1. 2. 3. 6.]

### Test 4: NN ReLU ###
✓ Input:  [-1.  2. -3.  4.]
  Output: [0. 2. 0. 4.]

All tests passed! ✓
```

## Key Concepts

### TableGen vs Hand-written C++

| Aspect | TableGen | Hand-written |
|--------|----------|--------------|
| Lines of Code | ~50 lines `.td` | ~500 lines C++ |
| Type Safety | Compile-time | Runtime |
| Maintainability | High (declarative) | Lower (imperative) |
| Boilerplate | Generated | Manual |
| Learning Curve | Medium | Steep |

### When to Use Custom Dialects

✅ **Use custom dialects when:**
- Domain-specific operations need different semantics
- Want to enable domain-specific optimizations
- Building a framework/compiler for specific domain
- Need clear separation between abstraction levels

❌ **Don't use custom dialects when:**
- Standard dialects (linalg, tensor) are sufficient
- Just composing existing operations
- Rapid prototyping (use Chapter 8's approach)

---

## 📚 Chapter 8 vs Chapter 9: The Complete Comparison

This section provides a deep dive into the architectural differences between prototyping (Chapter 8) and production (Chapter 9) approaches to custom MLIR dialects.

---

### 1. How Each Chapter Defines a Dialect

#### **Chapter 8: Implicit String-Based Dialect**

**Philosophy**: "If it parses, it's valid" - dialect exists only as MLIR text strings.

**Code Location**: `python/lowering.py`

```python
class MLIRLowering:
    def lower_add(self, node):
        # Just generate MLIR text with "nn.add" operation
        return f"""
            %{node.id} = nn.add %{node.inputs[0]}, %{node.inputs[1]} 
                : memref<{self._shape_str(node.shape)}xf32>
        """
```

**What happens**:
- No formal dialect registration
- Operations defined by text patterns
- Parser sees `nn.add` and accepts it (no validation)
- Works for prototyping, but no type checking or verification

**Pros**: 
- ✅ 10 minutes to add new operation (just edit string)
- ✅ Zero C++ code
- ✅ Perfect for rapid iteration

**Cons**:
- ❌ No compile-time error checking
- ❌ Typos cause runtime failures
- ❌ No IDE support (autocomplete, refactoring)

---

#### **Chapter 9: Formal TableGen Dialect**

**Philosophy**: "Define once, generate everywhere" - dialect is a first-class MLIR citizen.

**Code Location**: `include/NN/NNDialect.td` + `include/NN/NNOps.td`

```tablegen
// NNDialect.td - Defines the dialect namespace
def NN_Dialect : Dialect {
  let name = "nn";
  let cppNamespace = "::mlir::nn";
  let summary = "Neural Network operations dialect";
}

// NNOps.td - Defines each operation formally
def NN_AddOp : NN_Op<"add"> {
  let summary = "element-wise addition";
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);
  let assemblyFormat = "$lhs `,` $rhs `,` $output attr-dict `:` type($lhs) `,` type($rhs) `,` type($output)";
}
```

**What TableGen generates** (automatically at build time):

```cpp
// Generated: NNOps.h.inc (~200 lines per operation)
class AddOp : public Op<AddOp, OpTrait::ZeroResults, ...> {
public:
  Value getLhs() { return getOperand(0); }
  Value getRhs() { return getOperand(1); }
  Value getOutput() { return getOperand(2); }
  
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  // ... ~180 more lines of boilerplate
};
```

**Registration** (`lib/NN/NNDialect.cpp`):

```cpp
void NNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NN/NNOps.cpp.inc"
  >();
}
```

**Pros**:
- ✅ Compile-time type checking
- ✅ IDE autocomplete and refactoring
- ✅ MLIR verifier catches errors early
- ✅ ~1000 lines of boilerplate generated from ~50 lines `.td`

**Cons**:
- ❌ 2-3 hours to add new operation (learn TableGen syntax)
- ❌ Requires C++ build system
- ❌ Steeper learning curve

---

### 2. How Each Chapter Lowers the Dialect

#### **Chapter 8: Python String Generation**

**Location**: `python/lowering.py`

**Architecture**:
```
Python Graph → Python Lowering → MLIR Text → C++ Parser → Standard Dialects
```

**Code**:
```python
def lower_add(self, node):
    lhs = node.inputs[0]
    rhs = node.inputs[1]
    rank = len(node.shape)
    
    # Generate linalg.generic MLIR text string
    return f"""
        linalg.generic {{
          indexing_maps = [affine_map<(d0) -> (d0)>, 
                           affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        }} ins(%{lhs}, %{rhs} : memref<{node.shape[0]}xf32>, memref<{node.shape[0]}xf32>)
           outs(%{node.id}_out : memref<{node.shape[0]}xf32>) {{
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %sum = arith.addf %arg0, %arg1 : f32
          linalg.yield %sum : f32
        }}
    """
```

**Block Diagram**:
```
┌──────────────┐
│ nn.add %a, %b│  ← Python string
└──────┬───────┘
       │ Python lowering (string manipulation)
       ↓
┌─────────────────────────────────┐
│ linalg.generic {                │
│   indexing_maps = [...]         │  ← Generated MLIR text
│   iterator_types = ["parallel"] │
│ } ins(%a, %b) outs(%out) {      │
│   ^bb0(%0, %1, %2):             │
│     %sum = arith.addf %0, %1    │  ← arith dialect
│     linalg.yield %sum           │
│ }                               │
└────────┬────────────────────────┘
         │ C++ parser
         ↓
    [Standard MLIR IR]
```

**Pros**:
- ✅ Easy to understand (just string templates)
- ✅ Rapid iteration (no recompilation)
- ✅ Good for learning MLIR syntax

**Cons**:
- ❌ Error-prone (typos, syntax mistakes)
- ❌ No validation until C++ parser runs
- ❌ Hard to maintain complex transformations

---

#### **Chapter 9: C++ Pattern Rewriting**

**Location**: `lib/Conversion/NNToStandard.cpp`

**Architecture**:
```
NN Dialect IR → C++ Rewrite Patterns → Standard Dialects IR
```

**Code**:
```cpp
struct NNAddOpLowering : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Create linalg.generic DIRECTLY as IR (not text!)
    SmallVector<utils::IteratorType> iteratorTypes(
        outputType.getRank(),
        utils::IteratorType::parallel);

    rewriter.create<linalg::GenericOp>(
        loc, 
        ValueRange{op.getLhs(), op.getRhs()},  // inputs
        ValueRange{op.getOutput()},             // outputs
        ArrayRef<AffineMap>{
            rewriter.getMultiDimIdentityMap(outputType.getRank()),
            rewriter.getMultiDimIdentityMap(outputType.getRank()),
            rewriter.getMultiDimIdentityMap(outputType.getRank())
        },
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.eraseOp(op);  // Remove original nn.add
    return success();
  }
};
```

**Block Diagram**:
```
┌─────────────────┐
│ nn::AddOp       │  ← C++ object (IR node)
│  - lhs: Value   │
│  - rhs: Value   │
│  - output: Value│
└────────┬────────┘
         │ Pattern matcher recognizes AddOp
         ↓
┌─────────────────────────────────┐
│ NNAddOpLowering::matchAndRewrite│  ← C++ rewrite pattern
│   - Get operands via IR API     │
│   - Create linalg.generic IR    │
│   - Create arith.addf IR        │
│   - Erase original op           │
└────────┬────────────────────────┘
         │ Direct IR manipulation
         ↓
┌─────────────────────────────┐
│ linalg::GenericOp          │  ← C++ object (IR node)
│   inputs: [Value, Value]   │
│   outputs: [Value]         │
│   body:                    │
│     ↳ arith::AddFOp        │  ← Nested IR
│     ↳ linalg::YieldOp      │
└─────────────────────────────┘
```

**Key Differences from Chapter 8**:

| Aspect | Chapter 8 (Python) | Chapter 9 (C++)  |
|--------|--------------------|------------------|
| **Input** | Python `node` object | `AddOp` IR object |
| **Output** | MLIR text string | IR objects directly |
| **Validation** | Parser (runtime) | Verifier (immediate) |
| **Errors** | String parsing fails | Compile errors |
| **Performance** | Parse overhead | Zero overhead |

**Pros**:
- ✅ Type-safe IR construction
- ✅ Immediate verification
- ✅ Zero text parsing overhead
- ✅ Can analyze/transform IR programmatically

**Cons**:
- ❌ More verbose
- ❌ Requires understanding MLIR C++ API
- ❌ Slower iteration (recompile needed)

---

### 3. Key Architectural Differences

#### **Execution Pipeline Comparison**

**Chapter 8**:
```
┌─────────────┐
│ Python Code │
└──────┬──────┘
       │
       ↓
┌─────────────────┐
│ Graph Builder   │  ← Python dataclasses
│ (graph_builder) │     Track operations
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│ Python Lowering │  ← String generation
│ (lowering.py)   │     nn.add → linalg text
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│ MLIR Text       │  ← Huge multi-line string
└──────┬──────────┘
       │
       ↓ (cross language boundary)
┌─────────────────┐
│ C++ Parser      │  ← mlir::parseSourceString
│ (bindings.cpp)  │     Parse text → IR
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│ MLIR Compiler   │  ← Standard passes
│ (compiler.cpp)  │     linalg → loops → LLVM
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│ ExecutionEngine │  ← JIT compile & run
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│ NumPy Arrays    │  ← Results via libffi
└─────────────────┘
```

**Chapter 9**:
```
┌─────────────────┐
│ Python MLIR Text│  ← User writes MLIR directly
└──────┬──────────┘
       │
       ↓ (cross language boundary)
┌─────────────────┐
│ C++ Parser      │  ← mlir::parseSourceString
│ (bindings.cpp)  │     Parse text → IR
│                 │     (includes nn dialect!)
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│ NN Dialect IR   │  ← nn::AddOp, nn::MulOp, etc.
│ (registered)    │     First-class IR nodes
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│ NN→Standard Pass│  ← C++ rewrite patterns
│ (NNToStandard)  │     AddOp → linalg::GenericOp
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│ Standard IR     │  ← linalg, arith, memref ops
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│ More Passes     │  ← linalg → loops → LLVM
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│ ExecutionEngine │  ← JIT compile & run
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│ NumPy Arrays    │  ← Results via libffi
└─────────────────┘
```

**Key Insight**: Chapter 9's dialect is **part of the IR**, not a text preprocessing step!

---

#### **Type Safety Comparison**

**Chapter 8 - Runtime Errors**:
```python
# Python lowering (typo in operation name)
def lower_add(self, node):
    return f"%{node.id} = nn.adddd %{lhs}, %{rhs}"  # Typo: "adddd"
    
# ❌ Error only when C++ parser runs:
# "unknown operation 'nn.adddd'"
```

**Chapter 9 - Compile-Time Errors**:
```cpp
// C++ rewrite pattern (typo in method name)
LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const {
  Value sum = rewriter.create<arith::AddFFOp>(loc, lhs, rhs);  // Typo: "AddFFOp"
  
  // ✅ Compile error immediately:
  // error: no member named 'AddFFOp' in namespace 'mlir::arith'
}
```

---

### 4. Step-by-Step: Creating a Dialect (Chapter 9)

This is your roadmap for reading and understanding Chapter 9's code.

#### **Phase 1: Define the Dialect (TableGen)**

**Files to read**: `include/NN/NNDialect.td`, `include/NN/NNOps.td`

**Order of operations**:

1. **Define dialect namespace** (`NNDialect.td`):
   ```tablegen
   def NN_Dialect : Dialect {
     let name = "nn";                          // Dialect prefix in MLIR
     let cppNamespace = "::mlir::nn";         // C++ namespace
     let useDefaultAttributePrinterParser = 0; // We don't need custom types
     let useDefaultTypePrinterParser = 0;
   }
   ```

2. **Define base operation class** (`NNOps.td`):
   ```tablegen
   class NN_Op<string mnemonic, list<Trait> traits = []> 
       : Op<NN_Dialect, mnemonic, traits>;
   ```

3. **Define each operation** (`NNOps.td`):
   ```tablegen
   def NN_AddOp : NN_Op<"add"> {
     let summary = "element-wise addition";
     let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);
     let assemblyFormat = "$lhs `,` $rhs `,` $output attr-dict `:` type($lhs) `,` type($rhs) `,` type($output)";
   }
   ```

**What to understand**:
- `let arguments`: What operands the operation takes
- `let results`: What values the operation produces (none for us - we use output parameter)
- `let assemblyFormat`: How the operation is printed/parsed in MLIR text
- `ins`/`outs`: Input vs output operands
- `AnyMemRef`: Type constraint (accepts any memref type)

**Build step**: Run `mlir-tblgen` to generate C++ code:
```bash
mlir-tblgen -gen-op-defs NNOps.td -o NNOps.cpp.inc
mlir-tblgen -gen-op-decls NNOps.td -o NNOps.h.inc
```

---

#### **Phase 2: Register the Dialect (C++)**

**Files to read**: `lib/NN/NNDialect.cpp`, `include/NN/NNDialect.h`

**Order of operations**:

1. **Include generated code** (`NNDialect.h`):
   ```cpp
   #include "NN/NNOpsDialect.h.inc"  // Generated dialect declaration
   ```

2. **Initialize dialect** (`NNDialect.cpp`):
   ```cpp
   #include "NN/NNOpsDialect.cpp.inc"  // Generated dialect definition
   
   void NNDialect::initialize() {
     addOperations<
   #define GET_OP_LIST
   #include "NN/NNOps.cpp.inc"  // Generated op list
     >();
   }
   ```

3. **Register in context** (`python/bindings.cpp`):
   ```cpp
   MLIRContext context;
   context.getOrLoadDialect<nn::NNDialect>();  // Load nn dialect
   ```

**What happens**:
- MLIR learns about "nn" dialect
- Parser recognizes `nn.add`, `nn.mul`, etc.
- Operations become first-class IR nodes

---

#### **Phase 3: Implement Lowering Patterns (C++)**

**Files to read**: `lib/Conversion/NNToStandard.cpp`

**Order of operations**:

1. **Create pattern class**:
   ```cpp
   struct NNAddOpLowering : public OpRewritePattern<AddOp> {
     using OpRewritePattern<AddOp>::OpRewritePattern;
     
     LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override;
   };
   ```

2. **Implement rewrite logic**:
   ```cpp
   LogicalResult NNAddOpLowering::matchAndRewrite(
       AddOp op, PatternRewriter &rewriter) const {
     
     // Step 1: Get information from original op
     auto loc = op.getLoc();
     auto outputType = cast<MemRefType>(op.getOutput().getType());
     
     // Step 2: Create new operations to replace it
     rewriter.create<linalg::GenericOp>(
         loc,
         ValueRange{op.getLhs(), op.getRhs()},  // Inputs
         ValueRange{op.getOutput()},             // Outputs
         /* ... affine maps, iterator types ... */,
         [&](OpBuilder &b, Location loc, ValueRange args) {
           // Step 3: Build operation body
           Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
           b.create<linalg::YieldOp>(loc, sum);
         });
     
     // Step 4: Remove original operation
     rewriter.eraseOp(op);
     return success();
   }
   ```

3. **Register patterns in pass**:
   ```cpp
   struct ConvertNNToStandardPass : public PassWrapper<...> {
     void runOnOperation() override {
       ConversionTarget target(getContext());
       target.addIllegalDialect<NNDialect>();  // NN must be lowered
       target.addLegalDialect<LinalgDialect>();  // Linalg is OK
       
       RewritePatternSet patterns(&getContext());
       patterns.add<NNAddOpLowering,
                    NNMulOpLowering,
                    NNMatMulOpLowering,
                    NNReLUOpLowering>(&getContext());
       
       applyPartialConversion(getOperation(), target, std::move(patterns));
     }
   };
   ```

**What happens**:
- Pass manager applies patterns
- Each `nn.add` is matched by `NNAddOpLowering`
- Pattern replaces `nn.add` with `linalg.generic` + `arith.addf`
- MLIR verifier ensures correctness after each transformation

---

#### **Phase 4: Build System Integration (CMake)**

**File to read**: `CMakeLists.txt`

**Order of operations**:

1. **Run TableGen during build**:
   ```cmake
   set(LLVM_TARGET_DEFINITIONS include/NN/NNOps.td)
   mlir_tablegen(include/NN/NNOps.h.inc -gen-op-decls)
   mlir_tablegen(include/NN/NNOps.cpp.inc -gen-op-defs)
   mlir_tablegen(include/NN/NNOpsDialect.h.inc -gen-dialect-decls)
   mlir_tablegen(include/NN/NNOpsDialect.cpp.inc -gen-dialect-defs)
   add_public_tablegen_target(NNOpsIncGen)
   ```

2. **Compile C++ sources**:
   ```cmake
   add_library(NNDialect
     lib/NN/NNDialect.cpp
     lib/NN/NNOps.cpp
     lib/Conversion/NNToStandard.cpp
   )
   add_dependencies(NNDialect NNOpsIncGen)  # Wait for TableGen
   ```

3. **Build Python module**:
   ```cmake
   pybind11_add_module(ch9 python/bindings.cpp)
   target_link_libraries(ch9 PRIVATE NNDialect MLIRExecutionEngine ...)
   ```

---

#### **Phase 5: Python Bindings (pybind11)**

**File to read**: `python/bindings.cpp`

**Order of operations**:

1. **Create compiler class**:
   ```cpp
   class NNCompiler {
     MLIRContext context_;
     
     NNCompiler() {
       context_.getOrLoadDialect<nn::NNDialect>();  // Register our dialect
       context_.getOrLoadDialect<linalg::LinalgDialect>();
       // ... register all dialects we need
     }
   };
   ```

2. **Parse MLIR with dialect support**:
   ```cpp
   OwningOpRef<ModuleOp> parseMLIR(const std::string& mlir_text) {
     return parseSourceString<ModuleOp>(mlir_text, &context_);
     // Parser now understands nn.add, nn.mul, etc.!
   }
   ```

3. **Apply lowering passes**:
   ```cpp
   bool lowerToLLVM(ModuleOp module) {
     PassManager pm(&context_);
     pm.addPass(createConvertNNToStandardPass());  // Our custom pass!
     pm.addPass(createConvertLinalgToLoopsPass());
     pm.addPass(createConvertSCFToCFPass());
     // ... more passes
     return succeeded(pm.run(module));
   }
   ```

4. **Expose to Python**:
   ```cpp
   PYBIND11_MODULE(ch9, m) {
     m.def("execute", &execute, 
           "Compile and execute MLIR with NN dialect");
   }
   ```

---

### 5. Philosophy & Common Patterns

#### **When to Create a Custom Dialect**

**✅ Good reasons**:
- **Domain-specific semantics**: Operations that don't map 1:1 to existing dialects
  - Example: `nn.attention` encapsulates complex pattern (scaled dot-product + softmax)
- **Enable domain optimizations**: Fusion opportunities visible only at high level
  - Example: `nn.linear` + `nn.relu` → fused kernel
- **Progressive lowering**: Multiple abstraction levels
  - Example: `nn` → `linalg` → `vector` → `llvm`
- **Framework integration**: Clean API for framework operators
  - Example: PyTorch → Torch-MLIR dialect → linalg

**❌ Bad reasons**:
- **Just for organization**: Use existing dialects + namespaces
- **Simple composition**: Use `func.func` to compose existing ops
- **One-off prototype**: Chapter 8's approach is better

---

#### **Common Implementation Patterns**

##### **Pattern 1: Element-wise Operations**

**Problem**: Binary operations (add, mul, sub, div)

**Solution**: Lower to `linalg.generic` with parallel iteration

```cpp
// Pattern for: output[i] = f(lhs[i], rhs[i])
rewriter.create<linalg::GenericOp>(
    loc,
    ValueRange{lhs, rhs},      // Inputs
    ValueRange{output},         // Outputs
    ArrayRef<AffineMap>{        // All identity maps
        identityMap, identityMap, identityMap
    },
    SmallVector<utils::IteratorType>(rank, parallel),  // All parallel
    [&](OpBuilder &b, Location loc, ValueRange args) {
      Value result = b.create<YourArithOp>(loc, args[0], args[1]);
      b.create<linalg::YieldOp>(loc, result);
    });
```

**Used for**: `nn.add`, `nn.mul`, `nn.div`, `nn.sub`

---

##### **Pattern 2: Reduction Operations**

**Problem**: Operations that reduce dimensions (sum, max, softmax)

**Solution**: `linalg.generic` with reduction iterator

```cpp
// Pattern for: output[i] = reduce(input[i, :])
SmallVector<utils::IteratorType> iterTypes = {parallel, reduction};

rewriter.create<linalg::GenericOp>(
    loc,
    ValueRange{input},
    ValueRange{output},
    getReductionMaps(rank),  // Preserve some dims, reduce others
    iterTypes,
    [&](OpBuilder &b, Location loc, ValueRange args) {
      Value reduced = b.create<ReductionOp>(loc, args[0], args[1]);
      b.create<linalg::YieldOp>(loc, reduced);
    });
```

**Used for**: `nn.softmax`, layer norm, attention scores

---

##### **Pattern 3: Structured Matrix Operations**

**Problem**: Matrix multiplication, convolution

**Solution**: Use specialized linalg named operations

```cpp
// Pattern for: output = lhs @ rhs
rewriter.create<linalg::FillOp>(loc, zero, output);  // Initialize to zero
rewriter.create<linalg::MatmulOp>(
    loc,
    ValueRange{lhs, rhs},
    ValueRange{output});
```

**Used for**: `nn.matmul`, `nn.conv2d`, `nn.linear`

---

##### **Pattern 4: Multi-stage Lowering**

**Problem**: Complex operations need multiple passes

**Solution**: Introduce intermediate dialects

```
nn.linear (high-level)
    ↓ (first pass: decompose)
nn.matmul + nn.bias_add (medium-level)
    ↓ (second pass: lower)
linalg.matmul + linalg.generic (low-level)
```

**Code**:
```cpp
// Pass 1: Decompose
struct DecomposeLinearPattern : public OpRewritePattern<nn::LinearOp> {
  LogicalResult matchAndRewrite(nn::LinearOp op, PatternRewriter &rewriter) const {
    auto matmul = rewriter.create<nn::MatMulOp>(op.getInput(), op.getWeight());
    auto biasAdd = rewriter.create<nn::BiasAddOp>(matmul, op.getBias());
    rewriter.replaceOp(op, biasAdd);
    return success();
  }
};

// Pass 2: Lower matmul to linalg
struct LowerMatMulPattern : public OpRewritePattern<nn::MatMulOp> {
  LogicalResult matchAndRewrite(nn::MatMulOp op, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(...);
    return success();
  }
};
```

---

#### **Cheat Sheet: Common Tasks**

##### **Add a new operation**:

1. Edit `NNOps.td`:
   ```tablegen
   def NN_MyOp : NN_Op<"myop"> {
     let arguments = (ins ...);
     let results = (outs ...);
   }
   ```

2. Rebuild (TableGen regenerates code automatically)

3. Add lowering pattern in `NNToStandard.cpp`:
   ```cpp
   struct NNMyOpLowering : public OpRewritePattern<MyOp> { ... };
   ```

4. Register pattern in pass:
   ```cpp
   patterns.add<NNMyOpLowering>(&getContext());
   ```

5. Test in Python:
   ```python
   mlir_code = "... nn.myop ..."
   result = ch9.execute(mlir_code, "func_name", inputs, output_shape)
   ```

---

##### **Debug a pattern**:

```cpp
LogicalResult matchAndRewrite(MyOp op, PatternRewriter &rewriter) const {
  // 1. Print operation being matched
  op->dump();  // or: llvm::errs() << "Lowering: " << op << "\n";
  
  // 2. Check types
  auto type = op.getOperand(0).getType();
  if (!type.isa<MemRefType>()) {
    return op->emitError("expected memref type");
  }
  
  // 3. Verify new operation
  auto newOp = rewriter.create<SomeOp>(...);
  if (failed(newOp.verify())) {
    return failure();
  }
  
  return success();
}
```

---

##### **Common errors and fixes**:

| Error | Cause | Fix |
|-------|-------|-----|
| `unknown operation 'nn.add'` | Dialect not registered | Add `context.getOrLoadDialect<NNDialect>()` |
| `type of operand ... is not buildable` | TableGen assembly format incomplete | Specify all types: `type($lhs) , type($rhs)` |
| `op was not bufferized` | Tensor operations in memref pipeline | Use memref types from the start |
| `pass failed to preserve dominance` | Invalid IR after transformation | Check SSA form, verify all uses updated |

---

### 6. Summary Table: Chapter 8 vs Chapter 9

| Aspect | Chapter 8 (Python + libffi) | Chapter 9 (TableGen) |
|--------|----------------------------|----------------------|
| **Dialect Definition** | Implicit (text strings) | Explicit (TableGen .td files) |
| **Lines of Code** | ~30 lines Python | ~50 lines TableGen → ~1000 lines generated C++ |
| **Type Checking** | Runtime (parser) | Compile-time (C++ compiler) |
| **Lowering** | String templates (Python) | IR rewriting (C++ patterns) |
| **Error Detection** | Late (at parse time) | Early (at compile time) |
| **IDE Support** | None | Full (autocomplete, refactoring) |
| **Development Speed** | Fast (edit, run) | Slower (edit, rebuild, run) |
| **Maintainability** | Hard (string manipulation) | Easy (declarative, type-safe) |
| **Performance** | Parse overhead (~1-5ms) | Zero overhead |
| **Learning Curve** | Gentle (Python, MLIR text) | Steep (C++, TableGen, MLIR API) |
| **Use Case** | Rapid prototyping, research | Production compilers, frameworks |
| **Debugging** | Print MLIR strings | MLIR verifier, C++ debugger |
| **Optimization Potential** | Limited | Full MLIR optimization infrastructure |
| **When to Use** | Exploring ideas, learning | Building real compilers |

---

### 7. Key Takeaways

**Chapter 8 teaches you**: "What is a custom dialect and why do we need one?"
- Understand MLIR abstraction levels
- See how high-level ops lower to low-level ops
- Rapid prototyping workflow

**Chapter 9 teaches you**: "How do real-world MLIR projects work?"
- Production-grade dialect engineering
- TableGen automated code generation
- Type-safe IR transformation
- Industrial compiler patterns

**Progression path**:
1. Start with **Chapter 8** for fast prototyping and learning
2. Graduate to **Chapter 9** when you need:
   - Type safety and verification
   - Performance (no parse overhead)
   - Complex multi-pass transformations
   - Integration with large codebases

**Next steps** (Chapter 10+):
- GPU code generation
- Advanced optimizations (tiling, fusion)
- Attention mechanisms
- Full transformer implementation