# Chapter 9 Tutorial: Custom MLIR Dialect with TableGen

**Learn-by-Doing: From Python Prototypes (Ch8) to Production C++ Dialects (Ch9)**

This tutorial provides a deep, operation-by-operation comparison between Chapter 8's Python approach and Chapter 9's TableGen/C++ approach for creating custom MLIR dialects.

> **Note**: This tutorial references the original nested directory structure (`include/NN/`, `lib/NN/`, `lib/Conversion/`, `python/`) for pedagogical comparison with Chapter 8. The current Chapter 9 has been simplified to use flat directories (`inc/`, `src/`) and a PyTorch-like `forward()` API. The concepts remain identical - only the file organization changed.

---

## Table of Contents

1. [Philosophy: Why Two Approaches?](#philosophy-why-two-approaches)
2. [Complete Walkthrough: The `add` Operation](#complete-walkthrough-the-add-operation)
3. [Complete Walkthrough: The `matmul` Operation](#complete-walkthrough-the-matmul-operation)
4. [Step-by-Step: Creating a Dialect](#step-by-step-creating-a-dialect)
5. [Common Patterns & Cheat Sheet](#common-patterns--cheat-sheet)

---

## Philosophy: Why Two Approaches?

**Chapter 8**: Rapid prototyping with Python
- Learn MLIR concepts without C++ complexity
- Iterate quickly on ideas
- Perfect for research and exploration

**Chapter 9**: Production-grade with TableGen
- Type safety and compile-time verification
- Industrial-strength compiler engineering
- Full MLIR optimization infrastructure

**Key Insight**: Chapter 8 teaches you **what** custom dialects are. Chapter 9 teaches you **how** production compilers are built.

---

## Complete Walkthrough: The `add` Operation

Let's trace how the `nn.add` operation is defined, implemented, and lowered in both chapters.

### Chapter 8: Python String-Based Approach

#### 1. **"Definition"** (Implicit)

**File**: None - operation exists only as text patterns

**Code**: Just write MLIR text with `nn.add`
```python
# python/lowering.py - No formal definition needed!
# If the MLIR parser accepts "nn.add", it exists
```

#### 2. **Implementation** (String Generation)

**File**: `ch.8.Custom-dialect/python/lowering.py`

**Code**:
```python
def lower_add(self, result_id: int, lhs_id: int, rhs_id: int, shape: List[int]) -> List[str]:
    """Lower nn.add to linalg.generic with arith.addf"""
    lines = []
    ind = self._indent()
    shape_str = self._shape_str(shape)
    memref_type = self._tensor_to_memref(shape)

    rank = len(shape)
    if rank == 1:
        indexing = "affine_map<(d0) -> (d0)>"
        iterator = '["parallel"]'
    elif rank == 2:
        indexing = "affine_map<(d0, d1) -> (d0, d1)>"
        iterator = '["parallel", "parallel"]'

    # Generate MLIR text as string
    lines.append(f"{ind}%{result_id} = memref.alloc() : {memref_type}")
    lines.append(f"{ind}linalg.generic {{")
    lines.append(f"{ind}  indexing_maps = [{indexing}, {indexing}, {indexing}],")
    lines.append(f"{ind}  iterator_types = {iterator}")
    lines.append(f"{ind}}} ins(%{lhs_id}, %{rhs_id} : {memref_type}, {memref_type})")
    lines.append(f"{ind}   outs(%{result_id} : {memref_type}) {{")
    lines.append(f"{ind}^bb0(%arg0: f32, %arg1: f32, %arg2: f32):")
    lines.append(f"{ind}  %sum = arith.addf %arg0, %arg1 : f32")
    lines.append(f"{ind}  linalg.yield %sum : f32")
    lines.append(f"{ind}}}")

    return lines
```

**What happens**:
1. Python function builds list of strings
2. Each string is a line of MLIR text
3. Strings concatenated to form complete MLIR module
4. Text sent to C++ to parse

**Generated MLIR** (for 1D, size 4):
```mlir
%2 = memref.alloc() : memref<4xf32>
linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>, 
                   affine_map<(d0) -> (d0)>,
                   affine_map<(d0) -> (d0)>],
  iterator_types = ["parallel"]
} ins(%0, %1 : memref<4xf32>, memref<4xf32>)
   outs(%2 : memref<4xf32>) {
^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
  %sum = arith.addf %arg0, %arg1 : f32
  linalg.yield %sum : f32
}
```

#### 3. **Lowering** (Already Done!)

**Key Point**: In Chapter 8, "lowering" happens **during string generation** in Python. The `nn.add` operation never exists as actual MLIR IR - we go directly from Python graph to standard dialects (linalg, arith).

**Flow**:
```
Python Graph Node (add)
         ↓
Python lower_add() function
         ↓
MLIR text string (linalg.generic + arith.addf)
         ↓
C++ Parser
         ↓
Standard Dialect IR (ready for LLVM lowering)
```

**Pros**:
- ✅ Simple: just string manipulation
- ✅ Fast iteration: edit Python, run immediately
- ✅ Easy to understand MLIR syntax

**Cons**:
- ❌ Error-prone: typos cause runtime failures
- ❌ No type checking until parse time
- ❌ Manual rank handling (if/else for 1D vs 2D)

---

### Chapter 9: TableGen/C++ Approach

#### 1. **Definition** (Formal TableGen)

**File**: `ch.9.TableGen-dialect/include/NN/NNOps.td`

**Code**:
```tablegen
def NN_AddOp : NN_Op<"add"> {
  let summary = "element-wise addition";
  let description = [{
    Performs element-wise addition of two memrefs: `output = lhs + rhs`

    Example:
    ```mlir
    nn.add %lhs, %rhs, %output : memref<4xf32>, memref<4xf32>, memref<4xf32>
    ```
  }];

  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);

  let assemblyFormat = "$lhs `,` $rhs `,` $output attr-dict `:` type($lhs) `,` type($rhs) `,` type($output)";
}
```

**What TableGen generates** (`build/x64-release/ch.9.TableGen-dialect/include/NN/NNOps.h.inc`):

```cpp
// Auto-generated: ~200 lines of C++ boilerplate!
class AddOp : public ::mlir::Op<AddOp, 
                                 ::mlir::OpTrait::ZeroResults,
                                 ::mlir::OpTrait::ZeroSuccessors,
                                 ::mlir::OpTrait::NOperands<3>::Impl,
                                 /* ... more traits ... */> {
public:
  using Op::Op;
  using Adaptor = AddOpAdaptor;

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nn.add");
  }

  // Accessor methods (auto-generated!)
  ::mlir::Value getLhs() { return getOperand(0); }
  ::mlir::Value getRhs() { return getOperand(1); }
  ::mlir::Value getOutput() { return getOperand(2); }

  // Parser/printer (auto-generated from assemblyFormat!)
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, 
                                     ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

  // Verification (auto-generated!)
  ::mlir::LogicalResult verify();

  // Builder methods (auto-generated!)
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::Value lhs,
                    ::mlir::Value rhs,
                    ::mlir::Value output);

  // ... ~150 more lines of boilerplate
};
```

**Key Point**: That ~50 lines of TableGen generated **~200 lines of production-grade C++ code** with:
- Type-safe accessors
- Automatic parsing/printing
- Verification logic
- Builder methods
- LLVM RTTI support

#### 2. **Implementation** (C++ Lowering Pattern)

**File**: `ch.9.TableGen-dialect/lib/Conversion/NNToStandard.cpp`

**Code**:
```cpp
struct NNAddOpLowering : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override {
    // Get operation metadata
    auto loc = op.getLoc();
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Build iterator types (all parallel for element-wise)
    SmallVector<utils::IteratorType> iteratorTypes(
        outputType.getRank(),
        utils::IteratorType::parallel);

    // Create linalg.generic DIRECTLY as IR (not text!)
    rewriter.create<linalg::GenericOp>(
        loc, 
        ValueRange{op.getLhs(), op.getRhs()},  // Inputs
        ValueRange{op.getOutput()},             // Outputs
        ArrayRef<AffineMap>{
            rewriter.getMultiDimIdentityMap(outputType.getRank()),
            rewriter.getMultiDimIdentityMap(outputType.getRank()),
            rewriter.getMultiDimIdentityMap(outputType.getRank())
        },
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // Build body: element-wise addition
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    // Remove original nn.add operation
    rewriter.eraseOp(op);
    return success();
  }
};
```

**What happens**:
1. Pattern matcher finds all `nn::AddOp` instances in IR
2. For each match, `matchAndRewrite` is called
3. Function builds **IR objects directly** (not text!)
4. Original op is erased, new ops inserted
5. MLIR verifier runs automatically

**Key Differences from Chapter 8**:

| Aspect | Chapter 8 | Chapter 9 |
|--------|-----------|-----------|
| **Input Type** | `result_id: int` | `AddOp op` (IR object) |
| **Get Operands** | `lhs_id, rhs_id` parameters | `op.getLhs()`, `op.getRhs()` methods |
| **Get Type Info** | `shape: List[int]` parameter | `op.getOutput().getType()` |
| **Output** | `List[str]` (MLIR text lines) | Direct IR manipulation |
| **Rank Handling** | `if rank == 1: ... elif rank == 2:` | `outputType.getRank()` (generic) |
| **Affine Maps** | String: `"affine_map<(d0) -> (d0)>"` | IR: `rewriter.getMultiDimIdentityMap(rank)` |
| **Error Detection** | Parse time (runtime) | Immediately (compile-time) |

#### 3. **Lowering** (Pattern Registration)

**File**: `ch.9.TableGen-dialect/lib/Conversion/NNToStandard.cpp`

**Code**:
```cpp
struct ConvertNNToStandardPass : public PassWrapper<ConvertNNToStandardPass, 
                                                     OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // Define what's legal/illegal
    ConversionTarget target(getContext());
    target.addIllegalDialect<NNDialect>();      // NN must be lowered
    target.addLegalDialect<arith::ArithDialect,  // These are OK
                          linalg::LinalgDialect,
                          memref::MemRefDialect,
                          func::FuncDialect>();

    // Register all lowering patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<NNAddOpLowering,      // ← Our add pattern
                 NNMulOpLowering,
                 NNMatMulOpLowering,
                 NNReLUOpLowering>(&getContext());

    // Apply patterns until all NN ops are gone
    if (failed(applyPartialConversion(getOperation(), target, 
                                       std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

// Make pass available
std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertNNToStandardPass() {
  return std::make_unique<ConvertNNToStandardPass>();
}
```

**Flow**:
```
nn::AddOp (IR node)
         ↓
Pattern matcher finds AddOp
         ↓
NNAddOpLowering::matchAndRewrite()
         ↓
Creates: linalg::GenericOp + arith::AddFOp (IR nodes)
         ↓
Erases: nn::AddOp
         ↓
MLIR Verifier runs
         ↓
Standard Dialect IR (ready for next passes)
```

**Pros**:
- ✅ Type-safe: compiler catches errors
- ✅ Generic: works for any rank automatically
- ✅ Zero overhead: no text parsing
- ✅ Verifier ensures correctness at each step
- ✅ IDE support: autocomplete, refactoring, debugging

**Cons**:
- ❌ More code: ~80 lines vs ~20 lines Python
- ❌ Requires C++ compilation
- ❌ Steeper learning curve

---

### Side-by-Side Comparison: nn.add Implementation

#### **Chapter 8: String Template**
```python
# Input: Python integers and list
def lower_add(self, result_id: int, lhs_id: int, rhs_id: int, shape: List[int]):
    # Build string
    return [
        f"  %{result_id} = memref.alloc() : memref<{shape[0]}xf32>",
        f"  linalg.generic {{",
        f"    indexing_maps = [affine_map<(d0) -> (d0)>, ...],",
        f"    iterator_types = [\"parallel\"]",
        f"  }} ins(%{lhs_id}, %{rhs_id} : ...) outs(%{result_id} : ...) {{",
        f"  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):",
        f"    %sum = arith.addf %arg0, %arg1 : f32",
        f"    linalg.yield %sum : f32",
        f"  }}"
    ]
    # Returns: List of strings
```

#### **Chapter 9: IR Construction**
```cpp
// Input: IR object
LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const {
    // Get IR info
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Build IR directly
    rewriter.create<linalg::GenericOp>(
        loc,
        ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{op.getOutput()},
        ArrayRef<AffineMap>{
            rewriter.getMultiDimIdentityMap(outputType.getRank()),
            rewriter.getMultiDimIdentityMap(outputType.getRank()),
            rewriter.getMultiDimIdentityMap(outputType.getRank())
        },
        SmallVector<utils::IteratorType>(outputType.getRank(), parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.eraseOp(op);
    return success();
    // Returns: LogicalResult (success/failure)
}
```

---

## Complete Walkthrough: The `matmul` Operation

Let's see a more complex operation to understand the differences better.

### Chapter 8: Python String-Based

**File**: `ch.8.Custom-dialect/python/lowering.py`

```python
def lower_matmul(self, result_id: int, lhs_id: int, rhs_id: int, 
                 lhs_shape: List[int], rhs_shape: List[int], result_shape: List[int]) -> List[str]:
    """Lower nn.matmul to linalg.matmul"""
    lines = []
    ind = self._indent()

    # Build type strings manually
    lhs_type = f"memref<{lhs_shape[0]}x{lhs_shape[1]}xf32>"
    rhs_type = f"memref<{rhs_shape[0]}x{rhs_shape[1]}xf32>"
    result_type = f"memref<{result_shape[0]}x{result_shape[1]}xf32>"

    # Generate MLIR text
    lines.append(f"{ind}%{result_id} = memref.alloc() : {result_type}")
    lines.append(f"{ind}linalg.fill ins(%cst_zero : f32) outs(%{result_id} : {result_type})")
    lines.append(f"{ind}linalg.matmul ins(%{lhs_id}, %{rhs_id} : {lhs_type}, {rhs_type})")
    lines.append(f"{ind}                outs(%{result_id} : {result_type})")

    return lines
```

**Generated MLIR** (for 2x3 @ 3x4):
```mlir
%3 = memref.alloc() : memref<2x4xf32>
linalg.fill ins(%cst_zero : f32) outs(%3 : memref<2x4xf32>)
linalg.matmul ins(%1, %2 : memref<2x3xf32>, memref<3x4xf32>)
              outs(%3 : memref<2x4xf32>)
```

**Challenges**:
- Must manually format shape strings: `f"{lhs_shape[0]}x{lhs_shape[1]}"`
- Need three separate shape parameters
- Easy to make mistakes in string formatting
- No validation until C++ parses it

---

### Chapter 9: C++ Pattern

**File**: `ch.9.TableGen-dialect/lib/Conversion/NNToStandard.cpp`

```cpp
struct NNMatMulOpLowering : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern<MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Create zero constant for initialization
    auto zeroAttr = rewriter.getFloatAttr(outputType.getElementType(), 0.0);
    auto zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);

    // Fill output with zeros
    rewriter.create<linalg::FillOp>(
        loc, ValueRange{zero}, ValueRange{op.getOutput()});

    // Create linalg.matmul
    rewriter.create<linalg::MatmulOp>(
        loc, ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{op.getOutput()});

    rewriter.eraseOp(op);
    return success();
  }
};
```

**Key Differences**:

| Aspect | Chapter 8 | Chapter 9 |
|--------|-----------|-----------|
| **Shape Info** | 3 separate `List[int]` parameters | `op.getOutput().getType()` |
| **Type Strings** | Manual: `f"memref<{M}x{N}xf32>"` | Automatic: `outputType` |
| **Zero Constant** | Pre-defined: `%cst_zero` | Created on-demand: `rewriter.getFloatAttr()` |
| **Fill Operation** | Text: `"linalg.fill ins(...)"` | IR: `rewriter.create<linalg::FillOp>()` |
| **MatMul** | Text: `"linalg.matmul ins(...)"` | IR: `rewriter.create<linalg::MatmulOp>()` |
| **Error Risk** | High (string typos, format errors) | Low (compile-time type checking) |

**Advantages of Chapter 9**:
1. **No manual shape formatting**: types flow from IR
2. **Automatic verification**: MLIR checks types match
3. **Refactoring-safe**: rename operations, compiler updates all uses
4. **Debuggable**: breakpoints, inspect IR objects

---

## Step-by-Step: Creating a Dialect (Chapter 9)

This is your roadmap for understanding Chapter 9's code organization.

### Phase 1: Define Operations (TableGen)

**Files**: `include/NN/NNDialect.td`, `include/NN/NNOps.td`

**Reading Order**:

1. **Dialect definition** (`NNDialect.td`):
   ```tablegen
   def NN_Dialect : Dialect {
     let name = "nn";                    // Prefix in MLIR: "nn.add"
     let cppNamespace = "::mlir::nn";   // C++ namespace
   }
   ```

2. **Base operation class** (`NNOps.td`):
   ```tablegen
   class NN_Op<string mnemonic, list<Trait> traits = []> 
       : Op<NN_Dialect, mnemonic, traits>;
   ```

3. **Each operation** (`NNOps.td`):
   ```tablegen
   def NN_AddOp : NN_Op<"add"> {
     let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);
     let assemblyFormat = "...";
   }
   ```

**Build**: CMake runs `mlir-tblgen` to generate C++ code

---

### Phase 2: Register Dialect (C++)

**Files**: `lib/NN/NNDialect.cpp`, `include/NN/NNDialect.h`

**Reading Order**:

1. **Header includes generated code**:
   ```cpp
   #include "NN/NNOpsDialect.h.inc"  // Generated
   ```

2. **Implementation initializes**:
   ```cpp
   void NNDialect::initialize() {
     addOperations<
   #define GET_OP_LIST
   #include "NN/NNOps.cpp.inc"  // Generated op list
     >();
   }
   ```

3. **Bindings register in context**:
   ```cpp
   context.getOrLoadDialect<nn::NNDialect>();
   ```

---

### Phase 3: Implement Lowering (C++)

**File**: `lib/Conversion/NNToStandard.cpp`

**Reading Order**:

1. **Pattern for each operation**:
   ```cpp
   struct NNAddOpLowering : public OpRewritePattern<AddOp> {
     LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override {
       // Build new IR to replace this op
     }
   };
   ```

2. **Pass that applies patterns**:
   ```cpp
   struct ConvertNNToStandardPass : public PassWrapper<...> {
     void runOnOperation() override {
       // Define conversion target
       // Register patterns
       // Apply conversion
     }
   };
   ```

3. **Factory function**:
   ```cpp
   std::unique_ptr<OperationPass<ModuleOp>> createConvertNNToStandardPass();
   ```

---

### Phase 4: Build Integration (CMake)

**File**: `CMakeLists.txt`

**Key Steps**:

1. **Run TableGen**:
   ```cmake
   mlir_tablegen(include/NN/NNOps.h.inc -gen-op-decls)
   mlir_tablegen(include/NN/NNOps.cpp.inc -gen-op-defs)
   ```

2. **Compile C++**:
   ```cmake
   add_library(NNDialect
     lib/NN/NNDialect.cpp
     lib/Conversion/NNToStandard.cpp
   )
   ```

3. **Build Python module**:
   ```cmake
   pybind11_add_module(ch9 python/bindings.cpp)
   ```

---

### Phase 5: Python Bindings (pybind11)

**File**: `python/bindings.cpp`

**Reading Order**:

1. **Compiler class registers dialects**:
   ```cpp
   context.getOrLoadDialect<nn::NNDialect>();
   context.getOrLoadDialect<linalg::LinalgDialect>();
   ```

2. **Parse with dialect support**:
   ```cpp
   parseSourceString<ModuleOp>(mlir_text, &context);
   // Parser now understands nn.add!
   ```

3. **Apply custom pass**:
   ```cpp
   pm.addPass(createConvertNNToStandardPass());
   ```

4. **Expose to Python**:
   ```cpp
   PYBIND11_MODULE(ch9, m) {
     m.def("execute", &execute);
   }
   ```

---

## Common Patterns & Cheat Sheet

### Pattern 1: Element-wise Binary Operations

**Problem**: Operations like add, mul, sub, div with pattern `out[i] = f(a[i], b[i])`

**TableGen**:
```tablegen
def NN_MyOp : NN_Op<"myop"> {
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);
  let assemblyFormat = "$lhs `,` $rhs `,` $output attr-dict `:` type($lhs) `,` type($rhs) `,` type($output)";
}
```

**C++ Pattern**:
```cpp
struct NNMyOpLowering : public OpRewritePattern<MyOp> {
  LogicalResult matchAndRewrite(MyOp op, PatternRewriter &rewriter) const override {
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    rewriter.create<linalg::GenericOp>(
        loc,
        ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{op.getOutput()},
        ArrayRef<AffineMap>{
            rewriter.getMultiDimIdentityMap(outputType.getRank()),
            rewriter.getMultiDimIdentityMap(outputType.getRank()),
            rewriter.getMultiDimIdentityMap(outputType.getRank())
        },
        SmallVector<utils::IteratorType>(outputType.getRank(), utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::MyArithOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, result);
        });

    rewriter.eraseOp(op);
    return success();
  }
};
```

**Used for**: `nn.add`, `nn.mul`, `nn.div`, `nn.sub`

---

### Pattern 2: Unary Operations

**Problem**: Operations like relu, sigmoid with pattern `out[i] = f(in[i])`

**TableGen**:
```tablegen
def NN_MyUnaryOp : NN_Op<"myunary"> {
  let arguments = (ins AnyMemRef:$input, AnyMemRef:$output);
  let assemblyFormat = "$input `,` $output attr-dict `:` type($input) `,` type($output)";
}
```

**C++ Pattern**:
```cpp
struct NNMyUnaryOpLowering : public OpRewritePattern<MyUnaryOp> {
  LogicalResult matchAndRewrite(MyUnaryOp op, PatternRewriter &rewriter) const override {
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    rewriter.create<linalg::GenericOp>(
        loc,
        ValueRange{op.getInput()},
        ValueRange{op.getOutput()},
        ArrayRef<AffineMap>{
            rewriter.getMultiDimIdentityMap(outputType.getRank()),
            rewriter.getMultiDimIdentityMap(outputType.getRank())
        },
        SmallVector<utils::IteratorType>(outputType.getRank(), utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<SomeOp>(loc, args[0]);
          b.create<linalg::YieldOp>(loc, result);
        });

    rewriter.eraseOp(op);
    return success();
  }
};
```

**Used for**: `nn.relu`, `nn.sigmoid`, `nn.tanh`

---

### Pattern 3: Named Linalg Operations

**Problem**: Standard linear algebra operations like matmul, conv

**TableGen**:
```tablegen
def NN_MatMulOp : NN_Op<"matmul"> {
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);
  let assemblyFormat = "$lhs `,` $rhs `,` $output attr-dict `:` type($lhs) `,` type($rhs) `,` type($output)";
}
```

**C++ Pattern**:
```cpp
struct NNMatMulOpLowering : public OpRewritePattern<MatMulOp> {
  LogicalResult matchAndRewrite(MatMulOp op, PatternRewriter &rewriter) const override {
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Initialize output to zero
    auto zeroAttr = rewriter.getFloatAttr(outputType.getElementType(), 0.0);
    auto zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{op.getOutput()});

    // Create named linalg operation
    rewriter.create<linalg::MatmulOp>(
        loc, ValueRange{op.getLhs(), op.getRhs()}, ValueRange{op.getOutput()});

    rewriter.eraseOp(op);
    return success();
  }
};
```

**Used for**: `nn.matmul`, `nn.conv2d`, `nn.batchmatmul`

---

### Cheat Sheet: Adding a New Operation

**Step 1**: Define in TableGen (`NNOps.td`)
```tablegen
def NN_NewOp : NN_Op<"newop"> {
  let arguments = (ins ...);
  let assemblyFormat = "...";
}
```

**Step 2**: Rebuild (TableGen auto-generates C++)
```bash
cmake --build build/x64-release --target ch9
```

**Step 3**: Add lowering pattern (`NNToStandard.cpp`)
```cpp
struct NNNewOpLowering : public OpRewritePattern<NewOp> {
  LogicalResult matchAndRewrite(NewOp op, PatternRewriter &rewriter) const override {
    // ... implement lowering
  }
};
```

**Step 4**: Register in pass
```cpp
patterns.add<NNNewOpLowering>(&getContext());
```

**Step 5**: Test
```python
mlir_code = """
  func.func @test(%a: memref<4xf32>, %b: memref<4xf32>) {
    nn.newop %a, %b : memref<4xf32>, memref<4xf32>
    return
  }
"""
result = ch9.execute(mlir_code, "test", [a, b], (4,))
```

---

### Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `unknown operation 'nn.add'` | Dialect not registered | Add `context.getOrLoadDialect<NNDialect>()` in compiler |
| `type of operand #0 is not buildable` | Assembly format missing type | Add `type($operand)` to format string |
| `op was not bufferized` | Using tensor types with memref pipeline | Change to memref in TableGen definition |
| `pass failed to preserve dominance` | Invalid IR after rewrite | Use `rewriter` methods, not direct `op->erase()` |

---

## Summary: Chapter 8 vs Chapter 9

### What Each Chapter Teaches

**Chapter 8**: 
- **Concept**: What are custom dialects and why?
- **Skills**: MLIR text generation, graph IR representation
- **Outcome**: Ability to prototype quickly

**Chapter 9**:
- **Concept**: How do production compilers work?
- **Skills**: TableGen, pattern rewriting, C++ MLIR API
- **Outcome**: Ability to build industrial-strength compilers

### Key Technical Differences

| Aspect | Chapter 8 | Chapter 9 |
|--------|-----------|-----------|
| **Dialect Existence** | Implicit (text pattern) | Explicit (registered in MLIR) |
| **Type Checking** | Runtime (parse time) | Compile-time |
| **Lowering Location** | Python (string generation) | C++ (IR rewriting) |
| **Error Detection** | Late (at parse) | Early (at compile) |
| **Code Volume** | ~30 lines Python per op | ~50 lines TableGen + ~80 lines C++ per op |
| **Generated Code** | 0 lines | ~200 lines per op |
| **Maintenance** | Manual string updates | Automatic from TableGen |
| **Performance** | Parse overhead (~1-5ms) | Zero overhead |
| **Debugging** | Print strings | MLIR verifier + C++ debugger |

### Progression Path

1. **Start with Chapter 8** to learn:
   - MLIR syntax and structure
   - What custom dialects enable
   - Basic lowering concepts
   - Fast prototyping workflow

2. **Graduate to Chapter 9** when you need:
   - Type safety and verification
   - Production-grade maintainability
   - Complex multi-pass transformations
   - Performance (no parsing overhead)
   - Team collaboration (compile-time errors)

3. **Use Both** in real projects:
   - Chapter 8 for rapid prototyping
   - Chapter 9 for production implementation
   - Iterate quickly, then formalize

---

## Conclusion

Both approaches have their place:

- **Chapter 8** is your **prototyping toolkit** - fast, flexible, educational
- **Chapter 9** is your **production framework** - robust, verifiable, scalable

Real-world MLIR projects often start with Chapter 8's approach (quick exploration) then migrate to Chapter 9's approach (production deployment). Now you understand both workflows and can choose the right tool for your needs!