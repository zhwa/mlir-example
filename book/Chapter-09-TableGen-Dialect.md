# Chapter 9: TableGen for Production Dialects

Chapter 8 demonstrated custom MLIR dialects using Python string generation—a pedagogical approach that made IR construction transparent and iteration rapid. We built the `nn` dialect, generated high-level operations as text, implemented lowering transformations in Python, and saw how multi-level IR enables optimization. That string-based approach taught dialect concepts clearly, but it has limitations for production systems: no compile-time type checking, manual error handling, string manipulation overhead, and lack of integration with MLIR's optimization infrastructure.

**TableGen** solves these problems by providing a declarative language for defining MLIR dialects. Instead of writing Python code to generate MLIR text or C++ code to build IR manually, you write TableGen specifications that describe operations, their semantics, constraints, and syntax. A code generator (mlir-tblgen) reads these specifications and produces thousands of lines of C++ boilerplate—operation classes, parsers, printers, verifiers, and builders—automatically. This generated code integrates seamlessly with MLIR's infrastructure, providing type safety, compile-time validation, and access to the full suite of transformation passes.

This chapter reimplements the `nn` dialect from Chapter 8 using TableGen and C++ pattern rewriters, demonstrating how production MLIR systems are built. We'll examine TableGen's syntax for operation definitions, understand what C++ code it generates, implement lowering passes using OpRewritePattern, and integrate everything with Python bindings that use OpBuilder for IR construction. The result is a dialect implementation matching industry standards—the same techniques used by Torch-MLIR, JAX, IREE, and other production AI compilers. By the end, you'll understand not just what custom dialects are (from Chapter 8) but how they're built idiomatically in real-world MLIR projects.

## 9.1 Why TableGen? The Case for Declarative Specifications

Before diving into syntax and implementation, let's understand why MLIR's designers chose TableGen for dialect definition rather than direct C++ coding or runtime registration systems. The motivation comes from observing patterns in compiler development: operations in a dialect share common structures (operands, results, attributes), require similar boilerplate (constructors, accessors, printers), and need consistent verification (type checking, constraint validation). Writing this code manually for each operation is tedious, error-prone, and maintainable only at small scale.

**The Problem with Manual Operation Implementation**. Consider defining an operation in pure C++. You'd write a class inheriting from `mlir::Op`, implement methods for parsing MLIR text, printing IR, verifying constraints, and building instances programmatically. For a simple element-wise addition operation, this requires 200-300 lines of boilerplate: template instantiations, accessor methods, trait declarations, error handling. Multiply by ten operations in your dialect, and you have thousands of lines of repetitive code. When you modify an operation's signature—say, adding an attribute—you manually update the class declaration, parser, printer, verifier, and builder. Mistakes are easy; comprehensive testing is hard.

**TableGen as a Specification Language**. TableGen inverts this approach: you specify **what** an operation is (its name, operands, results, constraints) in a declarative format, and a code generator produces **how** to implement it. The specification is concise—often 5-15 lines per operation—and changes to the specification automatically propagate to all generated code. This declarative style is common in compiler infrastructure: yacc/bison for parsers, flex for lexers, and LLVM TableGen (which MLIR's TableGen extends) for instruction selection and register allocation. The principle: expert developers encode patterns once in the generator, users describe their specific cases declaratively, and the generator produces correct implementation code.

**What TableGen Generates**. For each operation defined in TableGen, mlir-tblgen produces:

1. **Operation Class**: A C++ class (e.g., `AddOp`) inheriting from `mlir::Op` with appropriate traits (number of operands/results, side effects, etc.). The class provides type-safe accessors for operands and results (`getLhs()`, `getRhs()`, `getOutput()`).

2. **Builders**: Static methods to construct the operation programmatically using OpBuilder. Multiple overloads handle different construction patterns (with/without attributes, with/without locations, etc.).

3. **Parsers and Printers**: Functions to read MLIR text syntax and write operations back to text. The `assemblyFormat` TableGen directive generates these automatically from a format specification.

4. **Verifiers**: Logic to check operation constraints at IR construction time. Type mismatches, incompatible shapes, or missing required attributes cause immediate compilation failures rather than silent bugs.

5. **Documentation**: Structured comments from the TableGen specification become searchable documentation for the dialect, accessible through MLIR's tools and integrated with IDE support.

This automation eliminates entire classes of bugs. If you change an operand's type from `AnyMemRef` to `Float32MemRef` in TableGen, the generated C++ code automatically enforces this constraint everywhere the operation is used. No manual updates needed, no risk of forgetting to update the verifier or builder.

**Type Safety and Compile-Time Validation**. Beyond code generation, TableGen enables compile-time correctness checking. When you build IR using generated OpBuilder methods, C++ type checking ensures arguments match the operation's signature. Try passing a `tensor<4xf32>` to an operation expecting `memref<4xf32>`, and the compiler rejects it. With Chapter 8's string generation, this error appears at runtime during MLIR parsing; with TableGen, it's caught at C++ compile time before any MLIR is generated. This shift-left of error detection accelerates development: fewer test-edit-debug cycles, more confidence in transformations.

**Integration with Optimization Infrastructure**. MLIR's pattern rewriting framework (used for optimization and lowering) works naturally with TableGen-defined operations. The `OpRewritePattern<AddOp>` base class provides type-safe access to operation fields, automatic matching, and integration with the rewrite driver. Dialect conversion passes compose seamlessly—you specify conversion targets (legal/illegal operations), provide rewrite patterns, and the framework handles application ordering, fixpoint computation, and rollback on failure. This infrastructure assumes operations are proper C++ classes with known structure, not text strings parsed at runtime.

**Stability and Versioning**. TableGen definitions serve as the source of truth for dialect evolution. When operations change—adding attributes, modifying constraints, deprecating old forms—the declarative specification makes these changes explicit and traceable. Production systems like StableHLO leverage this to generate compatibility layers: old IR versions can be automatically upgraded through generated conversion logic that compares TableGen definitions across versions. This versioning support is critical for long-lived projects where IR files persist across compiler updates.

**The Learning Curve Tradeoff**. TableGen adds a learning curve: you must understand its syntax (record types, dags, multiclass), operation traits (SideEffects, Commutative, etc.), and code generation mechanics. For simple prototypes or one-off experiments, string generation (Chapter 8) might be faster. But for dialects intended for reuse, extension, or optimization, TableGen pays dividends quickly. Once familiar with the patterns (which we'll cover systematically), defining new operations becomes mechanical: copy an existing definition, modify operands/results, regenerate code, implement lowering pattern. The overhead is fixed; the benefits scale with dialect size.

**Production Ecosystem Standards**. Every major MLIR project relies on TableGen as its source of truth. This includes frontends like TensorFlow (TF dialect) and PyTorch (Torch dialect); interchange standards like StableHLO (used by XLA and JAX); and execution environments like IREE (HAL and VM dialects) and Flang (FIR dialect). When you join these projects or integrate with them, understanding TableGen is mandatory—it's the lingua franca of MLIR dialect development. Learning it through our `nn` dialect example prepares you for reading, understanding, and contributing to production codebases. The investment in Chapter 9's techniques transfers directly to professional compiler development.

This chapter walks through TableGen systematically: dialect definition, operation specifications, generated code examination, pattern rewriter implementation, and Python API construction. We'll compare each step with Chapter 8's string approach, clarifying what TableGen automates and why. By the end, you'll write production-quality dialect definitions confidently, understand the generated C++ code thoroughly, and appreciate the engineering principles behind MLIR's design.

## 9.2 From Python Strings to TableGen Declarations

Let's examine the fundamental shift in how we define operations, comparing Chapter 8's string-based approach with Chapter 9's declarative TableGen specifications. This comparison clarifies what changes, what stays the same, and why the new approach better serves production needs. We'll use the `nn.add` operation as our example, tracing it through both implementations to understand the architectural differences.

**Chapter 8: Implicit Definition Through Text Generation**. In Chapter 8, operations didn't have formal definitions—they existed as patterns we emitted in Python strings. The `nn.add` operation was whatever text the `lower_add()` function generated:

```python
def lower_add(self, result_id: int, lhs_id: int, rhs_id: int, shape: List[int]) -> List[str]:
    lines = []
    memref_type = self._tensor_to_memref(shape)
    
    # Generate operation text directly
    lines.append(f"  %{result_id} = memref.alloc() : {memref_type}")
    lines.append(f"  linalg.generic {{ ... }} ins(%{lhs_id}, %{rhs_id} : ...)")
    # ... more string building
    return lines
```

The operation's syntax, semantics, and constraints existed only in this Python function. If you wanted to know what `nn.add` looked like in MLIR, you'd run the code and inspect the generated string. There was no declaration saying "the add operation takes two memrefs of the same shape and writes to a third," you just generated text that hopefully did that. Verification happened at MLIR parsing time, when mlir::parseSourceString tried to interpret your generated text. Errors appeared as parser diagnostics, not at operation generation time.

**Chapter 9: Explicit Declaration in TableGen**. With TableGen, operations are declared formally in `.td` files before any IR is generated. The `nn.add` operation in [inc/NNOps.td](../ch.9.TableGen-dialect/inc/NNOps.td):

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

This 12-line declaration specifies everything about the operation: its mnemonic (`add`), what operands it takes (three memrefs named `lhs`, `rhs`, `output`), how it appears in MLIR text (the `assemblyFormat`), and what it means (the `description`). From this specification, mlir-tblgen generates approximately 200 lines of C++ code handling construction, parsing, printing, and verification.

**The TableGen Advantage**. TableGen's declarative approach provides several benefits:

1. **Single Source of Truth**: The .td file is the authoritative definition. Generated code, documentation, and tools all derive from it consistently.

2. **Verification at Appropriate Levels**: Structural constraints (number of operands, type categories) verified at C++ compile time. Semantic constraints (shape compatibility, value ranges) verified at IR construction time.

3. **Incremental Development**: Add operations by writing small TableGen records, not implementing full parsing/printing/verification manually. The pattern scales: first operation takes learning time, subsequent operations are fast.

4. **Professional Ecosystem Integration**: Your dialect works with mlir-opt (pass testing), mlir-translate (format conversion), IDE plugins (syntax highlighting, autocomplete), and documentation generators automatically.

**The Learning Investment**. Understanding TableGen requires learning a new syntax and thinking declaratively rather than imperatively. Instead of "write code to generate strings representing operations," you think "declare what operations look like and let the generator produce implementation code." This shift feels awkward initially—experienced programmers want to control implementation details. But TableGen's constraints (limited expressiveness, rigid structure) are features, not bugs. They force consistency, prevent clever hacks, and ensure generated code follows MLIR best practices. Trust the generator; specify declaratively.

We'll explore TableGen syntax systematically in the next sections, starting with the dialect definition file, then examining operation records in detail, and finally looking at the generated C++ code to understand what TableGen produces.

## 9.3 TableGen Basics: Dialect Definition

TableGen files (`.td` extension) use a record-based syntax where you define **records**—instances of structured data, analogous to classes or structs—with **fields** (properties) that the code generator reads. MLIR's TableGen provides base records for common patterns: `Dialect` for dialect definitions, `Op` for operations, `Type` for custom types, `Attr` for attributes. You define specific records by inheriting from these bases and filling in fields. Let's start with the dialect definition in [inc/NNDialect.td](../ch.9.TableGen-dialect/inc/NNDialect.td), understanding each piece.

**File Structure and Includes**. TableGen files begin with includes bringing in base definitions:

```tablegen
//===- NNDialect.td - NN dialect definition ----------------*- tablegen -*-===//
#ifndef NN_DIALECT
#define NN_DIALECT

include "mlir/IR/OpBase.td"
```

The `include "mlir/IR/OpBase.td"` imports MLIR's core TableGen definitions: the `Dialect` record, `Op` base class, type constraints (`AnyType`, `AnyMemRef`, etc.), and trait specifications (`Pure`, `SameOperandsAndResultType`, etc.). These are the building blocks for defining custom dialects. The `#ifndef` guards prevent multiple inclusion (TableGen has C-preprocessor-like directives, though it's not actually C).

**Dialect Record Definition**. The dialect itself is defined with a `def` statement:

```tablegen
def NN_Dialect : Dialect {
  let name = "nn";
  let summary = "Neural Network Operations Dialect";
  let description = [{
    This dialect provides high-level operations for neural networks,
    including linear layers, activations, and normalization operations.

    Operations are designed to lower to standard MLIR dialects (linalg, arith, math).
  }];

  let cppNamespace = "::mlir::nn";

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
}
```

Let's examine each field:

**`let name = "nn";`** - The dialect's identifier in MLIR IR. Operations from this dialect will have the `nn.` prefix (e.g., `nn.add`, `nn.matmul`). This name must be unique across all loaded dialects in an MLIRContext.

**`let summary = "...";`** - A one-line description appearing in generated documentation and diagnostic messages. Keep it concise—this is what users see in error messages when the dialect fails to load or operations don't parse.

**`let description = [{...}];`** - Detailed documentation in TableGen's multi-line string syntax (`[{...}]`). This becomes structured documentation accessible through MLIR tools. The description should explain the dialect's purpose, design principles, and how it fits into larger compilation pipelines. For production dialects, this documentation is crucial—new contributors need context for why operations exist and how to use them.

**`let cppNamespace = "::mlir::nn";`** - The C++ namespace for generated code. All generated classes (`AddOp`, `MulOp`, etc.) will be in this namespace. The `::mlir::nn` choice follows MLIR conventions: top-level `mlir` namespace, dialect-specific subnamespace (`nn`). This prevents name collisions with other dialects and makes code organization clear.

**`let useDefaultAttributePrinterParser = 0;`** - Disables automatic generation of attribute printing/parsing code. For simple dialects without custom attributes, we can skip this complexity. If your dialect has custom attributes (e.g., neural network configuration like `#nn.conv_config<stride=2>`), you'd enable this and implement printing/parsing methods manually.

**`let useDefaultTypePrinterParser = 0;`** - Tells TableGen we are not providing custom type printing/parsing implementations. Our `nn` dialect exclusively reuses types from the Builtin (for scalar types like `f32` and `i32`), MemRef, and Tensor dialects (`memref<4xf32>`, `tensor<2x3xf32>`), so we don't define any custom types. Setting this to `1` would require manually implementing `printType/parseType` methods in C++. Production dialects often introduce domain-specific types—TensorFlow's `!tf.string` and `!tf.control`, PyTorch's `!torch.vtensor` and `!torch.optional`, StableHLO's `!stablehlo.token`, IREE's `!hal.buffer_view`—which require custom parsing logic to handle their internal structure and attributes.

**Base Operation Definition**. After the dialect, we define a base class for our operations:

```tablegen
class NN_Op<string mnemonic, list<Trait> traits = []> :
    Op<NN_Dialect, mnemonic, traits>;
```

This TableGen class (not a record—no `def`, just `class`) serves as a template for actual operations. It inherits from MLIR's `Op` class, passing three arguments:

1. **`NN_Dialect`** - Associates operations with our dialect. Generated code will register these operations with the dialect.
2. **`mnemonic`** - The operation name (e.g., `"add"`, `"matmul"`). Combined with the dialect name, this produces fully-qualified names like `nn.add`.
3. **`traits`** - A list of operation traits (properties like `Pure`, `Commutative`, `SameOperandsAndResultType`). Traits affect code generation and enable optimizations.

The `list<Trait> traits = []` syntax provides a default empty trait list, which individual operations can override. This pattern (base class with defaults, concrete definitions with specifics) is common in TableGen—it enables code reuse while allowing customization.

**File Footer**. The file ends with an include guard close:

```tablegen
#endif // NN_DIALECT
```

Standard C-style guard preventing multiple definition errors. TableGen processes includes textually, so these guards matter for complex project structures where files include each other.

**Code Generation**. When you run mlir-tblgen on this file, it generates [inc/NNDialect.h.inc](../build/x64-release/ch.9.TableGen-dialect/inc/NNDialect.h.inc) (included by your hand-written `NNDialect.h`) and [src/NNDialect.cpp.inc](../build/x64-release/ch.9.TableGen-dialect/src/NNDialect.cpp.inc) (included by `NNDialect.cpp`). These generated files contain:

```cpp
// NNDialect.h.inc
namespace mlir {
namespace nn {

class NNDialect : public ::mlir::Dialect {
public:
  explicit NNDialect(::mlir::MLIRContext *context);
  
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("nn");
  }
  
  // Operation registration methods (generated based on operations in NNOps.td)
  void initialize();
};

} // namespace nn
} // namespace mlir
```

The generated dialect class handles:
- Registration with MLIRContext
- Loading operations when the dialect is loaded
- Providing the dialect namespace to the framework

Your hand-written `NNDialect.cpp` includes this generated code and adds any custom initialization logic (none needed for our simple dialect, but complex dialects might register custom types, attributes, or canonicalization patterns here).

This dialect foundation supports all operation definitions we'll write next. The `NN_Op` base class provides a consistent starting point for `AddOp`, `MatMulOp`, `ReLUOp`, and others, ensuring they share common infrastructure while customizing their specific semantics. Before diving into specific operations, let's understand TableGen's core syntax—the language constructs you'll use to write all dialect definitions.

### 9.3.1 TableGen Language Essentials

TableGen is a domain-specific language for describing structured data that code generators can process. Unlike general-purpose programming languages, TableGen focuses on **declaration**, not execution—you describe what things are, not what to do. Understanding its core constructs takes away the mystery and makes reading `.td` files straightforward.

**Classes versus Definitions**. TableGen has two primary constructs:

A **`class`** is a template or pattern (like a C++ class template). It defines structure with parameters:

```tablegen
class NN_Op<string mnemonic, list<Trait> traits = []> :
    Op<NN_Dialect, mnemonic, traits>;
```

This declares "NN_Op is a pattern for operations in the NN dialect, parameterized by a mnemonic string and optional traits list." The template parameters (`<...>`) specify what changes between instances. The default value `= []` means traits is optional.

A **`def`** is a concrete instance (like a C++ object). It instantiates a class with specific values:

```tablegen
def NN_AddOp : NN_Op<"add"> {
  // Concrete definition
}
```

This creates an actual operation record named `NN_AddOp` by instantiating the `NN_Op` template with mnemonic `"add"` and default (empty) traits.

**Analogy**: Think of `class` as a cookie cutter (template), `def` as a cookie (specific instance). You design cookie cutters once, then stamp out many cookies. TableGen's code generator reads the `def` records (the cookies) and produces C++ code for each.

**The `let` Binding**. Fields in TableGen records are set with `let`:

```tablegen
def NN_AddOp : NN_Op<"add"> {
  let summary = "element-wise addition";
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs);
  let results = (outs AnyMemRef);
}
```

Each `let` assigns a value to a field inherited from the parent class. The `Op` base class (which `NN_Op` inherits from) defines fields like `summary`, `description`, `arguments`, `results`, `assemblyFormat`, etc. You fill them in with `let` statements. Think of it like C++ member initialization: the class declares the members, you provide the values.

**DAG Syntax (Directed Acyclic Graph)**. TableGen uses **DAG notation** for structured lists. A DAG looks like `(operator arg1:$name1, arg2:$name2, ...)`:

```tablegen
let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);
//              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//              This is a DAG
```

Breaking down the syntax:
- `ins` - The DAG operator (means "input operands")
- `AnyMemRef:$lhs` - First element: type `AnyMemRef` with name `$lhs`
- `,` - Separator between elements
- `$name` - The dollar sign indicates a bind variable (accessible in generated code)

**Common DAG operators**:
- `ins` - Input operands
- `outs` - Output results
- `attr` - Attributes (compile-time parameters)

For results:

```tablegen
let results = (outs F64Tensor:$result);
```

This defines a single output called `$result` with type `F64Tensor`. The names (`$lhs`, `$result`) become accessor method names in generated C++: `getLhs()`, `getResult()`.

**String Interpolation**. Multi-line strings use `[{ }]` delimiters (avoiding escape nightmares with quotes):

```tablegen
let description = [{
  This operation performs element-wise addition.
  It's similar to numpy's `+` operator.
  
  Example:
      nn.add %a, %b, %out : memref<4xf32>, memref<4xf32>, memref<4xf32>
}];
```

The content between `[{` and `}]` is literal text, including newlines and special characters. No escaping needed—write markdown, code examples, whatever you need. This becomes docstring content in generated code and documentation.

**Inheritance**. Definitions can inherit from classes, bringing in all the parent's fields:

```tablegen
def NN_AddOp : NN_Op<"add"> { ... }
//             ^^^^^^^^^^^^
//             Inheriting from NN_Op
```

This is like C++ single inheritance: `NN_AddOp` gets all fields from `NN_Op` (which got them from `Op`), plus any you add in the `{ }` body. Changes to the base class automatically affect all inheritors—modify `NN_Op` to add a trait, and all operations get it.

**Type Constraints**. The types (`AnyMemRef`, `F64Tensor`) are predefined type constraints from `mlir/IR/OpBase.td`. Common ones:

- `AnyType` - Any MLIR type
- `AnyMemRef` - Any memref type (any shape, any element type)
- `F32MemRef` - Memref with f32 elements specifically
- `1DTensorOf<[F32]>` - One-dimensional tensor of f32 values
- `AnyTypeOf<[...list...]>` - Union type (any of the listed types)

These constraints integrate with verification—if you specify `F32MemRef:$input`, the generated verifier ensures the operand is actually a memref with f32 elements. Type violations cause early errors, not runtime surprises.

**Traits as Lists**. Traits (operation properties) are specified as lists in angle brackets:

```tablegen
def ConstantOp : NN_Op<"constant", [Pure]> {
  //                                 ^^^^^^
  //                                 List of traits
}
```

Multiple traits: `[Pure, Commutative, SameOperandsAndResultType]`. Traits affect code generation (e.g., `Pure` enables dead code elimination), optimization opportunities, and verification logic. Each trait name corresponds to a C++ template class that adds behavior to the operation.

**Why This Syntax?** TableGen's syntax evolved from LLVM's needs: describe machine instructions, register sets, and compiler backends declaratively. The DAG syntax naturally represents instruction operands; the class/def distinction separates patterns from instances; the let binding makes defaults and inheritance clear. MLIR inherited this infrastructure and adapted it for operations, types, and dialects. The syntax isn't accidental—it solves real problems in compiler infrastructure generation.

Understanding these constructs—`class`, `def`, `let`, DAG syntax, inheritance, type constraints—makes any `.td` file readable. When you see:

```tablegen
def NN_MatMulOp : NN_Op<"matmul", [Pure]> {
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);
  let results = (outs);
}
```

You can decode it: "Define a concrete operation called NN_MatMulOp, inheriting from NN_Op template, with mnemonic 'matmul', having Pure trait, taking three memref operands named lhs/rhs/output, and producing no results." The TableGen compiler (mlir-tblgen) reads this record and generates the corresponding C++ operation class.

With this language foundation, let's examine specific operation definitions and see how these constructs combine to produce production-quality MLIR operations.

## 9.4 Operation Definition Syntax: The Add Operation

Individual operations are defined in [inc/NNOps.td](../ch.9.TableGen-dialect/inc/NNOps.td), each as a `def` record inheriting from our `NN_Op` base class. Let's examine the `add` operation in detail, understanding every line of its TableGen specification and what it means for code generation and usage.

**The Complete Definition**:

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

**Operation Name and Inheritance**. The definition starts with `def NN_AddOp : NN_Op<"add">`. The `NN_AddOp` identifier is the name of the generated C++ class—you'll write `AddOp op` (namespace-qualified as `mlir::nn::AddOp`) when working with this operation in C++. The inheritance `: NN_Op<"add">` specifies the operation's mnemonic (what appears after `nn.` in MLIR text). When combined with the dialect name from `NN_Dialect`, this produces the fully-qualified operation name `nn.add`.

**Documentation Fields**. The `summary` and `description` fields serve multiple purposes. The summary is a single line appearing in error messages and quick-reference documentation—think of it as a docstring or tooltip. The description is comprehensive documentation supporting markdown formatting, code examples, and detailed semantic explanations. This documentation becomes accessible through mlir-opt's `--help` flag, IDE hover tooltips, and generated HTML documentation. For production dialects, thorough descriptions are critical—they're the primary way users learn operation semantics without reading implementation code.

The `[{...}]` syntax for multi-line strings is TableGen-specific. Inside these delimiters, you can write multi-paragraph text with embedded code blocks, markdown formatting, and examples. The code example showing `nn.add %lhs, %rhs, %output : ...` demonstrates the operation's syntax, providing a template for users. When writing descriptions, include typical use cases, constraints (e.g., "shapes must match"), and semantic details (e.g., "uses IEEE 754 addition").

**Arguments Specification**. The `let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);` line defines the operation's operands. This uses TableGen's `DAG` syntax, which looks like function call syntax but represents structured data. The breakdown:

- **`ins`** - Keyword indicating these are input operands (as opposed to attributes, which use `attr`).
- **`AnyMemRef`** - A type constraint from MLIR's OpBase.td. This constraint accepts any memref type (`memref<4xf32>`, `memref<2x3xi32>`, etc.) but rejects other types (tensors, integers, etc.). Type constraints enable compile-time verification: if you try to pass a `tensor<4xf32>` to an operation expecting `AnyMemRef`, C++ compilation fails with a clear error.
- **`:$lhs`** - The colon followed by a dollar-sign name assigns the operand a C++ accessor name. This generates a method `Value getLhs()` in the operation class. The dollar-sign syntax ($name) is TableGen's way of defining named values within dags.

Our add operation takes three operands—two inputs and one output—all memrefs. The out-parameter pattern (`%output` is written to, not returned) matches the convention from Chapters 7 and 8: caller allocates result buffers, operations write in-place. This avoids tensor-to-memref bufferization complexity (a topic Chapter 4 covered extensively). For operations returning values, you'd use `let results = (outs AnyMemRef:$result);` instead of putting the output in `arguments`.

**Assembly Format Specification**. The `assemblyFormat` field defines how the operation appears in MLIR text. This format string generates both parser (text → IR) and printer (IR → text) code automatically. The syntax:

```tablegen
let assemblyFormat = "$lhs `,` $rhs `,` $output attr-dict `:` type($lhs) `,` type($rhs) `,` type($output)";
```

Each element in the format string has specific meaning:

- **`$lhs`, `$rhs`, `$output`** - References to the operands defined in `arguments`. These print the SSA value names (e.g., `%0`, `%1`, `%2`) and parse SSA values from text.
- **`,` (comma)** - Literal comma in the syntax. Backquoted literals (`\`,\``) are printed/parsed exactly as written.
- **`attr-dict`** - Placeholder for operation attributes (if any). Even though our add operation has no attributes, including `attr-dict` is conventional—it allows future additions without syntax breakage.
- **`:`** - Literal colon, conventional separator before type information.
- **`type($lhs)`, `type($rhs)`, `type($output)`** - Print the type of each operand. This produces output like `memref<4xf32>, memref<4xf32>, memref<4xf32>`.

The generated parser reads text matching this pattern and constructs an `AddOp` instance. The generated printer writes text in this format. This bidirectional generation is powerful: one format specification ensures parsing and printing are consistent (no "I print X but parse Y" bugs).

**Generated C++ Class**. From this 10-line TableGen definition, mlir-tblgen generates approximately 200 lines of C++ in `NNOps.h.inc` and `NNOps.cpp.inc`. The generated `AddOp` class includes:

```cpp
class AddOp : public ::mlir::Op<AddOp, 
                                 ::mlir::OpTrait::ZeroResults,
                                 ::mlir::OpTrait::ZeroSuccessors,
                                 ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nn.add");
  }

  // Accessor methods
  ::mlir::Value getLhs() { return getOperand(0); }
  ::mlir::Value getRhs() { return getOperand(1); }
  ::mlir::Value getOutput() { return getOperand(2); }

  // Builder methods
  static void build(::mlir::OpBuilder &builder,
                    ::mlir::OperationState &state,
                    ::mlir::Value lhs,
                    ::mlir::Value rhs,
                    ::mlir::Value output);

  // Parser and printer
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

  // Verification
  ::mlir::LogicalResult verify();
};
```

We'll examine this generated code in detail in the next section. For now, understand that the TableGen definition—10 lines of declarative specification—produces all boilerplate code needed for a fully functional MLIR operation. You don't write parsers, you don't write printers, you don't write accessor methods. You declare **what** the operation is, TableGen generates **how** it works.

This operation definition pattern—`def` record, documentation, arguments, assembly format—repeats for every operation in the dialect.

## 9.5 Matrix Multiplication: Handling Complex Operations

The `matmul` operation demonstrates how TableGen handles operations with richer semantics than simple element-wise transformations. Matrix multiplication involves two input matrices with specific shape constraints (inner dimensions must match) and produces an output with dimensions derived from inputs. Let's examine its definition and understand the patterns for complex operations.

**The MatMul Definition**:

```tablegen
def NN_MatMulOp : NN_Op<"matmul"> {
  let summary = "matrix multiplication";
  let description = [{
    Performs matrix multiplication: `C = A @ B`

    Supports 2D matrices with compatible shapes:
    - A: [M, K]
    - B: [K, N]
    - C: [M, N]

    Example:
    ```mlir
    nn.matmul %a, %b, %c : memref<2x3xf32>, memref<3x4xf32>, memref<2x4xf32>
    ```
  }];

  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);

  let assemblyFormat = "$lhs `,` $rhs `,` $output attr-dict `:` type($lhs) `,` type($rhs) `,` type($output)";
}
```

**Structural Similarity to Add**. Notice the definition structure mirrors `AddOp` almost exactly: same `arguments` specification (three memrefs), same `assemblyFormat`, identical accessor generation. This consistency is intentional—MLIR's operation model treats all operations uniformly at the structural level. Whether an operation performs element-wise addition or matrix multiplication is semantic detail, not structural difference. Both have operands (values they consume) and follow the same IR construction patterns.

**Semantic Constraints**. The difference lies in semantics, which the description explains: matmul requires 2D matrices with compatible shapes. The first operand's second dimension (K) must equal the second operand's first dimension (K). These constraints aren't encoded in the TableGen definition—`AnyMemRef` accepts any memref type, including incompatible shapes. Instead, constraints are verified at IR construction time (in generated builder methods) or through custom verifiers.

**Shape Verification Options**. For matmul, we have several verification strategies:

1. **Runtime Verification**: The generated `verify()` method (from TableGen) can check shapes dynamically:

```cpp
LogicalResult MatMulOp::verify() {
  auto lhsType = getLhs().getType().cast<MemRefType>();
  auto rhsType = getRhs().getType().cast<MemRefType>();
  auto outType = getOutput().getType().cast<MemRefType>();
  
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 || outType.getRank() != 2)
    return emitOpError("matmul requires 2D matrices");
  
  if (lhsType.getShape()[1] != rhsType.getShape()[0])
    return emitOpError("inner dimensions must match");
  
  // Check output shape...
  return success();
}
```

This verification runs when the operation is constructed or when IR is verified, catching shape mismatches early.

2. **Compile-Time Verification via Traits**: For stricter checking, TableGen traits can enforce properties. However, shape compatibility is too specific for general traits—traits like `SameOperandsShape` exist, but "first operand's second dim equals second operand's first dim" is matmul-specific.

3. **Type-Level Constraints**: Advanced TableGen allows custom type constraints:

```tablegen
def MatrixType : MemRefOf<[F32], [2]>;  // 2D f32 memref
let arguments = (ins MatrixType:$lhs, MatrixType:$rhs, MatrixType:$output);
```

This restricts operations to 2D float32 memrefs at the type system level. For our pedagogical dialect, dynamic verification suffices.

**Multiple Operations, Shared Patterns**. Let's look at the other operations in our dialect to see pattern repetition:

**Element-Wise Multiplication**:

```tablegen
def NN_MulOp : NN_Op<"mul"> {
  let summary = "element-wise multiplication";
  let description = [{
    Performs element-wise multiplication: `output = lhs * rhs`

    Example:
    ```mlir
    nn.mul %lhs, %rhs, %output : memref<4xf32>, memref<4xf32>, memref<4xf32>
    ```
  }];

  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);

  let assemblyFormat = "$lhs `,` $rhs `,` $output attr-dict `:` type($lhs) `,` type($rhs) `,` type($output)";
}
```

Identical structure to `AddOp`, only the operation name and description differ. This repetition suggests opportunities for abstraction—TableGen's `class` definitions and multiclass patterns let you factor common structure.

**ReLU Activation**:

```tablegen
def NN_ReLUOp : NN_Op<"relu"> {
  let summary = "ReLU activation function";
  let description = [{
    Applies the Rectified Linear Unit activation function:
    `output = max(0, input)`

    Example:
    ```mlir
    nn.relu %input, %output : memref<2x4xf32>, memref<2x4xf32>
    ```
  }];

  let arguments = (ins AnyMemRef:$input, AnyMemRef:$output);

  let assemblyFormat = "$input `,` $output attr-dict `:` type($input) `,` type($output)";
}
```

A **unary** operation (one input, one output) rather than binary. The pattern is the same: arguments specify operands, assemblyFormat defines syntax. The semantic difference (unary vs. binary) doesn't change the definition structure.

**Advanced Operations: Softmax and Linear**. Our dialect includes operations with different patterns:

```tablegen
def NN_SoftmaxOp : NN_Op<"softmax", [Pure, SameOperandsAndResultType]> {
  let summary = "Softmax activation function";
  let description = [{
    Applies the softmax function along the last dimension:
    `output[i] = exp(input[i]) / sum(exp(input[j]))`

    Example:
    ```mlir
    %probs = nn.softmax %logits : tensor<2x4xf32>
    ```
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($result)";
}
```

This operation uses **tensor** types (not memrefs) and **returns a value** (not out-parameter). The `let results = (outs ...)` syntax specifies return values, and the traits `[Pure, SameOperandsAndResultType]` indicate the operation is side-effect-free and preserves types. This demonstrates TableGen's flexibility: different operations can follow different conventions (memref out-param vs. tensor return) within the same dialect.

**The Linear Layer**:

```tablegen
def NN_LinearOp : NN_Op<"linear", [Pure]> {
  let summary = "Linear layer (fully connected)";
  let description = [{
    Performs a linear transformation with optional bias:
    `output = input @ weight^T + bias`

    Arguments:
    - input: [batch_size, in_features]
    - weight: [out_features, in_features]
    - bias: [out_features] (optional)

    Example:
    ```mlir
    %output = nn.linear %input, %weight, %bias 
        : tensor<2x3xf32>, tensor<4x3xf32>, tensor<4xf32> -> tensor<2x4xf32>
    ```
  }];

  let arguments = (ins 
    AnyTensor:$input,
    AnyTensor:$weight,
    Optional<AnyTensor>:$bias
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict 
    `:` type($input) `,` type($weight) (`,` type($bias)^)? `->` type($result)
  }];
}
```

This operation demonstrates **optional operands** (`Optional<AnyTensor>:$bias`) and **conditional syntax** in assemblyFormat (the `(`,` $bias^)?` pattern means "optionally print comma and bias"). The `^` marker in format strings indicates conditions: `$bias^` means "if bias exists." This generates parsing and printing logic that handles both `nn.linear %x, %w : ...` (no bias) and `nn.linear %x, %w, %b : ...` (with bias).

**TableGen Patterns for Code Reuse**. To reduce repetition for similar operations, TableGen supports `class` definitions with parameters:

```tablegen
class BinaryElementwiseOp<string mnemonic, string description> : NN_Op<mnemonic> {
  let summary = description;
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);
  let assemblyFormat = "$lhs `,` $rhs `,` $output attr-dict `:` type($lhs) `,` type($rhs) `,` type($output)";
}

def NN_AddOp : BinaryElementwiseOp<"add", "element-wise addition">;
def NN_MulOp : BinaryElementwiseOp<"mul", "element-wise multiplication">;
def NN_SubOp : BinaryElementwiseOp<"sub", "element-wise subtraction">;
```

This pattern (not in our current implementation, but common in production dialects) factors shared structure, making new operations one-line declarations. Our explicit definitions make learning easier; production code favors conciseness.

**Comparison with Chapter 8**. In Chapter 8, we didn't define operations at all—we generated lowered IR directly. There was no `nn.matmul` operation in Chapter 8's MLIR; we went straight to `linalg.matmul`. With TableGen, `nn.matmul` is a real operation that exists in IR, can be parsed from text, optimized, and then lowered. This separation of concerns (high-level operations vs. their implementations) enables staged compilation where each level is independently analyzable and transformable.

The matmul definition shows that complex operations with rich semantics still follow simple TableGen patterns: declare operands, specify syntax, document semantics. Verification logic (checking shapes, types, attributes) happens in generated or hand-written C++ code, not in TableGen itself. TableGen describes structure; C++ enforces semantics. Let's examine what C++ code TableGen generates from these definitions.

## 9.6 Generated C++ Code: What TableGen Produces

Understanding generated code is crucial for effective TableGen use: you need to know what methods are available, how to call builders, and what the operation class interface provides. Let's examine the generated `AddOp` class in detail, understanding each piece and how it integrates with MLIR's infrastructure. The generated code lives in [build/x64-release/ch.9.TableGen-dialect/inc/NNOps.h.inc](../build/x64-release/ch.9.TableGen-dialect/inc/NNOps.h.inc) and is included by hand-written header files.

**Class Declaration and Inheritance**. The generated `AddOp` class looks like:

```cpp
class AddOp : public ::mlir::Op<AddOp,
                                 ::mlir::OpTrait::ZeroResults,
                                 ::mlir::OpTrait::ZeroSuccessors,
                                 ::mlir::OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using Adaptor = AddOpAdaptor;
  // ... methods follow
};
```

The inheritance from `mlir::Op<AddOp, Traits...>` uses the Curiously Recurring Template Pattern (CRTP): the class inherits from a template parameterized by itself. This pattern enables compile-time polymorphism—MLIR's `Op` base class can call methods on the derived class without virtual function overhead. The traits specify compile-time properties:

- **`ZeroResults`** - Operation produces no SSA values (we use out-parameters, not returns)
- **`ZeroSuccessors`** - Operation has no control flow (not a branch/jump)
- **`NOperands<3>::Impl`** - Operation takes exactly 3 operands

These traits enable static verification: try constructing an `AddOp` with two operands, and C++ template instantiation fails with a compiler error.

**Accessor Methods**. The generated class provides type-safe accessors for each operand:

```cpp
::mlir::Value getLhs() { return getOperand(0); }
::mlir::Value getRhs() { return getOperand(1); }
::mlir::Value getOutput() { return getOperand(2); }
```

These methods retrieve operands by index from the underlying `Operation` storage, but with meaningful names from your TableGen specification. Instead of writing `op.getOperand(0)` (which operand is 0?), you write `op.getLhs()` (clear intent). The accessors return `mlir::Value`, MLIR's SSA value type. Values are lightweight handles to operation results or block arguments—they're copyable, pass-by-value, and thread-safe for reading.

**Builder Methods**. Multiple `build()` static methods construct operations:

```cpp
static void build(::mlir::OpBuilder &builder,
                  ::mlir::OperationState &state,
                  ::mlir::Value lhs,
                  ::mlir::Value rhs,
                  ::mlir::Value output) {
  state.addOperands({lhs, rhs, output});
}
```

OpBuilder uses these methods when you write `builder.create<AddOp>(loc, lhs, rhs, output)`. The builder handles operation insertion (which block, which position), location tracking (source information for debugging), and state management. Multiple `build()` overloads support different construction patterns—with/without explicit types, with/without attributes, etc.

**Parser and Printer**. Generated from `assemblyFormat`:

```cpp
static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                   ::mlir::OperationState &result);
void print(::mlir::OpAsmPrinter &p);
```

The parser reads MLIR text matching your format specification and populates an `OperationState` (the build-in-progress structure). The printer writes the operation back to text. You rarely call these directly—they're invoked by mlir-opt when reading/writing .mlir files.

**Verification**. The `verify()` method checks operation constraints:

```cpp
::mlir::LogicalResult verify();
```

For simple operations like add (where any three memrefs suffice), verification is minimal. For complex operations (matmul with shape constraints), you'd implement custom verification logic. Generated verifiers check structural properties (operand count, types match constraints); hand-written verifiers check semantic properties (compatible shapes, valid attribute values).

**The Operation Name**. A compile-time constant provides the fully-qualified name:

```cpp
static constexpr ::llvm::StringLiteral getOperationName() {
  return ::llvm::StringLiteral("nn.add");
}
```

This name is used for operation registration, text parsing, and diagnostics. It combines your dialect name and operation mnemonic.

**Comparison with Chapter 8**. In Chapter 8, we had none of this—no classes, no type safety, no builders. Operations were strings we emitted. Here, operations are C++ classes with APIs that prevent errors. Try passing an integer to `build()` expecting a Value, and C++ compilation fails. Try constructing AddOp with two operands instead of three, and template instantiation fails. These compile-time checks catch bugs before testing, accelerating development.

This generated code—200+ lines from 10 lines of TableGen—is production-ready, following MLIR conventions perfectly. You don't maintain it manually; TableGen regenerates it whenever definitions change.

### 9.6.1 Understanding Builders: Optional Convenience Constructors

Builders often confuse newcomers to TableGen. Let's clarify exactly what they are, when to use them, and how they relate to the generated code. **A builder is a convenience constructor** for creating operations—syntactic sugar that makes C++ code cleaner and more readable.

**Default Builders: Automatically Generated**. Every operation gets default builders automatically, even if you don't specify custom ones in TableGen. For our `AddOp` with three operands, TableGen generates:

```cpp
// Default builder (always generated):
static void build(OpBuilder &builder, OperationState &state,
                  Value lhs, Value rhs, Value output) {
  state.addOperands({lhs, rhs, output});
}
```

This `OpBuilder` parameter is the same builder we've used throughout earlier chapters—MLIR's standard IR construction API. It takes all operands explicitly. You use it as:

```cpp
builder.create<AddOp>(loc, lhs, rhs, output);
```

The builder signature matches your TableGen `arguments` specification exactly. No custom builder declaration needed—this always exists.

**Custom Builders: When and Why**. You add custom builders to make common usage patterns easier. Consider these scenarios:

**Scenario 1: Type Inference**. Suppose your operation's result type always matches the first operand's type:

```tablegen
def MyOp : NN_Op<"my_op"> {
  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyMemRef:$result);
  
  // Custom builder that infers result type from input
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];
}
```

Generated custom builder:

```cpp
static void build(OpBuilder &builder, OperationState &state,
                  Value input) {
  // Automatically infer result type from input
  state.addOperands({input});
  state.addTypes({input.getType()});
}
```

Usage:

```cpp
// Clean: result type inferred
builder.create<MyOp>(loc, input);

// vs default builder (verbose):
builder.create<MyOp>(loc, input.getType(), input);
```

**Scenario 2: Multiple Input Formats**. Operations might be constructed from different representations. MLIR's ConstantOp has builders for both dense attributes and scalar values:

```tablegen
let builders = [
  OpBuilder<(ins "DenseElementsAttr":$value)>,  // From attribute
  OpBuilder<(ins "double":$value)>               // From scalar
];
```

This enables:

```cpp
// Builder 1: tensor constant from dense data
auto dense = DenseElementsAttr::get(type, {1.0, 2.0, 3.0});
builder.create<ConstantOp>(loc, dense);

// Builder 2: scalar constant from double
builder.create<ConstantOp>(loc, 42.0);
```

Same operation, different construction paths, both type-safe and convenient.

**Scenario 3: Default Arguments**. Some operands might have sensible defaults:

```tablegen
let builders = [
  OpBuilder<(ins "Value":$input,
                 CArg<"bool", "false">:$transpose)>
];
```

The `CArg<Type, Default>` specifies an optional C++ argument with a default value. Usage:

```cpp
builder.create<MyOp>(loc, input);              // Uses false
builder.create<MyOp>(loc, input, true);         // Explicit true
```

**When to Add Custom Builders**:

1. **Type inference**: Result types derivable from operands
2. **Multiple formats**: Different ways to construct the same operation
3. **Convenience**: Hide complexity or provide defaults
4. **Common patterns**: Simplify frequently-used constructions

**When NOT to Add Custom Builders**:

- Default builder suffices (just pass all operands)
- Operation construction is rare (not worth the complexity)
- Type inference is ambiguous (explicit types clearer)

Builders are optional. If omitted, the default builder works fine. Add them when they genuinely improve code readability, not out of habit.

**The Three Layers Revisited**. Understanding builders requires seeing the full picture:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: TableGen Definition (.td file)                     │
│ → DECLARE structure: operands, results, traits              │
│ → OPTIONALLY declare custom builders                        │
│ → "Here's WHAT the operation is"                            │
└───────────────────────┬─────────────────────────────────────┘
                        ↓ mlir-tblgen generates
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Generated C++ Code (.inc files)                    │
│ → Operation class (AddOp)                                   │
│ → Default builders (always)                                 │
│ → Custom builders (if specified)                            │
│ → Accessors, parsers, printers, verifiers                   │
└───────────────────────┬─────────────────────────────────────┘
                        ↓ you implement
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Usage Code (MLIRGen.cpp, patterns, etc.)           │
│ → USE builder.create<AddOp>(...)                            │
│ → Call generated accessors (getLhs(), getRhs())             │
│ → "Here's how to CREATE and USE operations"                 │
└─────────────────────────────────────────────────────────────┘
```

TableGen defines structure. Generated code implements structure. Your code uses structure. Builders sit in layer 2 but are used in layer 3—they bridge specification and usage.

### 9.6.2 File Organization and Build Process

MLIR dialects follow a standard three-layer structure separating specification, generation, and implementation:

**1. TableGen Source (.td files)**: The authoritative specification in `inc/NNOps.td`:
```tablegen
def NN_AddOp : NN_Op<"add"> {
  let summary = "element-wise addition";
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);
  let assemblyFormat = "$lhs `,` $rhs `,` $output attr-dict `:` type($lhs) `,` type($rhs) `,` type($output)";
}
```

**2. Generated Code (.inc files)**: CMake runs `mlir-tblgen` during build to generate ~200 lines of C++ per operation in `build/.../NNOps.h.inc` and `NNOps.cpp.inc`—operation classes, accessors, builders, parsers, and printers. Never edit these files manually.

**3. Hand-Written Implementation (.cpp files)**: Custom logic in `src/NNOps.cpp` for semantics TableGen can't express:
```cpp
LogicalResult AddOp::verify() {
  auto lhsType = getLhs().getType().cast<MemRefType>();
  auto rhsType = getRhs().getType().cast<MemRefType>();
  if (lhsType.getShape() != rhsType.getShape())
    return emitOpError("operand shapes must match");
  return success();
}
```

**Why This Matters**: Changes to .td files automatically propagate to all generated code. You focus on semantics (verification, lowering), not boilerplate (accessors, parsers). When exploring production MLIR projects (TensorFlow, PyTorch, Flang), start with .td files for operation structure, check .cpp for custom logic, and skip .inc files (generated code).

## 9.7 C++ Pattern Rewriters: Implementing Lowering

Lowering operations from the `nn` dialect to standard MLIR dialects uses **OpRewritePattern**, a template class that matches operations and replaces them with equivalent lower-level IR. This pattern-based approach integrates with MLIR's dialect conversion framework, handling operation replacement, value mapping, and rollback automatically. Before examining specific lowering implementations, let's understand MLIR's pattern matching framework—how patterns are discovered, applied, and composed.

### 9.7.1 The Pattern Matching Framework

MLIR's pattern rewriting system walks through IR, matching operations against registered patterns and applying transformations when matches succeed. Think of it as regular expressions for code: patterns specify what to find (the **matcher**) and what to replace it with (the **rewriter**).

**Pattern Structure**. Every pattern inherits from `OpRewritePattern<OpType>` and implements `matchAndRewrite()`:

```cpp
class MyPattern : public OpRewritePattern<MyOp> {
public:
  MyPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<MyOp>(context, benefit) {}
  
  LogicalResult matchAndRewrite(MyOp op, PatternRewriter &rewriter) const override {
    // 1. Match: Check if this op matches our pattern
    if (!matchesCondition(op))
      return failure();  // Pattern doesn't apply
    
    // 2. Rewrite: Transform the op
    rewriter.replaceOp(op, newValues);
    return success();  // Pattern applied
  }
};
```

The framework calls `matchAndRewrite()` for every operation of type `MyOp` in the IR. Return `success()` if you applied a transformation, `failure()` if the pattern doesn't match (allowing other patterns to try).

**Greedy Rewriting**. MLIR uses **greedy pattern application**: repeatedly apply patterns until no more matches, reaching a fixed point:

```
Initial IR
    ↓
Apply pattern 1 → IR changed
    ↓
Apply pattern 2 → IR changed
    ↓
Apply pattern 1 → IR changed again
    ↓
Try pattern 1 → No match
Try pattern 2 → No match
    ↓
Fixed point reached (done)
```

This greedy approach continues until the IR stops changing. The framework maintains a worklist of operations to check, adding operations back when their operands change (potential new optimization opportunities).

**Pattern Benefits**. Patterns have **benefit scores** (positive integers) determining application priority:

```cpp
NNAddOpLowering(MLIRContext *context)
    : OpRewritePattern<AddOp>(context, /*benefit=*/1) {}
```

Higher benefit patterns are tried first. This priority system handles situations where multiple patterns match the same operation:

- **Benefit 2**: More specific, preferred pattern
- **Benefit 1**: General fallback pattern

Use benefits to express "prefer this transformation over that one" without complex coordination logic. Most patterns use benefit 1 (default); increase it for specialization cases.

**Pattern Ordering**. When multiple patterns match:

1. Sort patterns by benefit (descending)
2. Try highest-benefit pattern first
3. If it succeeds, mark IR changed and continue
4. If it fails (returns `failure()`), try next pattern
5. If all fail, move to next operation

This ordering ensures specific optimizations run before general ones, improving optimization effectiveness.

**The PatternRewriter API**. The `PatternRewriter` parameter provides methods for IR modification:

```cpp
// Replace operation with new values
rewriter.replaceOp(oldOp, newValues);

// Replace operation with another operation
auto newOp = rewriter.create<NewOp>(loc, operands);
rewriter.replaceOp(oldOp, newOp.getResults());

// Erase operation (must have no uses)
rewriter.eraseOp(op);

// Modify operation in-place
rewriter.updateRootInPlace(op, [&] {
  op->setOperand(0, newValue);
});

// Create new operations at insertion point
rewriter.setInsertionPoint(op);
Value result = rewriter.create<SomeOp>(loc, args);

// Inline region contents
rewriter.inlineRegionBefore(region, insertPoint);
```

The rewriter tracks all changes, enabling rollback if verification fails. Use it for all IR modifications in patterns—never modify IR directly.

### 9.7.2 The OpInterface System: Polymorphism for IR

The pattern rewriting we've seen uses `OpRewritePattern<AddOp>` to match specific operations. But notice something: `OpRewritePattern<T>` itself is a form of **interface**—it declares "any operation that wants custom lowering must implement `matchAndRewrite()`." MLIR generalizes this concept through its **interface system**, enabling polymorphism across operations and dialects. Understanding interfaces is crucial because many MLIR features—including the pattern system we just used—rely on them.

**The Interface Problem**. Suppose we want to write a shape inference pass. Without interfaces:

```cpp
void inferShapes(Operation *op) {
  if (auto addOp = dyn_cast<nn::AddOp>(op)) {
    addOp.getResult().setType(addOp.getLhs().getType());
  } else if (auto mulOp = dyn_cast<nn::MulOp>(op)) {
    mulOp.getResult().setType(mulOp.getLhs().getType());  // Same logic!
  } else if (auto matmulOp = dyn_cast<nn::MatmulOp>(op)) {
    // Different logic for matmul...
  } else if (/* and on and on for every operation */) {
    // ...
  }
}
```

This approach fails on multiple fronts: it's not extensible (adding operations requires modifying the pass), doesn't work across dialects (hardcoded to NN dialect), contains duplicated code for similar operations, and creates tight coupling between the pass and specific operation types.

**The Interface Solution**. With interfaces:

```cpp
void inferShapes(Operation *op) {
  if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
    shapeOp.inferShapes();  // Works for ANY operation implementing the interface!
  }
}
```

The interface provides **polymorphism**: different operations implement the same method differently, but the pass treats them uniformly. Note: `dyn_cast` is LLVM's type-casting utility that returns null on failure (unlike C++'s `dynamic_cast` which throws exceptions), making it safer and more efficient for compiler code that frequently tests types.

**Interfaces vs. Traits**. Recall traits from Chapter 9's operation definitions (like `Pure`). How do interfaces differ?

- **Traits**: Compile-time properties (`Pure` means "no side effects"), static checks, no runtime behavior
- **Interfaces**: Runtime polymorphism, dynamic dispatch, provide methods with behavior

**Analogy**: Trait = "This operation IS pure" (property). Interface = "This operation CAN infer shapes" (capability).

**OpRewritePattern as an Interface**. The pattern system we just used relies on interfaces! When you write:

```cpp
class AddOpLowering : public OpRewritePattern<nn::AddOp> {
  LogicalResult matchAndRewrite(nn::AddOp op, PatternRewriter &rewriter) const override;
};
```

You're implementing the **RewritePattern interface**. The framework calls your `matchAndRewrite()` polymorphically without knowing your specific pattern class. This is why patterns compose: the framework works with the interface, not concrete classes.

**Common MLIR Interfaces**. MLIR provides many built-in interfaces:

- `CastOpInterface`: For cast operations (`areCastCompatible()`)
- `CallOpInterface`: For function calls (`getCallableForCallee()`)
- `InferTypeOpInterface`: For type inference (`inferReturnTypes()`)
- `MemoryEffectOpInterface`: Describes memory side effects
- `LoopLikeOpInterface`: For loop constructs (`moveOutOfLoop()`)

Each enables generic algorithms working across operations and dialects. Passes use interfaces to remain dialect-agnostic.

**When to Use Interfaces**. Create interfaces when multiple operations need the same capability (like shape inference or auto-differentiation), you want generic passes working across dialects, or operations need polymorphic behavior with dynamic dispatch based on operation type. Don't create interfaces for operation-specific behavior—use regular methods instead. Interfaces are for **shared contracts** across multiple operations.

The next subsection shows how to define custom interfaces in TableGen, but understanding that `OpRewritePattern` itself uses interfaces helps explain why MLIR's pattern system is so flexible. Let's continue with lowering patterns, then we'll return to custom interfaces in section 9.8.5.

### 9.7.3 Lowering Add Operation

Let's examine the add operation's lowering in [src/NNToStandard.cpp](../ch.9.TableGen-dialect/src/NNToStandard.cpp), understanding the C++ pattern matching and IR construction APIs.

**Pattern Class Structure**:

```cpp
struct NNAddOpLowering : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                  PatternRewriter &rewriter) const override {
    // Lowering logic here
    return success();
  }
};
```

The pattern inherits from `OpRewritePattern<AddOp>`, which provides:
- Automatic matching: the framework calls this pattern whenever it encounters `AddOp`
- Type-safe operation access: the `op` parameter is typed as `AddOp`, not generic `Operation*`
- Rewriter access: the `rewriter` parameter provides IR construction methods

The `using OpRewritePattern<AddOp>::OpRewritePattern;` line inherits the constructor, allowing pattern registration. The `matchAndRewrite()` method returns `success()` if rewriting succeeded, `failure()` if the pattern doesn't apply (for conditional patterns).

**Lowering Add to Linalg.Generic**. The implementation:

```cpp
LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override {
  auto loc = op.getLoc();
  auto outputType = cast<MemRefType>(op.getOutput().getType());

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

  rewriter.eraseOp(op);
  return success();
}
```

**Step-by-Step Breakdown**:

1. **Extract Metadata**: `op.getLoc()` gets source location for debugging. `op.getOutput().getType()` retrieves the output memref type, which we cast to `MemRefType` for shape queries.

2. **Build Iterator Types**: Element-wise operations use parallel iterators (no dependencies between elements). We create a vector of `parallel` iterators, one per dimension. For 2D memrefs, this is `[parallel, parallel]`.

3. **Create Linalg.Generic**: The `rewriter.create<linalg::GenericOp>()` call constructs the lowered operation. Arguments:
   - **Location**: `loc` for debug info
   - **Inputs**: `ValueRange{op.getLhs(), op.getRhs()}` - the two input memrefs
   - **Outputs**: `ValueRange{op.getOutput()}` - the output memref (in-place)
   - **Indexing Maps**: Three identity maps (one per operand), meaning each iterator index directly maps to memref dimensions
   - **Iterator Types**: Our parallel iterator vector
   - **Body Builder Lambda**: A C++ lambda constructing the region body

  Note: We use `SmallVector` (LLVM's optimized vector container) instead of `std::vector` because it stores small arrays inline without heap allocation, improving performance for the common case where vectors contain few elements—critical in compiler code that creates millions of temporary vectors.

4. **Body Construction**: The lambda receives an `OpBuilder` (for building IR inside the region) and `ValueRange args` (scalar arguments extracted from input/output memrefs). We create `arith.addf %arg0, %arg1` and yield the result.

5. **Erase Original**: `rewriter.eraseOp(op)` removes the `nn.add` operation from IR. The rewriter tracks this replacement, updating all uses of the operation automatically.

**Comparison with Chapter 8**. Chapter 8's Python lowering generated strings:

```python
lines.append(f"  linalg.generic {{ ... }} ins(%{lhs_id}, %{rhs_id} : ...)")
```

Chapter 9's C++ lowering builds IR directly using OpBuilder:

```cpp
rewriter.create<linalg::GenericOp>(loc, inputs, outputs, maps, iterators, bodyBuilder);
```

The C++ approach type-checks arguments at compile time, verifies operations as they're built, integrates with the pattern rewriting framework, and handles value mapping and operation replacement automatically. No string formatting, no parse-time errors, no debugging text generation code—you build IR using typed C++ APIs.

**Matrix Multiplication Lowering**:

```cpp
struct NNMatMulOpLowering : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern<MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Zero constant for initialization
    auto zeroAttr = rewriter.getFloatAttr(outputType.getElementType(), 0.0);
    auto zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);

    // Fill output with zeros
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{op.getOutput()});

    // Matrix multiply (accumulates into output)
    rewriter.create<linalg::MatmulOp>(
        loc, ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{op.getOutput()});

    rewriter.eraseOp(op);
    return success();
  }
};
```

This follows the same pattern as add: extract metadata, create constants, build target operations (`linalg.fill` and `linalg.matmul`), erase original. The `linalg.MatmulOp` creation is simpler than generic—it's a named structured operation, not a generic with custom body.

**ReLU Lowering with Max**:

```cpp
struct NNReLUOpLowering : public OpRewritePattern<ReLUOp> {
  using OpRewritePattern<ReLUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReLUOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    
    auto zeroAttr = rewriter.getFloatAttr(outputType.getElementType(), 0.0);
    auto zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    
    SmallVector<utils::IteratorType> iteratorTypes(
        outputType.getRank(), utils::IteratorType::parallel);

    rewriter.create<linalg::GenericOp>(
        loc,
        ValueRange{op.getInput()},
        ValueRange{op.getOutput()},
        ArrayRef<AffineMap>{
            rewriter.getMultiDimIdentityMap(outputType.getRank()),
            rewriter.getMultiDimIdentityMap(outputType.getRank())
        },
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value maxVal = b.create<arith::MaximumFOp>(loc, zero, args[0]);
          b.create<linalg::YieldOp>(loc, maxVal);
        });

    rewriter.eraseOp(op);
    return success();
  }
};
```

ReLU uses `linalg.generic` with `arith.maximumf` instead of `addf`. The structure is identical—same pattern class, same rewriter usage, same operation erasure. This consistency is OpRewritePattern's strength: once you understand the pattern, all lowerings follow the same structure.

**The Conversion Pass**. Patterns are collected into a pass:

```cpp
struct ConvertNNToStandardPass
    : public PassWrapper<ConvertNNToStandardPass, OperationPass<ModuleOp>> {
  
  void runOnOperation() override {
    ConversionTarget target(getContext());
    
    // NN dialect operations are illegal (must be lowered)
    target.addIllegalDialect<NNDialect>();
    
    // Standard dialects are legal (lowering targets)
    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect, 
                          memref::MemRefDialect, func::FuncDialect>();
    
    // Register rewrite patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<NNAddOpLowering, NNMulOpLowering, 
                 NNMatMulOpLowering, NNReLUOpLowering>(&getContext());
    
    // Apply conversion
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
```

The pass defines a `ConversionTarget` (what's legal/illegal), registers patterns, and calls `applyPartialConversion()` to apply them. The framework iterates through IR, matches operations against patterns, applies rewrites, and repeats until fixpoint or failure. This infrastructure handles:
- Pattern application ordering
- Value remapping when operations are replaced
- Rollback if conversion fails
- Verification after each rewrite

You write pattern classes; MLIR handles application strategy.

**Production Pattern Libraries**. Real-world dialects have hundreds of operations and dozens of lowering patterns. MLIR provides utilities for organizing them:

- **Multiclass TableGen patterns**: Define lowering patterns in TableGen, generate C++ code
- **Interface-based lowering**: Operations implement interfaces that describe how to lower them
- **Cost models**: Patterns can specify costs, letting the rewriter choose optimal rewrites

For our pedagogical dialect, explicit pattern classes suffice. But understand that TableGen and interfaces can generate much of this code automatically—just as TableGen generates operation classes from definitions, it can generate pattern classes from declarative specifications.

This C++ pattern rewriting approach—type-safe, composable, integrated with MLIR's infrastructure—is how production compilers implement transformations. It's more complex than Python string manipulation, but the benefits (type safety, verifiability, tooling support) justify the investment.

### 9.7.5 TypeConverter: Managing Type Transformations in Dialect Lowering

The pattern rewriters we've implemented (Add, MatMul, ReLU) share a subtle assumption: **input and output types remain unchanged during lowering**. An `nn.add` operating on `memref<128x128xf32>` lowers to `linalg.generic` operating on the same memref type. But many real-world dialect lowerings require **type conversions**—transforming not just operations but the types they work with. A prime example is **bufferization** (covered in Chapter 4), which converts immutable tensor types to mutable memref types, fundamentally changing how data is represented in memory. This section explores MLIR's `TypeConverter` mechanism, showing how to handle such type changes systematically during dialect lowering.

**When Type Conversion Matters**. Consider lowering from high-level tensor types to low-level memref types:

```mlir
// High-level: tensor dialect (immutable, value semantics)
%result = tensor.add %lhs, %rhs : tensor<128x128xf32>

// Low-level: memref dialect (mutable, buffer semantics)
%buffer = memref.alloc() : memref<128x128xf32>
%result_memref = memref.add %lhs_memref, %rhs_memref, %buffer : memref<128x128xf32>
```

The operation changed (`tensor.add` → `memref.add`), but crucially the **type** changed too (`tensor<...>` → `memref<...>`). This creates cascading consequences:

- All operations consuming `%result` must expect memrefs, not tensors
- Function signatures using tensors must be updated to memrefs
- Block arguments passing tensors need type conversions
- Constants may require different representations

Manually tracking these type changes across large IR modules is error-prone. MLIR's `TypeConverter` automates this bookkeeping.

**TypeConverter Basics**. A TypeConverter defines rules for transforming types:

```cpp
class TensorToMemrefConverter : public TypeConverter {
public:
  TensorToMemrefConverter() {
    // Rule: Convert tensor types to memref types
    addConversion([](RankedTensorType tensorType) -> Type {
      return MemRefType::get(
          tensorType.getShape(),
          tensorType.getElementType(),
          MemRefLayoutAttrInterface{},  // Default layout
          nullptr  // No memory space
      );
    });
    
    // Rule: Keep memref types unchanged
    addConversion([](MemRefType type) { return type; });
    
    // Rule: Keep scalar types (i32, f32, etc.) unchanged
    addConversion([](Type type) {
      if (type.isa<IntegerType, FloatType, IndexType>())
        return type;
      return Type();  // nullptr means "can't convert"
    });
  }
};
```

Each `addConversion()` call registers a lambda taking an input type and returning an output type (or `Type()` for "no conversion possible"). The converter applies these rules during dialect lowering, transforming types systematically.

**Using TypeConverter in Patterns**. Instead of `OpRewritePattern`, use `OpConversionPattern` when types change:

```cpp
struct TensorAddOpLowering : public OpConversionPattern<tensor::AddOp> {
  using OpConversionPattern<tensor::AddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::AddOp op,
      OpAdaptor adaptor,  // Contains converted operands
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    
    // Allocate memref for result (tensors are values, memrefs are buffers)
    auto resultType = getTypeConverter()->convertType(op.getType())
                          .cast<MemRefType>();
    Value resultBuffer = rewriter.create<memref::AllocOp>(loc, resultType);
    
    // Create memref operation (operands already converted via adaptor)
    rewriter.create<memref::AddOp>(
        loc,
        adaptor.getLhs(),  // Converted to memref
        adaptor.getRhs(),  // Converted to memref
        resultBuffer
    );
    
    rewriter.replaceOp(op, resultBuffer);
    return success();
  }
};
```

**Key Differences from OpRewritePattern**:

1. **OpAdaptor**: The `adaptor` parameter provides **converted operands**. If the original `tensor::AddOp` took `tensor<128xf32>` operands, `adaptor.getLhs()` returns the memref-converted equivalent. You don't manually convert operands—the framework does it.

2. **getTypeConverter()**: Access the registered TypeConverter to explicitly convert types (like the result type in the example).

3. **ConversionPatternRewriter**: A specialized rewriter tracking type mappings. When you `replaceOp(old, new)`, it records that operations using `old`'s results should now use `new`.

**Materialization: Handling Type Boundaries**. Sometimes types can't convert directly—you need **materialization operations** to bridge representations. Example: keeping some operations in tensor dialect while others use memref:

```mlir
%tensor_result = tensor.add %a, %b : tensor<128xf32>
%memref_consumer = memref.load %buffer[...]  // Needs memref, not tensor
```

If lowering is partial (some operations lowered, some not), we need a `tensor.to_memref` operation bridging the gap. TypeConverter supports this via **target materialization**:

```cpp
class TensorToMemrefConverter : public TypeConverter {
public:
  TensorToMemrefConverter() {
    addConversion(/* ... type conversion rules ... */);
    
    // Materialization: Insert bufferization when tensor→memref boundary appears
    addTargetMaterialization([](OpBuilder &builder,
                                 MemRefType resultType,
                                 ValueRange inputs,
                                 Location loc) -> Value {
      assert(inputs.size() == 1 && "expected single tensor input");
      // Insert explicit tensor→memref conversion
      return builder.create<bufferization::ToMemrefOp>(
          loc, resultType, inputs[0]
      );
    });
    
    // Source materialization: memref→tensor (opposite direction)
    addSourceMaterialization([](OpBuilder &builder,
                                 TensorType resultType,
                                 ValueRange inputs,
                                 Location loc) -> Value {
      assert(inputs.size() == 1);
      return builder.create<bufferization::ToTensorOp>(
          loc, resultType, inputs[0]
      );
    });
  }
};
```

**Materialization Types**:

- **Target Materialization**: Converts **to** target types (e.g., tensor → memref when lowering to memref dialect)
- **Source Materialization**: Converts **from** target types (e.g., memref → tensor when not all operations lowered yet)
- **Argument Materialization**: Converts block arguments (function parameters, loop induction variables)

The framework inserts these materialization operations automatically at type boundaries. You define the conversion logic; MLIR determines where to insert.

**Full vs Partial Conversion**. MLIR supports two conversion modes:

```cpp
// Full conversion: All illegal operations MUST be converted
if (failed(applyFullConversion(module, target, std::move(patterns)))) {
  signalPassFailure();
}

// Partial conversion: Some illegal operations can remain (insert materializations)
if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
  signalPassFailure();
}
```

**Full conversion** fails if any illegal operation remains after pattern application. Use this when lowering must be complete (e.g., final code generation—no high-level ops can remain).

**Partial conversion** allows illegal operations to coexist with legal ones, inserting materializations at boundaries. Use this for gradual lowering (e.g., lower some tensor ops to memref while keeping others in tensor form temporarily).

Chapter 9's `ConvertNNToStandardPass` uses partial conversion—it allows tensor/memref operations to coexist because we're lowering custom dialect (`nn`) to standard dialects, not all the way to LLVM.

**Signature Conversion**. When function signatures involve converted types, use `ConversionPatternRewriter::convertRegionTypes()`:

```cpp
struct FuncOpTypeConversion : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp funcOp,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    // Convert function signature
    auto funcType = funcOp.getFunctionType();
    TypeConverter::SignatureConversion signatureConversion(
        funcType.getNumInputs()
    );
    
    // Convert each input type
    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
      Type convertedType = getTypeConverter()->convertType(
          funcType.getInput(i)
      );
      signatureConversion.addInputs(i, convertedType);
    }
    
    // Convert result types
    SmallVector<Type> convertedResults;
    if (failed(getTypeConverter()->convertTypes(
            funcType.getResults(), convertedResults))) {
      return failure();
    }
    
    // Update function signature
    rewriter.modifyOpInPlace(funcOp, [&] {
      funcOp.setType(rewriter.getFunctionType(
          signatureConversion.getConvertedTypes(),
          convertedResults
      ));
    });
    
    // Convert region (body) types
    if (failed(rewriter.convertRegionTypes(
            &funcOp.getBody(),
            *getTypeConverter(),
            &signatureConversion))) {
      return failure();
    }
    
    return success();
  }
};
```

This pattern updates function signatures **and** the function body's block arguments to match converted types. The framework ensures all uses of arguments receive correctly-typed values.

**Why TypeConverter Matters**. As your dialects grow complex, type conversions become pervasive:

- **Quantization**: f32 → i8 (lowering floating-point ML models to quantized inference)
- **Bufferization**: tensor → memref (Chapter 4's bufferization is systematic type conversion)
- **Target-specific types**: Converting generic `vector<4xf32>` to target-specific SIMD types (AVX, NEON)
- **ABI compliance**: Ensuring function signatures match calling conventions (struct passing, vector registers)

TypeConverter centralizes this complexity. Without it, every pattern must manually track type conversions—error-prone and unmaintainable. With it, declare conversion rules once, and patterns automatically get converted operands.

**Practical Example: Chapter 4 Revisited**. Chapter 4's bufferization (tensor → memref lowering) is essentially TypeConverter in action. If you revisit that chapter with TypeConverter knowledge, the "one-shot bufferization" pass is:

1. Define TensorToMemrefConverter (tensor → memref rules)
2. Register ConversionPatterns for tensor operations (tensor.add → memref.add, etc.)
3. Apply partial conversion (some operations stay in tensor form until later passes)
4. Insert materializations (bufferization.to_memref, bufferization.to_tensor) at boundaries

The complexity you saw in Chapter 4—tracking which operations bufferized, managing intermediate buffers—is TypeConverter handling type changes systematically.

**Debugging Type Conversions**. When type conversion goes wrong (common!), use these techniques:

```cpp
// Print type conversions during pattern application
auto convertedType = getTypeConverter()->convertType(originalType);
llvm::errs() << "Converted " << originalType << " to " << convertedType << "\n";

// Check if type is legal
if (!getTypeConverter()->isLegal(type)) {
  llvm::errs() << "Type " << type << " is not legal in target dialect\n";
}

// Dump IR after type conversion pass
module.dump();  // See where materializations were inserted
```

Common errors include missing conversion rules (TypeConverter returns null), using original operands instead of `adaptor.getOperands()`, and mismatched materializations causing type system inconsistencies.

TypeConverter handles type transformations systematically through conversion rules, OpConversionPattern for type-changing lowerings, and materializations for type boundaries. For simple dialect lowering like our `nn` to Linalg, TypeConverter isn't needed since types stay consistent. But production compilers with multiple lowering stages (HLO → Linalg → Affine → SCF → LLVM) rely on TypeConverter to make type conversions composable and maintainable across stages.

## 9.8 OpBuilder and Python Integration

The Python API in Chapter 9 differs fundamentally from Chapter 8: instead of generating MLIR text strings, it uses C++ OpBuilder to construct IR directly. This approach matches production ML frameworks like Torch-MLIR and JAX—Python provides high-level APIs, C++ handles IR construction, and the two integrate through pybind11 bindings. Let's examine the architecture in [src/bindings.cpp](../ch.9.TableGen-dialect/src/bindings.cpp), understanding how Python tensor operations become MLIR operations.

**The Tensor Class**. Python users work with a `Tensor` class wrapping NumPy arrays:

```python
import ch9
import numpy as np

a = ch9.Tensor(np.array([1., 2., 3., 4.], dtype=np.float32))
b = ch9.Tensor(np.array([5., 6., 7., 8.], dtype=np.float32))
c = a + b  # Operator overloading builds computation graph
result = ch9.forward(c)  # Compile and execute
```

The `Tensor` class doesn't execute operations immediately (eager execution). Instead, it tracks operations symbolically, building a computation graph. The `+` operator returns a new `Tensor` representing "add a and b," not the computed result. This deferred execution pattern matches PyTorch JIT and TensorFlow graph mode—operations build a graph, `forward()` compiles and executes it.

**C++ Graph Builder**. The C++ implementation maintains a graph of operations:

```cpp
class Graph {
  struct Node {
    std::string op_type;       // "add", "mul", "matmul", etc.
    std::vector<int> inputs;   // Input tensor IDs
    int output_id;             // Result tensor ID
    std::vector<int64_t> shape;// Output shape
  };
  
  std::vector<Node> nodes;
  std::map<int, py::array_t<float>> tensors;  // Tensor data
  int next_id = 0;
};
```

When Python calls `a + b`, the C++ code:
1. Creates a new tensor ID for the result
2. Adds a node to the graph: `{op_type: "add", inputs: [a.id, b.id], output_id: result.id}`
3. Returns a Python `Tensor` wrapping the result ID

No MLIR is generated yet—we're just recording operations.

**OpBuilder IR Construction**. The `forward()` function converts the graph to MLIR using OpBuilder:

```cpp
py::array_t<float> forward(const Tensor& output_tensor) {
  MLIRContext context;
  context.loadDialect<NNDialect, func::FuncDialect, 
                      memref::MemRefDialect>();
  
  OpBuilder builder(&context);
  auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());
  
  // Build function signature
  SmallVector<Type> inputTypes, resultTypes;
  for (auto& [id, array] : graph.tensors) {
    if (is_input(id)) {
      inputTypes.push_back(getMemRefType(builder, array));
    }
  }
  resultTypes.push_back(getMemRefType(builder, output_array));
  
  auto funcType = builder.getFunctionType(inputTypes, resultTypes);
  auto func = builder.create<func::FuncOp>(
      builder.getUnknownLoc(), "main", funcType);
  
  Block* entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  
  // Map input tensor IDs to function arguments
  std::map<int, Value> valueMap;
  for (auto [i, id] : enumerate(input_ids)) {
    valueMap[id] = entryBlock->getArgument(i);
  }
  
  // Build operations in topological order
  for (auto& node : graph.nodes) {
    Value lhs = valueMap[node.inputs[0]];
    Value rhs = valueMap[node.inputs[1]];
    
    // Allocate output
    auto outputType = getMemRefType(builder, node.shape);
    Value output = builder.create<memref::AllocOp>(
        builder.getUnknownLoc(), outputType);
    
    // Create operation
    if (node.op_type == "add") {
      builder.create<AddOp>(builder.getUnknownLoc(), lhs, rhs, output);
    } else if (node.op_type == "mul") {
      builder.create<MulOp>(builder.getUnknownLoc(), lhs, rhs, output);
    } else if (node.op_type == "matmul") {
      builder.create<MatMulOp>(builder.getUnknownLoc(), lhs, rhs, output);
    }
    
    valueMap[node.output_id] = output;
  }
  
  builder.create<func::ReturnOp>(builder.getUnknownLoc(), valueMap[output_tensor.id]);
  
  // Compile and execute (same as Chapter 8: lower to LLVM, JIT, call via libffi)
  lowerToLLVM(module);
  void* fnPtr = compile(module, "main");
  return execute(fnPtr, input_arrays, output_shape);
}
```

**Topological Ordering Requirement**. The code above assumes `graph.nodes` is in topological order, meaning each operation's inputs are computed before the operation itself. This dependency ordering is fundamental to IR construction—you can't use a value before it's defined. Chapter 10 covers topological traversal algorithms in depth, showing how to sort arbitrary computation graphs into valid execution order.

**OpBuilder Methods**. Key OpBuilder APIs:

```cpp
// Create operations
Value result = builder.create<AddOp>(loc, lhs, rhs, output);

// Get types
MemRefType type = MemRefType::get({4}, builder.getF32Type());

// Manage insertion point
builder.setInsertionPointToStart(block);
builder.setInsertionPointAfter(op);

// Create blocks
Block* block = builder.createBlock(&region);
```

OpBuilder provides methods for every aspect of IR construction: creating operations (using generated `build()` methods), managing insertion points (where new operations appear), creating types, and building blocks/regions. The builder tracks context, handles operation insertion automatically, and enforces invariants (e.g., operations must be inserted into blocks, blocks must be in regions).

**Operator Overloading in Python**. The Pythonic API comes from pybind11 bindings:

```cpp
py::class_<Tensor>(m, "Tensor")
  .def(py::init<py::array_t<float>>())
  .def("__add__", &Tensor::add)
  .def("__mul__", &Tensor::mul);

py::def("matmul", &Graph::matmul);
py::def("forward", &forward);
```

Python's `a + b` calls `Tensor.__add__()`, which calls the C++ `add()` method, which records an add node. This provides PyTorch-like syntax while building MLIR IR underneath.

The pattern is universal: Python for user APIs and dynamic behavior, C++ for IR construction and optimization, pybind11 for integration. Our Chapter 9 implementation, while pedagogical, follows real production patterns faithfully.

This architecture—TableGen-defined operations, C++ pattern rewriters, OpBuilder-based Python API—represents industrial MLIR dialect development. It's more complex than Chapter 8's string approach, but it scales to thousands of operations, enables sophisticated optimizations, and integrates with the broader MLIR ecosystem.

### 9.8.5 Writing Custom Interfaces in TableGen

Section 9.7.2 introduced interfaces conceptually, showing how `OpRewritePattern` uses the interface system. Now let's see how to **define custom interfaces** in TableGen, enabling generic algorithms that work across operations and dialects. While Chapter 9 doesn't require custom interfaces (we use existing MLIR ones like `OpRewritePattern`), understanding interface definition is crucial for advanced dialect development—particularly when building extensible compiler infrastructure.

**Interface Definition in TableGen**. Interfaces are defined like operations, using TableGen. Create `ShapeInferenceInterface.td`:

```tablegen
#ifndef SHAPE_INFERENCE_INTERFACE
#define SHAPE_INFERENCE_INTERFACE

include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface for operations that can infer their output shapes from input shapes.
    Used by shape inference passes to propagate concrete types through IR.
  }];

  let methods = [
    InterfaceMethod<
      "Infer and set the output shape for this operation.",
      "void", "inferShapes"
    >
  ];
}

#endif // SHAPE_INFERENCE_INTERFACE
```

**Breaking It Down**:

1. **OpInterface Declaration**: `def ShapeInferenceOpInterface : OpInterface<"ShapeInference">` defines an interface named `ShapeInference` (the C++ class name). The definition name `ShapeInferenceOpInterface` is for TableGen reference.

2. **Description**: The `let description` provides documentation explaining the interface's purpose and when to use it.

3. **InterfaceMethod**: Declares methods operations must implement. Parameters:
   - Description: `"Infer and set the output shape..."`
   - Return type: `"void"`
   - Method name: `"inferShapes"`

4. **Methods with Arguments**: Interfaces can have methods with parameters:

```tablegen
InterfaceMethod<
  "Check if two shapes are compatible",
  "bool", "areShapesCompatible",
  (ins "ArrayRef<int64_t>":$shape1, "ArrayRef<int64_t>":$shape2)
>
```

This generates: `virtual bool areShapesCompatible(ArrayRef<int64_t> shape1, ArrayRef<int64_t> shape2) = 0;`

5. **Default Implementations**: Interfaces can provide default method implementations:

```tablegen
InterfaceMethod<
  "Get number of dimensions",
  "unsigned", "getRank",
  (ins), [{
    return $_op.getType().cast<ShapedType>().getRank();
  }]
>
```

The `$_op` variable refers to the operation implementing the interface. Operations can override defaults if needed.

**Declaring Interface Implementation**. Operations declare interface implementation in their trait list:

```tablegen
def AddOp : NN_Op<"add", [
    Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
  ]> {
  let summary = "element-wise addition";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor);
}
```

`DeclareOpInterfaceMethods<ShapeInferenceOpInterface>` tells TableGen: "This operation implements `ShapeInference` interface, generate method declarations." TableGen generates in the C++ header:

```cpp
class AddOp : /* ... */ public ShapeInference {
public:
  void inferShapes();  // Declaration only, you implement in .cpp
};
```

**Implementing Interface Methods**. In `Dialect.cpp`, provide the implementation:

```cpp
void AddOp::inferShapes() {
  // Element-wise operations preserve shape
  getResult().setType(getLhs().getType());
}
```

For operations with different logic (like `TransposeOp`):

```cpp
void TransposeOp::inferShapes() {
  auto inputType = cast<RankedTensorType>(getOperand().getType());
  
  // Reverse dimensions
  SmallVector<int64_t, 2> dims(llvm::reverse(inputType.getShape()));
  
  // Create transposed type
  getResult().setType(
    RankedTensorType::get(dims, inputType.getElementType())
  );
}
```

**Using Interfaces in Passes**. Generic passes use `dyn_cast` to query interfaces:

```cpp
void MyPass::runOnOperation() {
  getOperation().walk([](Operation *op) {
    if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
      shapeOp.inferShapes();  // Polymorphic call!
    }
  });
}
```

The `dyn_cast<ShapeInference>(op)` checks if `op` implements the interface. If yes, returns a handle to the interface; if no, returns `nullptr`. This works across **any dialect**—the pass doesn't know about specific operations, only the interface contract.

Interfaces enable extensibility (new operations integrate without modifying passes), dialect independence (passes work on any dialect implementing the interface), and code reuse (one shape inference pass instead of N operation-specific implementations). Chapter 14 explores advanced patterns including multiple interfaces per operation, interface inheritance, and type/attribute interfaces.

### 9.8.6 Advanced TableGen Features for Production Dialects

The TableGen specifications we've covered—operation definitions (section 9.4), generated code (section 9.6), and interfaces (section 9.8.5)—form the foundation for custom dialects. But production MLIR projects use additional TableGen features that dramatically improve productivity and code quality. This section surveys advanced techniques you'll encounter in real codebases: trait composition, constraints, and custom assembly formats. We'll also preview **Declarative Rewrite Rules (DRR)**, deferring detailed coverage to Chapter 14 where we explore production optimization patterns.

#### Declarative Rewrite Rules (DRR): Preview of Chapter 14

Section 9.7 showed C++ pattern rewriters—classes matching operations and replacing them with lower-level equivalents. For dialect lowering (like our `nn.add` → `linalg.generic` transformations), C++ patterns provide necessary flexibility. But for simple optimizations like constant folding or algebraic simplification, writing full C++ pattern classes becomes verbose.

**Chapter 14's DRR**. Declarative Rewrite Rules express optimizations in TableGen instead of C++:

```tablegen
// Chapter 14 preview: Optimization patterns in TableGen
def FoldDoubleTranspose : Pat<
  (TransposeOp (TransposeOp $x)),  // Match: transpose(transpose($x))
  (replaceWithValue $x)             // Replace: just $x
>;

def FoldAddZero : Pat<
  (AddOp $x, (ConstantOp $zero)),  // Match: add(x, 0.0)
  (replaceWithValue $x),            // Replace: x
  [(IsZeroFloat $zero)]             // Constraint: verify zero
>;
```

These TableGen patterns generate C++ code automatically—the same `OpRewritePattern` classes we wrote manually in section 9.7, but generated from concise declarative specifications.

**Why Defer to Chapter 14?** DRR is primarily an **optimization tool**, not a lowering tool.
DRR shines for patterns like "remove redundant operations," "fold constants," "simplify expressions"—optimizations that don't change abstraction levels. Chapter 9's lowering patterns (transforming `nn.add` to `linalg.generic`) require C++ flexibility for constructing complex target IR.
For now, focus on C++ `OpRewritePattern`—it's the workhorse for dialect lowering and gives you complete control over IR transformation.

#### Trait Composition: Declaring Operation Properties

Section 9.4 briefly mentioned traits—properties like "this operation has no side effects." TableGen provides rich trait vocabulary:

```tablegen
def AddOp : MyDialect_Op<"add", [
  Pure,                // No side effects (enables dead code elimination)
  Commutative,         // add(a, b) == add(b, a) (enables reordering)
  SameOperandsAndResultType,  // All operands and results have same type
  Elementwise          // Operates element-wise (enables fusion)
]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
```

Each trait affects how MLIR treats the operation:

**Pure** (formerly `NoSideEffect`): Operation doesn't modify memory or state. Enables:
- Dead code elimination (remove unused pure ops)
- Common subexpression elimination (reuse identical pure op results)
- Reordering (pure ops can move across other pure ops)

**Commutative**: Operand order doesn't matter. Enables:
- Canonicalization (sort operands for pattern matching)
- CSE across operand permutations (recognize `add(a, b)` == `add(b, a)`)

**SameOperandsAndResultType**: Type constraint enforcement. Catches bugs at verification time:
```mlir
%result = myop.add %a, %b : f32, i32 → f32  // ERROR: operand types differ
```

**Elementwise**: Operation applies independently to each element (like map). Enables:
- Fusion (elementwise ops chain without intermediate buffers)
- Vectorization (scalar elementwise → vector elementwise)

**Custom Traits**. Define domain-specific traits:

```tablegen
// Trait: Operation is safe to speculate (run before branches resolve)
def Speculatable : OpTrait<"Speculatable"> {
  let cppNamespace = "::mlir::OpTrait";
}

def MyExpensiveOp : MyDialect_Op<"expensive", [Speculatable]> {
  // Can be speculatively executed (if cheap enough)
}
```

Optimization passes query traits:
```cpp
if (op->hasTrait<OpTrait::Speculatable>()) {
  // Safe to move this op before branch
  hoistOutOfConditional(op);
}
```

Traits provide **declarative metadata** enabling generic optimizations.

#### Advanced Constraints: Type and Attribute Validation

Section 9.4 used `AnyType` and `F32MemRef`—basic type constraints. Production dialects need precise constraints:

```tablegen
// Ranked tensor of floating-point type
def FloatTensor : TensorOf<[AnyFloat]>;

// 2D memref with specific element type
def Matrix2D : MemRefOf<[F32, F64], [2]>;

// Memref with at least 2 dimensions
def MemRefRank2Plus : MemRefOf<[AnyType], [2, -1]>;

// Integer attribute in range [0, 7]
def SmallInt : Confined<I32Attr, [IntMinValue<0>, IntMaxValue<7>]>;

// Array attribute with exactly 4 elements
def Vec4Attr : Confined<ArrayAttr, [ArrayCount<4>]>;
```

These constraints generate verification code:
```cpp
// Generated verifier for Matrix2D operand
if (!operandType.isa<MemRefType>())
  return emitError("expected memref type");
auto memref = operandType.cast<MemRefType>();
if (memref.getRank() != 2)
  return emitError("expected 2D memref");
if (!memref.getElementType().isa<Float32Type, Float64Type>())
  return emitError("expected f32 or f64 element type");
```

Detailed constraints catch errors early (at IR construction time), not late (during lowering or execution).

**Custom Constraints**. Define domain-specific checks:

```tablegen
def DivisibleBy4 : AttrConstraint<
  CPred<"$_self.cast<IntegerAttr>().getInt() % 4 == 0">,
  "integer divisible by 4"
>;

def TiledMatMulOp : MyDialect_Op<"tiled_matmul"> {
  let arguments = (ins
    AnyMemRef:$A,
    AnyMemRef:$B,
    AnyMemRef:$C,
    DivisibleBy4:$tileSize  // Must be multiple of 4
  );
}
```

The generated verifier enforces `tileSize` divisibility, preventing runtime errors from misaligned tiles.

#### Custom Assembly Format: Pretty IR Printing

By default, TableGen generates verbose IR syntax:
```mlir
%result = nn.add(%lhs, %rhs, %output) : (memref<128xf32>, memref<128xf32>, memref<128xf32>) -> ()
```

Custom assembly formats make IR readable:

```tablegen
def AddOp : MyDialect_Op<"add"> {
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);
  
  let assemblyFormat = [{
    $lhs `,` $rhs `->` $output attr-dict `:` type($lhs)
  }];
}
```

Generated IR:
```mlir
%result = nn.add %lhs, %rhs -> %output : memref<128xf32>
```

Much cleaner! The format string specifies:
- Operand order: `$lhs`,` $rhs`
- Custom syntax: `->` between inputs and output
- Type printing: `:` type($lhs)` (print lhs type once, inferring others)

**Format Directives**:
- `$operand`: Print operand
- `attr-dict`: Print attributes (if any)
- `type($operand)`: Print operand type
- Literal strings: `,`, `->`, `to`, etc.
- `custom<Name>($operand)`: Invoke custom parsing/printing (defined in C++)

**Complex Example**:
```tablegen
let assemblyFormat = [{
  `<` $transposeA `,` $transposeB `>` 
  `(` $A `,` $B `)` `->` $C 
  attr-dict `:` type($A) `,` type($B) `->` type($C)
}];
```

Generates:
```mlir
nn.matmul <false, true> (%A, %B) -> %C : memref<128x256xf32>, memref<256x512xf32> -> memref<128x512xf32>
```

Boolean attributes (`transposeA`) appear inline, types are explicit, syntax resembles mathematical notation. Custom formats improve debugging through readable IR, match user expectations (mathematical notation for ML ops), reduce visual clutter, and enforce consistent IR style—transforming complex operations from unreadable multi-line messes into source-code-like clarity.

#### Putting It Together: Production Dialect Checklist

When building real dialects, apply these advanced features:

1. **Operations**: Use TableGen with precise constraints, appropriate traits, and custom assembly formats
2. **Interfaces**: Implement standard interfaces (`InferTypeOpInterface`, `MemoryEffectOpInterface`) for ecosystem integration
3. **Lowering**: Use C++ `OpConversionPattern` for complex transformations requiring full IR construction control
4. **Documentation**: TableGen comments generate HTML docs automatically
5. **Testing**: Use MLIR's `FileCheck` to verify generated IR matches expectations

Example combining techniques:
```tablegen
def MatMulOp : MyDialect_Op<"matmul", [
  Pure,  // No side effects
  DeclareOpInterfaceMethods<InferTypeOpInterface>  // Implement type inference
]> {
  let summary = "Matrix multiplication operation";
  let description = [{
    Performs C = A @ B where A is MxK, B is KxN, C is MxN.
    Supports optional transposition of inputs.
  }];
  
  let arguments = (ins
    FloatTensor:$A,
    FloatTensor:$B,
    DefaultValuedAttr<BoolAttr, "false">:$transposeA,
    DefaultValuedAttr<BoolAttr, "false">:$transposeB
  );
  
  let results = (outs FloatTensor:$result);
  
  let assemblyFormat = [{
    `(` $A `,` $B `)`
    (`transpose_a` $transposeA^)?
    (`transpose_b` $transposeB^)?
    attr-dict `:` type($A) `,` type($B) `->` type($result)
  }];
}
```

This 25-line TableGen spec:
- Defines operation with type constraints
- Declares trait (Pure) and interface (InferTypeOpInterface)
- Provides documentation
- Specifies custom IR syntax

The generated C++ (~500 lines) handles parsing, printing, verification, type inference, and pattern matching—all from declarative specifications.

## 9.9 Conclusion: Declarative Specifications for Production

This chapter demonstrated production-quality dialect implementation using TableGen, OpRewritePattern, and OpBuilder—MLIR's approach to making common patterns easy to express correctly.

**Core Concepts**: TableGen's ~10 lines defining AddOp generate ~200 lines of correct C++ (builders, accessors, parsers, printers, verifiers), eliminating maintenance burden. OpRewritePattern provides type-safe lowering through template matching, with the framework handling pattern ordering, value remapping, and rollback. OpBuilder enables direct IR construction with compile-time type checking and IDE autocomplete.

**Production Usage**: Industrial projects use these techniques universally—Torch-MLIR defines 1000+ operations in TableGen, JAX uses StableHLO dialect operations, TensorFlow uses MHLO/CHLO dialects, and IREE's VM/Flow/HAL dialects all rely on TableGen. The pattern is consistent: declarative specifications for operations, generated C++ for classes, OpRewritePattern for transformations, OpBuilder for IR construction.

**Chapter 10: Optimization Patterns**. Next, we introduce **Declarative Rewrite Rules (DRR)**, extending TableGen's declarative approach to optimizations. Instead of writing C++ pattern matchers, you'll express transformations directly in TableGen: "if you see pattern X, replace it with pattern Y." This reduces 30-line C++ patterns to 3-line TableGen declarations, making optimizations composable and maintainable.

**Chapter 11-12**: Chapter 11 explores attention mechanisms with complex operation signatures, and Chapter 12 builds complete transformer models. The composability we've established—TableGen for definitions, OpRewritePattern for lowering, OpBuilder for construction—scales to arbitrarily complex models.