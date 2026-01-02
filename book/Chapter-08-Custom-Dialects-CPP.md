# Chapter 8: Custom Dialects with Python and C++

Chapter 7 built a computation graph system where operations tracked dependencies symbolically, then generated complete MLIR modules with standard dialects (Linalg, Arith, SCF). That approach works—our graphs compose operations and produce correct, executable code—but it has limitations. Every operation generates its lowering immediately into low-level dialects, mixing high-level intent (matmul, relu) with implementation details (nested loops, index calculations). This makes the generated IR verbose and obscures the computation's structure. When you inspect the MLIR for a two-layer neural network, you see dozens of operations for loop construction before seeing what the network actually does.

**Custom dialects** solve this by introducing operation types that represent domain-specific computations at the right abstraction level. Instead of generating twenty lines of Linalg and Arith operations for a matrix multiply, we generate one `nn.matmul` operation that captures the intent directly. The dialect defines what operations mean (semantics), what types they accept (constraints), and how they print (assembly format). Lowering to standard dialects happens separately, as a transformation pass that runs after the high-level IR is constructed. This separation—high-level representation first, implementation details later—is the essence of MLIR's multi-level approach.

This chapter demonstrates custom dialects using a **string-based Python implementation** that generates MLIR text directly, focusing on understanding dialect concepts without the complexity of TableGen or OpBuilder APIs. We define an `nn` (neural network) dialect with operations like `nn.add`, `nn.matmul`, and `nn.relu`, build computation graphs in Python that generate `nn` dialect IR, then implement lowering passes in Python that convert `nn` operations to standard MLIR. The C++ side handles MLIR text parsing, compilation, and execution using the universal libffi approach from Chapter 7. This architecture teaches dialect design clearly while showing how Python and C++ responsibilities partition naturally in MLIR-based systems.

## 8.1 Why Custom Dialects? The Abstraction Gap

When we wrote Chapter 7's graph API, every operation immediately generated its implementation using Linalg, SCF, and Arith operations. Adding two vectors produced `memref.alloc`, `scf.for`, `memref.load`, `arith.addf`, `memref.store`—five operation types for a conceptually simple addition. Matrix multiplication with Linalg was better (one `linalg.matmul`), but ReLU and softmax fell back to explicit loops. For a two-layer MLP with two matmuls and one ReLU, the generated MLIR contained over fifty operations before any optimization passes ran. This explosion of low-level details makes several problems evident.

**Problem 1: IR Verbosity**. When debugging or inspecting computation graphs, you want to see the logical structure—which operations feed into which others, where data flows. With low-level IR, this structure drowns in implementation mechanics. Counting loop nests to determine how many operations a function contains isn't feasible. High-level operations make structure explicit: seeing `nn.matmul`, `nn.relu`, `nn.matmul` tells you immediately what the network does. This clarity matters for both human inspection and automated analysis.

**Problem 2: Optimization Opportunities**. Optimizations that reason about domain semantics (like "this matrix multiply followed by ReLU can be fused" or "these two attention blocks can share Q/K/V projections") require recognizing patterns in the IR. With explicit loops, the pattern matching code must reconstruct the intent from implementation details—a difficult, fragile process. If operations retain high-level types like `nn.attention`, the optimizer sees intent directly and can apply domain-specific transformations confidently. The pattern "matmul → relu" is easy to detect when spelled `nn.matmul %x, %W` followed by `nn.relu %result`. It's much harder when spelled as thirty operations involving allocation, loops, comparisons, and stores.

**Problem 3: Flexibility in Lowering**. Different backends or optimization levels may want different lowerings of the same high-level operation. Matrix multiplication might lower to `linalg.matmul` for CPU, to `gpu.matmul` for GPUs, or to vendor-specific intrinsics on specialized accelerators. If your graph builder directly emits Linalg, you've committed to one lowering strategy. With a high-level `nn.matmul`, the lowering is a separate pass that chooses the appropriate implementation based on target, optimization flags, or tensor shapes. This flexibility is critical for portable AI compilers targeting diverse hardware.

**Problem 4: Multiple Compilation Paths**. Production ML systems often need different code generation strategies for different phases of model execution. Training might use one set of passes emphasizing numerical precision and gradient computation, while inference emphasizes latency and throughput. Serving systems might apply aggressive optimizations that would be unsafe during training. If high-level operations encode these intents explicitly (`nn.matmul` vs. `training.matmul_backward`), the compiler can route them to appropriate lowering strategies. Without custom dialects, you're forced to handle these differences outside MLIR, losing the benefits of a unified IR.

Custom dialects address all these issues by inserting a new abstraction level between user-facing APIs and standard MLIR dialects. The `nn` dialect captures neural network semantics directly, making IR readable, optimization patterns recognizable, and lowering strategies flexible. This chapter's implementation uses Python to generate `nn` dialect text, then implements lowering passes (also in Python) that convert `nn` operations to standard dialects. The approach is pedagogical—production systems use TableGen (Chapter 9) for dialect definition—but it teaches the core concepts clearly without code generation machinery obscuring the principles.

## 8.2 The String-Based Approach: Python for IR, C++ for Compilation

Before diving into implementation, let's understand the architecture choices for this chapter. We use a **string-based approach** where Python code generates MLIR text directly, and C++ parses that text for compilation and execution. This differs from Chapter 7, where C++ code built MLIR in-memory using OpBuilder APIs, and from Chapter 9, where TableGen generates C++ classes that build IR. Each approach has merits; the string-based method is excellent for teaching because it makes every IR construct explicit.

**Why Strings?** When you write `%0 = nn.add %arg0, %arg1 : tensor<4xf32>` in Python code and emit it as a string, you see exactly what gets parsed and compiled. There's no abstraction hiding details, no C++ types to wrestle with, no template errors to debug. If the IR is wrong, you see it immediately in the generated string. For learning dialect concepts—operation syntax, type constraints, custom attributes—this transparency is invaluable. You iterate quickly: edit Python code, print MLIR string, inspect what changed, repeat. The feedback loop is instant.

**Why Not Strings in Production?** Production systems rarely generate IR as text for several reasons. First, **error handling**: string manipulation doesn't catch type errors or malformed operations until runtime parsing, while OpBuilder APIs validate at compile time. Second, **performance**: parsing text is slower than constructing IR in-memory, though for model compilation (not inference), this overhead is negligible. Third, **tooling**: OpBuilder integrates with MLIR's verification infrastructure, catching invariant violations immediately, while string generation delays errors until parsing. Fourth, **composition**: combining IR fragments from multiple sources is tricky with strings (concatenation, scope management) but natural with OpBuilder (just create operations in the same block).

For our educational goals, these production concerns don't outweigh the learning benefits. We want to understand what dialects **are** before learning how to build them idiomatically. The string approach teaches dialect syntax, operation semantics, type systems, and lowering patterns without the cognitive load of C++ template metaprogramming or TableGen's declarative language. Once these concepts are clear, Chapter 9's TableGen approach will make sense as a productivity tool, not a mysterious code generator.

**Architecture: Division of Responsibilities**. Our system splits cleanly along Python/C++ boundaries. Python handles graph construction (the `Graph` class from Chapter 7, adapted for `nn` dialect), high-level IR generation (emitting `nn` operations as text), and lowering (converting `nn` text to standard MLIR text). C++ handles MLIR text parsing (using `mlir::parseSourceString`), compilation (the same pass pipeline as Chapter 7), and execution (the libffi-based universal executor from Chapter 7.13). This division exploits each language's strengths: Python for rapid prototyping and string manipulation, C++ for performance-critical compilation and execution.

The Python `Graph` class tracks operations using dataclasses—simple, readable data structures that store operation types, operands, and results. When generating IR, Python code walks the graph and emits MLIR strings using f-strings and basic formatting. The `MLIRLowering` class implements transformation logic entirely in Python, reading operations from the graph and outputting lowered MLIR text. This Python-side lowering is unusual compared to production MLIR (which uses C++ rewrite patterns), but it's pedagogically powerful: you see lowering logic directly, without pattern matchers or dialect interfaces obscuring the transformations.

On the C++ side, we use MLIR's public APIs but in the simplest possible way. The `parseMLIR()` function takes a string and returns a `ModuleOp`—no custom parsing logic, just the built-in text parser. The compilation pipeline uses the same passes as Chapter 7 (Linalg → Loops → LLVM), configured with the same `PassManager` setup. Execution uses the same libffi-based `execute()` function that marshals memref descriptors and dispatches dynamically. The C++ code is under 150 lines total, demonstrating how much heavy lifting MLIR's infrastructure provides. You focus on dialect semantics, not infrastructure.

## 8.3 The nn Dialect: Operations and Syntax

Our `nn` (neural network) dialect provides five core operations covering the functionality from Chapter 7's graph API. Each operation has a **syntax** (how it appears in MLIR text), **semantics** (what computation it represents), and **type constraints** (what tensor shapes and types it accepts). Let's examine each operation in detail, understanding both the MLIR representation and the design choices behind it.

**Element-Wise Addition**: `nn.add`

```mlir
%result = nn.add %lhs, %rhs : tensor<4xf32>
```

Semantics: Element-wise floating-point addition. For inputs of shape `[4]`, computes `result[i] = lhs[i] + rhs[i]` for all indices. The operation requires matching shapes for both operands and returns a tensor of the same shape. Type signature: `(tensor<...xf32>, tensor<...xf32>) -> tensor<...xf32>` where all shapes match.

The syntax follows MLIR conventions: operation name (`nn.add`), operands in SSA form (`%lhs`, `%rhs`), a colon separator, and the result type. We specify only the result type because operand types can be inferred (they must match the result). Production dialects often omit redundant type information, but for clarity, our initial implementation includes it. The operation has no side effects (does not modify global state or memory), is deterministic (same inputs always produce same outputs), and is parallelizable (each output element computed independently).

**Element-Wise Multiplication**: `nn.mul`

```mlir
%result = nn.mul %lhs, %rhs : tensor<4xf32>
```

Semantics: Element-wise floating-point multiplication. Identical to `nn.add` structurally, differing only in the scalar operation applied. This parallelism in operation design is intentional—element-wise operations share a pattern, making them candidates for a generic `nn.elementwise` operation with an attribute specifying the scalar function. For teaching, separate operations clarify intent; Chapter 9's TableGen approach will show how to define operation families generically.

**Matrix Multiplication**: `nn.matmul`

```mlir
%result = nn.matmul %lhs, %rhs : tensor<2x3xf32>, tensor<3x4xf32> -> tensor<2x4xf32>
```

Semantics: Matrix multiplication with standard linear algebra semantics. For `lhs` of shape `[M, K]` and `rhs` of shape `[K, N]`, computes `result[i,j] = sum_k lhs[i,k] * rhs[k,j]`, producing output shape `[M, N]`. The constraint that `lhs`'s second dimension must equal `rhs`'s first dimension is implicit in the type signature—if violated, compilation will fail during verification.

The syntax includes all three tensor types explicitly because they can't be inferred from each other (the shapes are related but distinct). This verbosity is typical for matrix operations: the compiler needs complete type information to generate efficient code, and explicit types prevent ambiguity during parsing. Note that we don't specify precision (float32 vs. float16) or layout (row-major vs. column-major)—these are implicit in the tensor type system and could be attributes if we needed flexibility.

**ReLU Activation**: `nn.relu`

```mlir
%result = nn.relu %input : tensor<2x4xf32>
```

Semantics: Rectified Linear Unit activation function. Computes `result[indices...] = max(0, input[indices...])` element-wise. This is a unary operation (one input, one output) with shapes matching. ReLU is ubiquitous in modern neural networks, serving as the default activation after linear layers in feed-forward networks, convolutional networks, and transformer MLPs.

The simplicity of `nn.relu` highlights why custom dialects matter: the operation encapsulates a computation pattern (element-wise comparison and selection) that's semantically distinct from addition or multiplication. Lowering can choose different implementation strategies—SCF loops with `arith.cmpf` and `arith.select` for CPU, or vendor intrinsics for GPUs—while the high-level operation remains unchanged. This flexibility would be impossible if we directly emitted implementation details in Chapter 7's style.

**Softmax Activation**: `nn.softmax`

```mlir
%result = nn.softmax %input : tensor<4xf32>
```

Semantics: Softmax probability normalization. From Chapter 6, we know softmax requires three passes: find maximum (for numerical stability), compute exponentials and sum, then normalize by the sum. Our `nn.softmax` operation encapsulates this multi-step computation as a single high-level operation, hiding implementation complexity. For inputs of shape `[N]`, computes `result[i] = exp(input[i] - max) / sum_j(exp(input[j] - max))`.

This operation demonstrates **semantic abstraction**: users don't need to know about three-pass algorithms, exponential sums, or numerical stability tricks. They just apply softmax where needed. Lowering handles the implementation details, potentially using different strategies for different backends (vectorized for CPU, fused kernels for GPU, table lookups for quantized inference). The high-level operation remains unchanged across these variations.

**Type System**. Our dialect uses MLIR's built-in `tensor` types with shape and element type. The choice of `tensor` vs. `memref` is significant: tensors represent immutable values in SSA form (functional style), while memrefs represent mutable buffers (imperative style). For high-level operations, tensors are more natural—we express data flow without worrying about buffer allocation or aliasing. Lowering to Linalg and eventually to memrefs happens later in the compilation pipeline, as the IR moves from abstract to concrete representations.

The shapes in tensor types (`<4xf32>`, `<2x3xf32>`) are **static**—known at compile time. Dynamic shapes (unknown dimensions written as `<?xf32>`) are possible but complicate our simple implementation. Production systems handle dynamic shapes throughout, adjusting code generation based on runtime size queries. For teaching, static shapes suffice: they clarify type checking without requiring shape inference infrastructure.

**Assembly Format**. The syntax we've defined is MLIR's **generic format**—operation name, operands, attributes, and result type separated by standard delimiters. MLIR supports **custom assembly formats** where operations can define their own pretty-printed syntax (e.g., `%result = matmul %lhs, %rhs : ...` without the `nn.` prefix). Chapter 9's TableGen approach will show how to specify custom formats declaratively. For now, generic format is sufficient and emphasizes that operations are just structured data (name + operands + types), not magic syntax.

This dialect—five operations, tensor types, straightforward semantics—provides everything needed to express the neural network computations from Chapter 7. The key difference: instead of generating fifty operations for a two-layer MLP, we generate seven `nn` operations, then lower them to fifty standard operations separately. The high-level IR stays readable; optimization passes can pattern-match easily; lowering strategies can vary by target. This is the power of custom dialects.

## 8.4 Graph Builder: Constructing nn Dialect IR

The `Graph` class in [python/graph_builder.py](../ch.8.Custom-dialect/python/graph_builder.py) provides a Python API for constructing computation graphs that emit `nn` dialect IR. This class is structurally similar to Chapter 7's graph builder—operations return symbolic values (operation IDs), which subsequent operations consume—but instead of generating standard MLIR immediately, it generates high-level `nn` operations. Let's examine the implementation to understand how symbolic execution in Python produces MLIR text.

**Data Structures**. The graph uses two simple dataclasses:

```python
@dataclass
class TensorValue:
    id: int              # SSA value ID (e.g., 0 for %0)
    shape: List[int]     # Tensor shape, e.g., [4] or [2, 3]

@dataclass
class Operation:
    op_type: str                # 'variable', 'add', 'mul', 'matmul', 'relu', 'softmax'
    operands: List[TensorValue] # Input values
    result: TensorValue         # Output value
    attrs: Dict[str, Any]       # Additional attributes (unused in our simple dialect)
```

A `TensorValue` represents an SSA value—the `%0`, `%1`, `%2` values in MLIR that hold intermediate results. The `id` field is the numeric suffix, and `shape` tracks the tensor dimensions. Python's simplicity shines here: no C++ template complexity, no inheritance hierarchies, just data. An `Operation` records what computation to perform (`op_type`), what inputs it consumes (`operands`), what output it produces (`result`), and any extra information (`attrs`). The structure mirrors MLIR's operation concept directly.

The `Graph` class maintains a list of operations and a counter for generating unique IDs:

```python
class Graph:
    def __init__(self):
        self.next_id = 0
        self.operations: List[Operation] = []
        self.variables: List[TensorValue] = []
```

As operations are added, they append to `operations`, and variables (inputs) are tracked separately in `variables`. This separation matters for IR generation: variables become function parameters, while other operations become function body.

**Adding Operations**. Each operation type has a builder method:

```python
def add(self, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    if lhs.shape != rhs.shape:
        raise ValueError(f"Shape mismatch: {lhs.shape} vs {rhs.shape}")
    result = self._new_value(lhs.shape)
    op = Operation('add', [lhs, rhs], result, {})
    self.operations.append(op)
    return result
```

The method validates inputs (shapes must match for element-wise addition), creates a new `TensorValue` for the result with the same shape, constructs an `Operation` recording the computation, appends it to the operations list, and returns the result value. The caller receives a `TensorValue` object they can pass to subsequent operations, building the dependency graph implicitly through Python object references.

This deferred execution pattern matches Chapter 7 and mirrors PyTorch's JIT or TensorFlow's tracing: operations don't execute when called, they record what should execute later. The `add()` method doesn't add numbers; it records that an addition should happen. Only when `get_mlir()` is called does the graph materialize into executable IR.

**Matrix Multiplication** enforces additional constraints:

```python
def matmul(self, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        raise ValueError(f"matmul requires 2D tensors, got {lhs.shape} and {rhs.shape}")
    M, K1 = lhs.shape
    K2, N = rhs.shape
    if K1 != K2:
        raise ValueError(f"matmul dimension mismatch: {lhs.shape} @ {rhs.shape}")
    result = self._new_value([M, N])
    op = Operation('matmul', [lhs, rhs], result, {})
    self.operations.append(op)
    return result
```

The method checks that inputs are 2D, verifies the inner dimensions match (`K1 == K2`), computes the output shape `[M, N]`, and records the operation. These Python-side validations catch errors early—at graph construction time, not during MLIR compilation. This immediate feedback aids debugging: if you accidentally try to multiply incompatible matrices, Python raises `ValueError` with a clear message before any MLIR is generated.

**Generating MLIR Text**. The `get_mlir()` method walks the graph and emits MLIR strings:

```python
def get_mlir(self, output: TensorValue, func_name: str = "main") -> str:
    lines = []
    lines.append("module {")

    # Function signature
    input_types = [self._format_shape(v.shape) for v in self.variables]
    output_type = self._format_shape(output.shape)

    func_sig = f"  func.func @{func_name}("
    for i, (var, type_str) in enumerate(zip(self.variables, input_types)):
        if i > 0:
            func_sig += ", "
        func_sig += f"{var} : {type_str}"
    func_sig += f") -> {output_type} {{"
    lines.append(func_sig)

    # Operations
    for op in self.operations:
        if op.op_type == 'variable':
            continue  # Variables are function parameters
        # ... emit operation string ...

    lines.append(f"    return {output} : {output_type}")
    lines.append("  }")
    lines.append("}")
    return '\n'.join(lines)
```

The method builds a `module` containing a single `func.func`. Variables become function parameters (not separate operations), and operations become the function body. The output value (the final computation result) is returned. This structure matches standard MLIR patterns for executable functions: a module wraps functions, functions take inputs and return outputs, operations transform inputs to outputs.

**Operation Emission**. Each operation type generates its syntax:

```python
if op.op_type == 'add':
    line = f"    {op.result} = nn.add {op.operands[0]}, {op.operands[1]} : {self._format_shape(op.result.shape)}"
elif op.op_type == 'matmul':
    lhs_type = self._format_shape(op.operands[0].shape)
    rhs_type = self._format_shape(op.operands[1].shape)
    out_type = self._format_shape(op.result.shape)
    line = f"    {op.result} = nn.matmul {op.operands[0]}, {op.operands[1]} : {lhs_type}, {rhs_type} -> {out_type}"
```

Python f-strings make this trivial: interpolate SSA values (`{op.result}`), operands, and types into strings following MLIR syntax. The `_format_shape()` helper converts Python lists like `[2, 3]` into MLIR tensor types like `tensor<2x3xf32>`. The simplicity of this approach—constructing strings directly—removes all abstraction layers between "what we want" and "what MLIR sees." For learning, this transparency is invaluable.

**Example Usage**:

```python
g = Graph()
x = g.variable([4])
y = g.variable([4])
z = g.add(x, y)
mlir_text = g.get_mlir(z, "add_vectors")
```

Produces:

```mlir
module {
  func.func @add_vectors(%0 : tensor<4xf32>, %1 : tensor<4xf32>) -> tensor<4xf32> {
    %2 = nn.add %0, %1 : tensor<4xf32>
    return %2 : tensor<4xf32>
  }
}
```

Clean, readable, high-level. The computation's intent is immediately clear: this function adds two 4-element vectors. Contrast with Chapter 7's output for the same operation: allocation, loop bounds, index calculations, loads, arithmetic, stores. The `nn` dialect raises the abstraction level exactly where needed.

This graph builder establishes our high-level IR generation. Next, we'll implement lowering: transforming this readable `nn` dialect IR into executable standard MLIR.

## 8.5 Lowering nn Dialect to Standard MLIR

The `MLIRLowering` class in [python/lowering.py](../ch.8.Custom-dialect/python/lowering.py) implements transformation logic that converts high-level `nn` operations into standard MLIR using Linalg, Arith, and MemRef dialects. This lowering happens entirely in Python through string manipulation—a pedagogical choice that makes transformation logic explicit without requiring familiarity with MLIR's C++ pattern rewriting infrastructure. Let's examine each lowering pattern to understand how high-level semantics map to implementation details.

**Why Lower to Standard Dialects?** MLIR provides no execution engine for custom dialects directly. To JIT compile and run our `nn` operations, we must convert them to dialects that have lowering paths to LLVM IR. The Linalg dialect provides high-level structured operations (matmul, generic), Arith provides scalar arithmetic (addf, mulf, maximumf), and MemRef provides buffer management (alloc, load, store). Together, these dialects express any computational pattern, and MLIR's built-in passes know how to lower them progressively to executable code. Our lowering transforms `nn` operations into combinations of these standard operations, leveraging existing infrastructure rather than implementing code generation from scratch.

**The Lowering Architecture**. The `MLIRLowering` class walks a computation graph and generates MLIR text where each `nn` operation is replaced with its standard dialect equivalent. The API is straightforward:

```python
lowering = MLIRLowering()
standard_mlir = lowering.lower_graph(graph, output_value, "function_name")
```

The returned string contains only standard MLIR dialects that the C++ compiler understands. Internally, `lower_graph()` emits a function signature (variables become `memref` parameters, not `tensor`), iterates through operations calling type-specific lowering methods, and generates the return statement. The critical insight: lowering is a **source-to-source transformation** operating on IR text, not a runtime transformation operating on executing code.

**Type Conversion: Tensor to MemRef**. The first lowering decision is type representation. Our high-level `nn` dialect uses `tensor` types (immutable, SSA values), but for execution, we need `memref` types (mutable buffers). The conversion is straightforward: `tensor<4xf32>` becomes `memref<4xf32>`, preserving shape and element type. This reflects a fundamental MLIR pattern called **bufferization**—converting value semantics to buffer semantics. Our lowering performs this conversion implicitly during function signature generation.

The function signature changes from:

```mlir
func.func @add(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> tensor<4xf32>
```

to:

```mlir
func.func @add(%arg0 : memref<4xf32>, %arg1 : memref<4xf32>, %out : memref<4xf32>)
```

Notice we also switch from return values to out-parameters. The `%out` buffer is provided by the caller (Python via execute()), and we write results directly into it. This out-parameter pattern matches C/C++ conventions and simplifies the calling convention—no need to marshal return values across the FFI boundary, just write to a pre-allocated buffer.

## 8.6 Element-Wise Operations: Lowering with Linalg.Generic

Element-wise operations (addition, multiplication, ReLU) share a common pattern: apply a scalar operation independently to each element of input tensors. Linalg's `linalg.generic` operation expresses this pattern beautifully, providing a structured way to generate element-wise computations without manual loop writing. Let's examine how `nn.add` lowers to `linalg.generic` with `arith.addf`.

**The Pattern**. For `nn.add %lhs, %rhs : tensor<4xf32>`, we generate:

```mlir
%result = memref.alloc() : memref<4xf32>
linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>,
                   affine_map<(d0) -> (d0)>,
                   affine_map<(d0) -> (d0)>],
  iterator_types = ["parallel"]
} ins(%lhs, %rhs : memref<4xf32>, memref<4xf32>)
  outs(%result : memref<4xf32>) {
^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
  %sum = arith.addf %arg0, %arg1 : f32
  linalg.yield %sum : f32
}
```

This MLIR deserves careful examination. The `linalg.generic` operation takes three parameters: `ins` (inputs to read), `outs` (outputs to write), and a region (code block) specifying the scalar computation. The **indexing maps** describe how iteration indices map to tensor positions. For 1D tensors, `affine_map<(d0) -> (d0)>` means "iteration index 0 directly indexes dimension 0"—a simple identity mapping. For 2D tensors, we'd use `affine_map<(d0, d1) -> (d0, d1)>` for both inputs and outputs, indicating element-wise operation across all dimensions.

The **iterator types** specify parallelism: `["parallel"]` means iterations don't have dependencies (output element `i` doesn't depend on output element `j`), allowing vectorization and parallel execution. For reductions (like sum or max over a dimension), we'd use `["reduction"]` for the reducing iterator. For our element-wise operations, all iterators are parallel, giving optimizers maximum freedom.

**The Region Block**. Inside the curly braces, we define the scalar computation. The block signature `^bb0(%arg0: f32, %arg1: f32, %arg2: f32)` declares three scalar arguments: two inputs from `ins`, one output from `outs`. These are scalar values (type `f32`, not `memref<...xf32>`), extracted automatically by `linalg.generic` at each iteration point. Our scalar computation is simple: `%sum = arith.addf %arg0, %arg1`, adding the input scalars. The `linalg.yield %sum` writes the result to the output buffer at the current iteration index. Linalg handles all loop generation, index calculation, and buffer access—we just specify the scalar operation.

**Python Implementation**:

```python
def lower_add(self, result_id: int, lhs_id: int, rhs_id: int, shape: List[int]) -> List[str]:
    lines = []
    ind = self._indent()
    memref_type = self._tensor_to_memref(shape)

    rank = len(shape)
    if rank == 1:
        indexing = "affine_map<(d0) -> (d0)>"
        iterator = '["parallel"]'
    elif rank == 2:
        indexing = "affine_map<(d0, d1) -> (d0, d1)>"
        iterator = '["parallel", "parallel"]'

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

The code constructs MLIR strings programmatically. We detect the tensor rank (1D or 2D) and choose appropriate indexing maps and iterator types. The method returns a list of strings that get joined with newlines—a simple way to build multi-line MLIR text with proper indentation. Multiplication (`nn.mul`) follows an identical pattern, substituting `arith.mulf` for `arith.addf`. ReLU uses `arith.maximumf %input, %zero` instead of addition.

**Why Linalg.Generic?** We could lower element-wise operations to explicit `scf.for` loops like Chapter 7 did. The advantage of `linalg.generic` is **structured operations** enable better optimization. MLIR's Linalg transformations know how to fuse multiple `linalg.generic` operations (combine loop nests), tile them (blocking for cache locality), and vectorize them (SIMD instructions). These optimizations rely on recognizing the structured pattern. Explicit loops are harder to analyze—the optimizer must reconstruct the pattern from imperative code. By keeping operations in Linalg form as long as possible, we give downstream passes maximum optimization opportunity.

**2D Generalization**. The same lowering works for 2D tensors with minimal changes:

```python
if rank == 2:
    indexing = "affine_map<(d0, d1) -> (d0, d1)>"
    iterator = '["parallel", "parallel"]'
```

Now `d0` represents the row dimension, `d1` the column dimension, and both are parallel. The generated `linalg.generic` iterates over all (row, column) pairs, applying the scalar addition. Linalg handles loop nest generation, bound calculations, and memory access automatically. This generalization demonstrates a key strength of structured operations: the same operation definition scales across ranks without code duplication.

**Scalar Constants**. Operations like ReLU need constants (zero for comparison). We emit these in the function prologue:

```mlir
%cst_zero = arith.constant 0.000000e+00 : f32
```

Then reference `%cst_zero` in the `arith.maximumf` operation. Constants are hoisted outside loops, computed once, and reused across iterations—a standard optimization that MLIR applies automatically.

Element-wise lowering demonstrates how high-level operations decompose into combinations of standard dialect operations. One `nn.add` becomes one allocation, one `linalg.generic`, and several internal operations. The verbosity has purpose: explicit representation enables optimization and analysis. Next, we'll examine matrix multiplication lowering, which uses a single structured Linalg operation rather than a generic pattern.

## 8.7 Matrix Multiplication and ReLU Lowering

Matrix multiplication represents a different pattern: not element-wise, but a **reduction** over a shared dimension. While element-wise operations use `linalg.generic` with parallel iterators, matmul uses the specialized `linalg.matmul` operation that encapsulates three-loop reduction semantics. Let's examine how `nn.matmul` lowers, then cover ReLU's simpler pattern.

**Matrix Multiplication with Linalg.Matmul**. For `nn.matmul %lhs, %rhs : tensor<2x3xf32>, tensor<3x4xf32> -> tensor<2x4xf32>`, we generate:

```mlir
%result = memref.alloc() : memref<2x4xf32>
%cst_zero = arith.constant 0.000000e+00 : f32
linalg.fill ins(%cst_zero : f32) outs(%result : memref<2x4xf32>)
linalg.matmul ins(%lhs, %rhs : memref<2x3xf32>, memref<3x4xf32>)
              outs(%result : memref<2x4xf32>)
```

Three operations accomplish the lowering. First, **allocation**: `memref.alloc()` allocates a 2×4 buffer for results. This happens at runtime (dynamic allocation), though for static shapes, MLIR could optimize to stack allocation (`memref.alloca`) or buffer reuse. Second, **initialization**: `linalg.fill` writes zero to every element of the result buffer. This is necessary because `linalg.matmul` accumulates into the output—it computes `result += lhs × rhs`, not `result = lhs × rhs`. Initializing to zero gives us pure multiplication. Third, **computation**: `linalg.matmul` performs the actual matrix multiply, reading from `lhs` and `rhs`, writing accumulated results to `result`.

**Why Separate Fill?** The accumulation semantics of `linalg.matmul` might seem awkward compared to a pure functional operation. The design choice reflects optimization opportunities: if you're computing `C = A × B + C` (common in iterative algorithms), the `outs` parameter can be your existing `C` matrix, and matmul adds directly without temporary allocation. For simple multiplication, we pay one initialization operation (`linalg.fill`), but gain the flexibility to express fused operations. This is a general Linalg pattern: output buffers are inputs and outputs simultaneously, enabling in-place updates and fusion.

**The Matmul Semantics**. Internally, `linalg.matmul` represents:

```python
for i in range(M):
    for j in range(N):
        for k in range(K):
            result[i, j] += lhs[i, k] * rhs[k, j]
```

Three nested loops: outer two parallel (over output rows and columns), inner one reduction (summing over the shared dimension `K`). The operation encapsulates this pattern without spelling out loops explicitly, allowing optimizations like loop tiling, permutation, and replacement with vendor BLAS libraries. When lowering to executable code, `convert-linalg-to-loops` pass expands matmul to SCF loops, which then lower to LLVM. But at this level, we keep the structured operation for maximum optimization potential.

**Python Implementation**:

```python
def lower_matmul(self, result_id: int, lhs_id: int, rhs_id: int,
                 lhs_shape: List[int], rhs_shape: List[int], result_shape: List[int]) -> List[str]:
    lines = []
    ind = self._indent()

    lhs_type = self._tensor_to_memref(lhs_shape)
    rhs_type = self._tensor_to_memref(rhs_shape)
    result_type = self._tensor_to_memref(result_shape)

    lines.append(f"{ind}%{result_id} = memref.alloc() : {result_type}")
    lines.append(f"{ind}linalg.fill ins(%cst_zero : f32) outs(%{result_id} : {result_type})")
    lines.append(f"{ind}linalg.matmul ins(%{lhs_id}, %{rhs_id} : {lhs_type}, {rhs_type})")
    lines.append(f"{ind}                outs(%{result_id} : {result_type})")

    return lines
```

Simple string formatting: allocate result, fill with zero, perform matmul. The types (`lhs_type`, `rhs_type`, `result_type`) are computed from the operation's shape information tracked during graph construction. Notice we reference `%cst_zero`, the constant emitted in the function prologue—MLIR allows forward references within a function, so the constant definition can appear after operations that use it in the source text (the parser and verifier handle ordering).

**ReLU Lowering**. ReLU uses `linalg.generic` like addition but with `arith.maximumf` instead of `arith.addf`:

```mlir
%result = memref.alloc() : memref<2x4xf32>
linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                   affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]
} ins(%input : memref<2x4xf32>)
  outs(%result : memref<2x4xf32>) {
^bb0(%arg0: f32, %arg1: f32):
  %relu_val = arith.maximumf %arg0, %cst_zero : f32
  linalg.yield %relu_val : f32
}
```

The operation has **one** input and one output (unary operation). The region block receives two arguments: `%arg0` from the input tensor, `%arg1` from the output (unused here). We compute `max(%arg0, 0)` and yield the result. The unary pattern is common for activations, normalizations, and other element-wise transformations. Linalg's generality handles binary, unary, or even ternary operations uniformly—adjust the `ins` and `outs` lists, and the region signature changes accordingly.

**Softmax Lowering**. Softmax is more complex, requiring a three-pass algorithm for numerical stability: (1) find max, (2) compute exp(x-max) and sum, (3) normalize. We implement this by chaining `linalg.generic` operations with reduction iterators.

For the reduction steps (finding max and sum), we use `linalg.generic` with `iterator_types = ["reduction"]`. This tells the compiler that the operation reduces dimensions. For example, finding the maximum value of a 1D tensor:

```mlir
linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
  iterator_types = ["reduction"]
} ins(%input : memref<4xf32>)
  outs(%max_alloc : memref<f32>) {
^bb0(%in: f32, %acc: f32):
  %max = arith.maximumf %in, %acc : f32
  linalg.yield %max : f32
}
```

The indexing maps show that the input is accessed by index `d0`, while the output (scalar accumulator) is accessed by `()`. The reduction iterator implies that `d0` is the reduction dimension. Inside the region, `%acc` holds the running maximum, which we update with `arith.maximumf`.

This pattern—chaining structured operations—is powerful. We build complex algorithms like Softmax from simple, composable primitives (generic, map, reduce) without ever writing a loop.

**Comparison with Chapter 7**. In Chapter 7, we generated ReLU using explicit `scf.for` loops, `memref.load`, `arith.cmpf`, `arith.select`, and `memref.store`. The generated IR involved multiple operations for 2D ReLU. Here, one `linalg.generic` expresses the same computation at a higher level. The difference isn't just verbosity—it's analyzability. Pattern matchers can detect "matmul followed by relu" when both are structured operations. With explicit loops, the pattern is buried in imperative code. Optimizers can fuse the two `linalg.generic` operations (matmul and relu), eliminating the intermediate buffer write. With explicit loops, fusion requires complex dependence analysis.

This demonstrates the custom dialect's value proposition: operations remain at appropriate abstraction levels through multiple compilation stages. High-level `nn` dialect captures user intent, mid-level Linalg captures computational patterns, low-level SCF captures control flow, and LLVM captures machine instructions. Each level enables different optimizations and analyses. MLIR's multi-level design isn't an abstraction for abstraction's sake—it's a tool for managing compilation complexity by separating concerns across layers.

## 8.8 Putting It Together: Complete Lowering Pipeline

The `lower_graph()` method orchestrates the complete transformation from `nn` dialect to standard MLIR. Let's walk through the full pipeline, understanding how individual operation lowerings compose into a complete, executable function. This method demonstrates how source-to-source transformations work in practice: inspect high-level operations, emit low-level equivalents, handle dependencies, and produce semantically equivalent IR.

**Function Signature Transformation**. The high-level `nn` function signature:

```mlir
func.func @mlp(%0 : tensor<2x3xf32>, %1 : tensor<3x4xf32>, %2 : tensor<4x2xf32>) 
    -> tensor<2x2xf32>
```

Becomes:

```mlir
func.func @mlp(%0 : memref<2x3xf32>, %1 : memref<3x4xf32>, %2 : memref<4x2xf32>, 
               %out : memref<2x2xf32>)
```

The transformation: (1) convert all `tensor` types to `memref`, (2) remove the return type, (3) add an `%out` parameter for the result. The Python code:

```python
input_types = [self._tensor_to_memref(v.shape) for v in graph.variables]
output_type = self._tensor_to_memref(output.shape)

func_sig = f"  func.func @{func_name}("
for i, (var, type_str) in enumerate(zip(graph.variables, input_types)):
    func_sig += f"%{var.id} : {type_str}, "
func_sig += f"%out : {output_type}) {{"
```

Variables from the graph become parameters with memref types. The output is appended as the final parameter. This signature matches our calling convention from Chapter 7: inputs as `memref` arguments (NumPy arrays marshaled via libffi), output pre-allocated by Python and passed as an out-parameter.

**Operation Lowering Loop**. We iterate through graph operations, lowering each to standard MLIR:

```python
for op in graph.operations:
    if op.op_type == 'variable':
        continue  # Variables are function parameters, not operations

    if op.op_type == 'add':
        op_lines = self.lower_add(op.result.id, op.operands[0].id, op.operands[1].id, op.result.shape)
    elif op.op_type == 'mul':
        op_lines = self.lower_mul(...)
    elif op.op_type == 'matmul':
        op_lines = self.lower_matmul(...)
    elif op.op_type == 'relu':
        op_lines = self.lower_relu(...)

    lines.extend(op_lines)
```

Each operation's lowering method returns a list of MLIR text lines. We accumulate these lines in order, building the function body progressively. The operations maintain their dependency order from the original graph—if operation 5 depends on operation 3, operation 3 appears first in the iteration. This topological ordering (implicit in our graph construction) ensures SSA values are defined before use.

**Copy to Output Parameter**. After all operations, we copy the final result to the output buffer:

```mlir
memref.copy %4, %out : memref<2x2xf32> to memref<2x2xf32>
```

The `memref.copy` operation performs an element-wise copy from the source (our computed result `%4`) to the destination (the output parameter `%out`). This operation is necessary because intermediate results are allocated locally, but the caller expects results in their provided buffer. The copy is explicit in the IR, allowing optimizers to eliminate it if possible (e.g., by having the final operation write directly to `%out`). For now, we keep it simple: compute results to local allocations, copy at the end.

**Return Statement**. The function ends with `return` (no value—we wrote to the out-parameter). The complete lowered function:

```mlir
module {
  func.func @mlp(%0 : memref<2x3xf32>, %1 : memref<3x4xf32>, %2 : memref<4x2xf32>,
                 %out : memref<2x2xf32>) {
    %cst_zero = arith.constant 0.000000e+00 : f32

    // First matmul: %3 = %0 @ %1
    %3 = memref.alloc() : memref<2x4xf32>
    linalg.fill ins(%cst_zero : f32) outs(%3 : memref<2x4xf32>)
    linalg.matmul ins(%0, %1 : memref<2x3xf32>, memref<3x4xf32>)
                  outs(%3 : memref<2x4xf32>)

    // ReLU: %4 = relu(%3)
    %4 = memref.alloc() : memref<2x4xf32>
    linalg.generic { ... } ins(%3) outs(%4) { ... }

    // Second matmul: %5 = %4 @ %2
    %5 = memref.alloc() : memref<2x2xf32>
    linalg.fill ins(%cst_zero : f32) outs(%5 : memref<2x2xf32>)
    linalg.matmul ins(%4, %2 : memref<2x4xf32>, memref<4x2xf32>)
                  outs(%5 : memref<2x2xf32>)

    // Copy to output
    memref.copy %5, %out : memref<2x2xf32> to memref<2x2xf32>
    return
  }
}
```

This IR is ready for standard MLIR compilation: Linalg operations will lower to loops, loops to control flow, control flow to LLVM, LLVM to machine code. The transformation from high-level `nn` dialect (5 operations: 2 matmuls, 1 relu, 2 variables) to mid-level standard dialects (15+ operations) happens entirely in Python string manipulation. No C++ required, no compiler infrastructure beyond basic string formatting.

This completes the lowering pipeline: high-level `nn` operations become standard Linalg, Arith, and MemRef operations through simple Python code. Next, we'll implement the C++ side: parsing this MLIR text, compiling it, and executing it using the techniques from Chapter 7.

## 8.9 C++ Compilation: Parsing and Pass Pipeline

The C++ side of our system handles three responsibilities: parsing MLIR text into in-memory IR, applying lowering passes to convert standard dialects to LLVM, and JIT compiling the result to executable machine code. This differs from Chapter 7, where we generated MLIR directly in C++ using OpBuilder APIs. Here, Python generates MLIR text strings, and C++ parses them. The implementation in [src/compiler.cpp](../ch.8.Custom-dialect/src/compiler.cpp) demonstrates how to work with MLIR text rather than direct IR construction.

**Parsing MLIR Text**. The key new API is `parseSourceString`:

```cpp
OwningOpRef<ModuleOp> parseMLIR(const std::string& mlirText) {
    mlir::ParserConfig config(context);
    return mlir::parseSourceString<ModuleOp>(mlirText, config);
}
```

This single function call uses MLIR's built-in text parser. The `parseSourceString` template function takes MLIR text as a string, parses it according to MLIR syntax rules, verifies the IR (checking type correctness, SSA form, operation constraints), and returns a `ModuleOp`. If parsing fails, it returns `nullptr` and prints diagnostics. This is essential for our string-based approach—Python generates text, C++ parses it back into structured IR.

**Pass Pipeline**. Once parsed, we apply the same lowering passes as any MLIR program:

```cpp
void runPasses(ModuleOp module) {
    PassManager pm(context);
    
    // Lower Linalg to loops
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
    
    // Lower SCF to Control Flow
    pm.addPass(createSCFToControlFlowPass());
    
    // Lower everything to LLVM dialect
    pm.addPass(createConvertMathToLibmPass());
    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createConvertSCFToControlFlowPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    
    if (failed(pm.run(module))) {
        throw std::runtime_error("Pass pipeline failed");
    }
}
```

The pipeline progressively lowers IR through dialect hierarchies: linalg→loops (scf.for), structured control flow→branches (cf.br), math operations→libm, then everything→LLVM dialect. These are the standard MLIR lowering passes documented in MLIR's dialect conversion documentation.

**ExecutionEngine: JIT Compilation**. MLIR provides `ExecutionEngine`, a wrapper around LLVM's JIT compiler:

```cpp
void* compileAndGetFunctionPtr(ModuleOp module, const std::string& funcName) {
    runPasses(module);  // Lower to LLVM dialect
    
    // Create ExecutionEngine with transformer for LLVM optimization
    auto transformer = mlir::makeOptimizingTransformer(
        /*optLevel=*/3,
        /*sizeLevel=*/0,
        /*targetMachine=*/nullptr
    );
    
    auto maybeEngine = mlir::ExecutionEngine::create(
        module,
        /*llvmModuleBuilder=*/nullptr,
        transformer
    );
    
    if (!maybeEngine) {
        throw std::runtime_error("Failed to create ExecutionEngine");
    }
    
    engine = std::move(*maybeEngine);
    
    // Lookup function pointer
    auto expectedPtr = engine->lookupPacked(funcName);
    if (!expectedPtr) {
        throw std::runtime_error("Function not found: " + funcName);
    }
    
    return reinterpret_cast<void*>(*expectedPtr);
}
```

The `ExecutionEngine::create()` method takes our LLVM-dialect module, translates it to LLVM IR, applies LLVM optimizations (controlled by `makeOptimizingTransformer`), and JIT compiles to native machine code. The `optLevel=3` enables aggressive optimization (inlining, vectorization, loop unrolling, etc.). The result is an execution engine containing compiled code for all functions in our module.

The `lookupPacked()` method retrieves a function pointer by name. MLIR's "packed" calling convention means the function signature matches what we generated: memref arguments as pointer/size/stride parameters. This pointer is what we'll call via libffi. The `void*` type-erasure is intentional—we don't know the signature at C++ compile time (it depends on runtime shapes), so we treat all functions as `void*` and use libffi to dispatch correctly.

**Initialization and State Management**. The `Compiler` class wraps this functionality with initialization:

```cpp
class Compiler {
public:
    Compiler() {
        // Initialize LLVM (must happen exactly once)
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        
        // Register all necessary dialects
        context.loadDialect<
            func::FuncDialect,
            arith::ArithDialect,
            memref::MemRefDialect,
            linalg::LinalgDialect,
            scf::SCFDialect,
            math::MathDialect,
            LLVM::LLVMDialect
        >();
        
        // Register translation from LLVM dialect to LLVM IR
        mlir::registerBuiltinDialectTranslation(context);
        mlir::registerLLVMDialectTranslation(context);
    }
    
private:
    MLIRContext context;
    std::unique_ptr<ExecutionEngine> engine;
};
```

Dialect registration is crucial: MLIR needs to know about all dialects our IR might contain (loaded from Python-generated text) and how to translate them. The `context.loadDialect<...>()` call loads dialect definitions—operation types, type systems, verification rules. Without registration, parsing would fail with "unknown operation" errors.

The LLVM initialization (`InitializeNativeTarget()`) configures target-specific code generation (x86, ARM, etc.). The translation registration (`registerLLVMDialectTranslation`) teaches MLIR how to convert LLVM dialect operations to LLVM IR primitives. These initialization steps happen once per process; the C++ singleton pattern ensures correctness even if multiple Python threads use the compiler.

**Error Handling**. All MLIR operations return results indicating success or failure. We check these and throw C++ exceptions, which pybind11 translates to Python exceptions. This gives Python code clear error messages:

```python
try:
    result = ch8.execute(mlir_text, "bad_function", inputs, shape)
except RuntimeError as e:
    print(f"Compilation failed: {e}")
```

The error message might indicate parsing failure (invalid MLIR syntax), pass failure (transformation error), or lookup failure (function name doesn't exist). For teaching, explicit error handling is valuable—students see exactly where and why compilation fails, not just "something went wrong."

This C++ implementation—150 lines managing parsing, passes, and compilation—demonstrates MLIR's power as an infrastructure. We didn't write a parser, implement lowering algorithms, or build a JIT compiler. We configured existing components, trusting the framework to handle complexity correctly and efficiently. This division of labor—Python for high-level logic, MLIR for compilation infrastructure—enables rapid development of AI compilers without reimplementing compiler fundamentals.

## 8.10 Universal Execution with libffi

The execution layer in [src/bindings.cpp](../ch.8.Custom-dialect/src/bindings.cpp) implements the universal `execute()` function that calls JIT-compiled functions with arbitrary signatures using libffi. As covered in detail in Chapter 7.13, libffi provides dynamic dispatch for functions with any signature, solving the binding explosion problem elegantly. Chapter 8 uses the same technique—the implementation follows the pattern established in Chapter 7, with memref marshaling helpers that convert NumPy arrays to MLIR's calling convention, and `ffi_call()` to execute the compiled function.

The key points from Chapter 7.13 apply here: MLIR's memref convention uses only pointer-sized values, making every argument `ffi_type_pointer` in libffi's terms. This uniformity enables one universal execution function to handle any shape combination, any number of inputs, and any operation type without code changes. The overhead is negligible for ML workloads—a few dozen CPU cycles for dispatch versus millions of operations for actual computation.

For the complete explanation of how libffi works, memref marshaling details, and performance characteristics, see Chapter 7.13. The implementation here is identical in principle, adapted for our string-based custom dialect workflow.

## 8.11 Testing: Verifying Graph-to-Code Correctness

Testing the string-based custom dialect implementation requires verifying that Python-generated MLIR text produces correct numerical results. The [test_jit.py](../ch.8.Custom-dialect/test_jit.py) test suite demonstrates this verification process, comparing our compiled functions against NumPy's reference implementations. Understanding these tests shows how to validate custom dialects and lowering logic systematically.

## 8.12 Composing Operations: Building Multi-Layer Networks

With our custom dialect and compilation infrastructure complete, let's examine how operations compose into larger computations. The [test_jit.py](../ch.8.Custom-dialect/test_jit.py) test suite demonstrates this progression from simple operations to multi-layer neural networks. Understanding these examples shows how the abstractions scale from primitives to production-sized models.

**Two-Layer MLP**. The test suite's most complex example builds a two-layer multi-layer perceptron:

```python
g = Graph()
x = g.variable([2, 3])      # Input: 2 samples, 3 features
W1 = g.variable([3, 4])     # Layer 1 weights
W2 = g.variable([4, 2])     # Layer 2 weights

h = g.matmul(x, W1)         # Hidden layer: [2, 3] × [3, 4] → [2, 4]
h_relu = g.relu(h)          # Activation
y = g.matmul(h_relu, W2)    # Output: [2, 4] × [4, 2] → [2, 2]

lowering = MLIRLowering()
mlir_text = lowering.lower_graph(g, y, "mlp")

result = ch8.execute(mlir_text, "mlp", [x_data, W1_data, W2_data], (2, 2))
```

Five high-level operations (three variables, two matmuls, one relu) express a complete neural network forward pass. The graph builder tracks dependencies implicitly—`h` depends on `x` and `W1`, `h_relu` depends on `h`, `y` depends on `h_relu` and `W2`. Lowering generates approximately 50 standard MLIR operations (allocations, loops, fills, arithmetic), but that complexity is hidden from users.

**Generated High-Level IR**. Before lowering, the `nn` dialect representation is:

```mlir
func.func @mlp(%0 : tensor<2x3xf32>, %1 : tensor<3x4xf32>, %2 : tensor<4x2xf32>) 
    -> tensor<2x2xf32> {
  %3 = nn.matmul %0, %1 : tensor<2x3xf32>, tensor<3x4xf32> -> tensor<2x4xf32>
  %4 = nn.relu %3 : tensor<2x4xf32>
  %5 = nn.matmul %4, %2 : tensor<2x4xf32>, tensor<4x2xf32> -> tensor<2x2xf32>
  return %5 : tensor<2x2xf32>
}
```

Clean, concise, intent-clear. Anyone reading this IR understands the computation immediately: two matrix multiplications with ReLU activation between them. Contrast with explicit loop IR from Chapter 7—you'd see loop nest construction, index calculations, and buffer operations, obscuring the network structure. The custom dialect's value is evident: appropriate abstractions make complex computations comprehensible.

**After Lowering**. The lowered standard MLIR (abridged for clarity):

```mlir
func.func @mlp(%0 : memref<2x3xf32>, %1 : memref<3x4xf32>, %2 : memref<4x2xf32>,
               %out : memref<2x2xf32>) {
  %cst_zero = arith.constant 0.000000e+00 : f32

  // First matmul
  %3 = memref.alloc() : memref<2x4xf32>
  linalg.fill ins(%cst_zero : f32) outs(%3 : memref<2x4xf32>)
  linalg.matmul ins(%0, %1 : ...) outs(%3 : ...)

  // ReLU
  %4 = memref.alloc() : memref<2x4xf32>
  linalg.generic { ... } ins(%3) outs(%4) {
  ^bb0(%a: f32, %b: f32):
    %relu_val = arith.maximumf %a, %cst_zero : f32
    linalg.yield %relu_val : f32
  }

  // Second matmul
  %5 = memref.alloc() : memref<2x2xf32>
  linalg.fill ins(%cst_zero : f32) outs(%5 : ...)
  linalg.matmul ins(%4, %2 : ...) outs(%5 : ...)

  // Copy to output
  memref.copy %5, %out : memref<2x2xf32> to memref<2x2xf32>
  return
}
```

The lowering preserved operation structure: three major computation stages (matmul, relu, matmul) remain identifiable even in standard dialects. Each stage uses appropriate abstractions—structured Linalg operations, not explicit loops. This demonstrates multi-level IR in action: high level (`nn`), mid level (Linalg), low level (loops/LLVM), with transformations between levels preserving semantic information as long as possible.

**Optimization Opportunities**. Even without explicit fusion passes, MLIR's default optimizations apply:

1. **Buffer reuse**: The `memref.alloc()` operations allocate heap memory, but later passes could stack-allocate or reuse buffers.
2. **Operation fusion**: A fusion pass could merge the first matmul and relu into one kernel, eliminating the intermediate buffer `%3`.
3. **Vectorization**: Linalg lowering to loops preserves parallelism information, enabling automatic vectorization.
4. **Tiling**: For larger matrices, tiling passes could block iterations for cache locality.

Our simple pipeline doesn't include these optimizations (we go straight from Linalg to loops to LLVM), but they're available as pass manager options. Chapter 10 will explore optimization passes systematically. For now, recognize that the IR structure we've created—high-level operations lowered progressively—makes these optimizations feasible. Direct loop generation (Chapter 7 style) would require significantly more effort to recover optimization opportunities.

**Validation Against NumPy**. The test suite validates results against NumPy:

```python
h_expected = x_data @ W1_data
h_relu_expected = np.maximum(0, h_expected)
y_expected = h_relu_expected @ W2_data

assert np.allclose(result, y_expected, rtol=1e-5)
```

This pattern—build graph, compile, execute, compare against reference—mirrors ML framework testing practices. PyTorch tests validate against eager-mode execution, TensorFlow tests validate against reference implementations, and our tests validate against NumPy. The assertion ensures numerical correctness within floating-point tolerance (5 decimal places). For learning, this validation is crucial: it proves our entire stack (graph building, lowering, compilation, execution) produces correct results.

**Scaling Up**. This two-layer network is small by modern standards, but the patterns scale directly. A 50-layer transformer would use the same API:

```python
x = g.variable([batch_size, seq_len, hidden_dim])
for i in range(50):
    x = transformer_block(g, x, layer_params[i])
output = g.matmul(x, output_weights)
```

Each transformer block is 10-15 operations (attention, layer norm, feed-forward), so 50 layers is 500-750 operations. Our graph builder handles this effortlessly—operations append to a list, IDs increment, shapes propagate. Lowering generates thousands of standard operations, but that's MLIR's job, not ours. Compilation might take seconds instead of milliseconds, but it's one-time cost amortized over many inferences.

The key lesson: appropriate abstractions at each level. Users work with `nn` operations (semantic level), lowering works with Linalg operations (computational pattern level), MLIR works with loops and LLVM (implementation level). Each level focuses on its concerns without being overwhelmed by details from other levels. This separation of concerns is what makes AI compilers tractable.

## 8.13 Looking Ahead: From Python Strings to TableGen

Our string-based custom dialect implementation in Chapter 8 demonstrates core concepts—high-level operations, lowering patterns, multi-level IR—without the complexity of MLIR's declarative specification tools. But production systems rarely generate MLIR text manually. The next chapter introduces **TableGen**, MLIR's domain-specific language for defining dialects, which automates much of what we've written by hand.

**What TableGen Provides**. Instead of Python f-strings to generate MLIR text:

```python
f"%{result} = nn.add %{lhs}, %{rhs} : {type}"
```

TableGen lets you write declarative operation definitions:

```tablegen
def AddOp : NeuralNetwork_Op<"add"> {
  let arguments = (ins F32Tensor:$lhs, F32Tensor:$rhs);
  let results = (outs F32Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
```

From this specification, TableGen generates C++ classes (`AddOp`), verifier methods (type checking), builders (constructing operations programmatically), and parsers (reading MLIR text). What took us 20 lines of Python string manipulation becomes 5 lines of declarative specification plus automatic code generation. This is how all MLIR built-in dialects (Linalg, Arith, Tensor, etc.) are defined.

**Lowering Patterns in C++**. Our Python lowering code:

```python
def lower_add(self, result_id, lhs_id, rhs_id, shape):
    # ... string manipulation ...
    return [f"{indent}%{result_id} = memref.alloc() : ...", ...]
```

Becomes C++ pattern rewriters in Chapter 9:

```cpp
class AddOpLowering : public OpRewritePattern<nn::AddOp> {
  LogicalResult matchAndRewrite(nn::AddOp op, PatternRewriter &rewriter) {
    auto result = rewriter.create<memref::AllocOp>(op.getLoc(), ...);
    auto generic = rewriter.create<linalg::GenericOp>(...);
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};
```

This C++ approach integrates with MLIR's dialect conversion framework, providing compile-time type safety, automatic pattern application, and composability with other rewrite patterns. Errors detected at C++ compile time, not Python runtime or MLIR parsing time. IDE support (autocomplete, jump-to-definition) works seamlessly. Debugging uses standard C++ tools.

**Benefits of TableGen**. Why learn another language (TableGen) when Python strings work? Several compelling reasons:

1. **Type Safety**: TableGen-generated C++ code is type-checked. You can't accidentally pass a `tensor<2xf32>` to an operation expecting `tensor<3xf32>`—the compiler rejects it. With strings, you discover type mismatches at MLIR parsing time or runtime.

2. **Verification**: TableGen definitions can include verifiers that check operation constraints. An operation requiring "inputs must have matching shapes" gets automatic verification generated. With strings, you write manual validation code.

3. **Documentation**: TableGen definitions are self-documenting. The operation's syntax, constraints, and semantics are centralized in one file. With strings, documentation lives separately (if at all) from implementation.

4. **Tooling**: MLIR's tooling (mlir-tblgen, mlir-opt) integrates with TableGen definitions. You get automatic operation documentation, parser generation, printer generation. With strings, you write all these by hand.

5. **Maintenance**: Changing an operation's signature in TableGen regenerates all affected C++ code automatically. With strings, you manually update every piece of code that references the operation.

**The Learning Path**. Chapter 8's string-based approach taught you **what** custom dialects are and **why** they matter. You built high-level operations, implemented lowering transformations, and saw how multi-level IR enables optimization. These concepts are fundamental—they apply regardless of implementation technique.

Chapter 9 will teach you **how** production systems implement these concepts using MLIR's idiomatic tools. TableGen for dialect definition, C++ pattern rewriters for lowering, dialect interfaces for extensibility. The jump from Python strings to TableGen might feel steep, but you now understand what the generated code does—you've written it manually. TableGen automates tedious parts while preserving the structure you've learned.

**Transitioning Code**. Our `nn` dialect operations:

```python
# Python string generation (Chapter 8)
g = Graph()
x = g.variable([4])
y = g.variable([4])
z = g.add(x, y)
mlir_text = g.get_mlir(z)
```

Becomes TableGen + C++ IR builders (Chapter 9):

```cpp
// C++ with TableGen-generated classes
auto x = builder.create<nn::VariableOp>(loc, tensor_type);
auto y = builder.create<nn::VariableOp>(loc, tensor_type);
auto z = builder.create<nn::AddOp>(loc, x, y);
```

The semantic structure is identical—variables and operations with dependencies—but the implementation uses MLIR's typed C++ APIs instead of string manipulation. The generated MLIR is the same; the path to generate it changes. Understanding both paths—manual string generation (Chapter 8) and idiomatic builders (Chapter 9)—gives you complete knowledge of how custom dialects work from first principles to production practice.

Chapter 9 will walk through TableGen syntax systematically, showing how to define operations, types, attributes, and interfaces. We'll implement the same `nn` dialect with TableGen, compare the generated C++ code against our manual implementation, and write lowering passes using pattern rewriters. By the end, you'll have both pedagogical clarity (from Chapter 8) and production capability (from Chapter 9)—a complete understanding of custom dialect development in MLIR.