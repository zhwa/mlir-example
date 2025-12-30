# Chapter 6: Softmax with Tensor Reductions

This chapter demonstrates **tensor-first softmax** using `linalg.reduce` for reductions and `linalg.generic` for element-wise operations, implementing the numerically stable softmax activation function.

## What You'll Learn

- **Linalg Reductions**: Using `linalg.reduce` for max and sum operations
- **Multi-Pass Tensor Algorithm**: Four functional passes with tensors
- **Math Dialect**: Using `math.exp` for exponential function
- **Numerical Stability**: Avoiding overflow with max subtraction technique
- **Tensor Operations**: Element-wise operations with `linalg.generic`

## The Kernel: Softmax

**Softmax** converts a vector of real numbers into a probability distribution:

```
output[i] = exp(input[i]) / sum(exp(input[j]))
```

**Problem**: For large values, `exp(input[i])` can overflow!

**Solution**: Numerically stable version:
```
max_val = max(input)
output[i] = exp(input[i] - max_val) / sum(exp(input[j] - max_val))
```

Subtracting the maximum value before `exp` prevents overflow while giving the same result.

## Tensor-First Four-Pass Algorithm

### Pass 1: Find Maximum Value with linalg.reduce
```mlir
%neg_inf = arith.constant 0xFF800000 : f32
%init_max = tensor.from_elements %neg_inf : tensor<f32>

%max_tensor = linalg.reduce ins(%input : tensor<?xf32>)
                            outs(%init_max : tensor<f32>)
                            dimensions = [0]
  (%in: f32, %init: f32) {
    %new_max = arith.maximumf %in, %init : f32
    linalg.yield %new_max : f32
  }
%max_val = tensor.extract %max_tensor[] : tensor<f32>
```

**Key Concepts:**
- `linalg.reduce`: Reduces a tensor along specified dimensions
- `tensor.from_elements`: Creates scalar tensor with initial value
- `tensor.extract`: Extracts scalar value from 0-d tensor
- Functional: No mutation, returns new tensor

### Pass 2: Compute exp(x - max) with linalg.generic
```mlir
%empty = tensor.empty(%size) : tensor<?xf32>

%exp_tensor = linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
  iterator_types = ["parallel"]
} ins(%input : tensor<?xf32>) outs(%empty : tensor<?xf32>) {
^bb0(%in: f32, %out: f32):
  %shifted = arith.subf %in, %max_val : f32
  %exp_val = math.exp %shifted : f32      // ← Math dialect!
  linalg.yield %exp_val : f32
} -> tensor<?xf32>
```

**Key Concepts:**
- `linalg.generic`: Element-wise operation on tensors
- `tensor.empty`: Allocates uninitialized tensor for output
- `math.exp`: Mathematical exponential function
- Parallel iterator: All elements processed independently

### Pass 3: Sum exp values with linalg.reduce
```mlir
%zero = arith.constant 0.0 : f32
%init_sum = tensor.from_elements %zero : tensor<f32>

%sum_tensor = linalg.reduce ins(%exp_tensor : tensor<?xf32>)
                           outs(%init_sum : tensor<f32>)
                           dimensions = [0]
  (%in: f32, %init: f32) {
    %new_sum = arith.addf %in, %init : f32
    linalg.yield %new_sum : f32
  }
%sum_val = tensor.extract %sum_tensor[] : tensor<f32>
```

### Pass 4: Normalize with linalg.generic
```mlir
%result = linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
  iterator_types = ["parallel"]
} ins(%exp_tensor : tensor<?xf32>) outs(%empty2 : tensor<?xf32>) {
^bb0(%in: f32, %out: f32):
  %normalized = arith.divf %in, %sum_val : f32
  linalg.yield %normalized : f32
} -> tensor<?xf32>
```

## Function Signature: Tensor-First

```mlir
func.func @softmax(%input: tensor<?xf32>) -> tensor<?xf32> {
  // Four-pass algorithm: find max, compute exp, sum, normalize
  // All operations on tensors (functional, immutable)
  return %result : tensor<?xf32>
}
```

After bufferization, this becomes:
```mlir
func.func @softmax(%input: memref<?xf32>, %output: memref<?xf32>) {
  // Return value converted to out-parameter
  return
}
```

## Why Tensor-First for Softmax?

### 1. **Industry Standard Pattern**
- Torch-MLIR, IREE, StableHLO all use tensor reductions
- `linalg.reduce` is the standard way to express reductions
- Matches PyTorch/JAX high-level semantics

### 2. **Better Optimization**
- Compiler can fuse passes (e.g., exp + sum into single loop)
- Parallel reduction patterns recognized by optimizer
- Bufferization chooses optimal memory layout

### 3. **Cleaner Semantics**
- No explicit loops or temporary buffers in high-level IR
- Reduction intent is explicit (`linalg.reduce`)
- Element-wise operations clearly parallel

### 4. **Composability**
- Easy to extend (e.g., add masking, axis parameter)
- Works with higher-dimensional tensors
- Can fuse with other tensor operations

## The Modern Pipeline

```
High-Level Tensor IR                                    (Functional)
  ├─ linalg.reduce (find max, compute sum)
  ├─ linalg.generic (exp, normalize)
  ├─ tensor.empty, tensor.extract
  └─ math.exp
          ↓ [Canonicalize]
          ↓ [One-Shot Bufferize]
Bufferized IR                                           (Imperative)
  ├─ memref operations
  ├─ linalg.generic on memrefs
  └─ Function boundaries: out-parameters
          ↓ [Linalg-to-Loops]
Loop IR
  ├─ scf.for loops
  ├─ memref.load/store
  └─ math.exp
          ↓ [Math-to-Libm/LLVM]
          ↓ [SCF-to-CF]
          ↓ [Convert to LLVM]
LLVM Dialect
  ├─ llvm.* instructions
  ├─ llvm.call @expf (libm)
  └─ cf.br branches
          ↓ [LLVM Translation]
Machine Code                                            (Executable)
```

## Comparison: Tensor-First vs Direct Loops

| Aspect | Tensor-First (Chapter 6) | Direct SCF (Old) |
|--------|-------------------------|------------------|
| **Reductions** | `linalg.reduce` (declarative) | `scf.for` with `iter_args` |
| **Element-wise** | `linalg.generic` (parallel) | Explicit `scf.for` loops |
| **Function sig** | `tensor<?xf32> -> tensor<?xf32>` | `(memref, memref) -> ()` |
| **Temp buffers** | Automatic (bufferization) | Manual `memref.alloca` |
| **Optimization** | Fusion, parallelization | Limited |
| **Industry** | ✅ Standard (Torch-MLIR, IREE) | ❌ Low-level |

## Implementation Files

### `ir.cpp` - Tensor-First Softmax IR Generation
- Creates 4-pass algorithm with tensor operations
- Uses `linalg.reduce` for max and sum reductions
- Uses `linalg.generic` for exp and normalization
- Function returns `tensor<?xf32>` (functional)

Key points:
- `tensor.from_elements`: Creates scalar tensor for reduction init
- `tensor.extract`: Extracts scalar from 0-d tensor
- `linalg.reduce`: Reduces tensor along dimension 0
- Affine maps for element-wise indexing

### `lowering.cpp` - Bufferization and Lowering Pipeline
- Registers bufferization interfaces for tensor operations
- Configures One-Shot Bufferize with function boundary handling
- Converts linalg to loops, math to libm calls
- Full pipeline: Tensor → MemRef → Loops → LLVM

Passes:
1. Canonicalize
2. One-Shot Bufferize
3. Buffer-Results-To-Out-Params
4. Linalg-To-Loops
5. Math-To-Libm/LLVM
6. SCF-To-CF
7. Convert-To-LLVM

### `jit.cpp` - JIT Compilation
- Compiles tensor-first softmax to native code
- After bufferization, function signature becomes memref out-parameter
- Links with libm for `expf` function
- Demonstrates end-to-end execution

### `bindings.cpp` - Python Bindings
- Already compatible with tensor-first architecture
- Wraps memref interface (after bufferization)

## Usage

```python
import numpy as np
import ch6_softmax

# Simple example
x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
probs = ch6_softmax.softmax(x)
# Result: [0.09, 0.24, 0.67] (sums to 1.0)

# Handles large values without overflow
x = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
probs = ch6_softmax.softmax(x)
# Still works correctly!
```

## Build

From the project root:
```bash
cmake --build --preset x64-release --target ch6-softmax
```

Or rebuild everything:
```bash
cmake --build --preset x64-release
```

probs = ch6_softmax.softmax(x)
# Result: still [0.09, 0.24, 0.67] - numerically stable!
```

## Test

```bash
cd ch.6.Softmax
python test_jit.py
```

Expected output:
- ✓ Basic softmax test passes
- ✓ Large values handled correctly (numerical stability)
- ✓ Edge cases (zeros) work
- ✓ Random values match NumPy

## Comparison with Chapter 5

| Aspect | Chapter 5 (SAXPY) | Chapter 6 (Softmax) |
|--------|-------------------|---------------------|
| Architecture | Tensor-first | Tensor-first |
| Operation | Element-wise linear | Non-linear with reductions |
| Linalg Ops | `linalg.generic` only | `linalg.reduce` + `linalg.generic` |
| Algorithm | Single pass | Four passes |
| Key Feature | Parallel element-wise | Reductions + element-wise |
| Math Ops | +, × | +, ×, ÷, **exp**, max |
| Numerical Concern | None | Overflow prevention |

## Why Four Passes?

**Can't we do it in one pass?**

No, because:
1. **Pass 1 (Find max)**: Need maximum value *before* computing exponentials
2. **Pass 2 (Compute exp)**: Can only compute exp(x - max) after knowing max
3. **Pass 3 (Sum)**: Need sum of all exp values *before* normalizing
4. **Pass 4 (Normalize)**: Can only divide by sum after computing it

This demonstrates that some algorithms inherently require multiple passes. The tensor-first approach makes these dependencies explicit.

## Next Steps

- **Chapter 7**: Neural network operations (ReLU, batch normalization)
- **Chapter 8**: Custom dialects for domain-specific operations
- **Beyond**: Multi-dimensional tensors, tiling, GPU acceleration

## Key Takeaways

1. **linalg.reduce**: Standard pattern for reductions (max, sum, product)
2. **Multi-pass algorithms**: Some operations require sequential stages
3. **Math dialect**: Mathematical functions (`exp`, `log`, etc.) with proper lowering
4. **Numerical stability**: Subtract max before exp to prevent overflow
5. **Tensor-first**: Even complex algorithms benefit from high-level tensor representation

---

**Previous**: [Chapter 5 - Vector Operations](../ch.5.Vector-ops/README.md)  
**Next**: [Chapter 7 - Neural Operations](../ch.7.Neural-ops/README.md)
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/raw_ostream.h"
```

**Key lesson:** In C++, forward declarations only work for pointers/references. If you instantiate objects or call methods, you need the complete type definition from headers.

### Issue 3: Understanding Pass Ordering

**Challenge:** Why does pass order matter?

**Answer:** MLIR passes must be ordered to create a valid lowering progression:

```
1. canonicalize          - Simplify IR (optional but recommended)
2. math-to-libm         - Lower math ops to library calls
3. math-to-llvm         - Convert remaining math ops
4. scf-to-cf            - Lower structured loops to branches
5. func-to-llvm         - Convert function signatures
6. arith-to-llvm        - Convert arithmetic operations
7. cf-to-llvm           - Convert control flow
8. memref-to-llvm       - Convert memory operations
9. reconcile-casts      - Clean up conversion artifacts
```

**Wrong order example:**
```cpp
pm.addPass(createConvertFuncToLLVMPass());  // Too early!
pm.addPass(createConvertMathToLibmPass());   // Math ops already gone
```

If you convert functions to LLVM first, subsequent passes may not recognize the converted operations.

**Key lesson:** High-level dialects → Mid-level dialects → LLVM dialect. Always lower from most abstract to most concrete.

### Debugging Tips

1. **Print IR at each stage:** Add `module->dump()` between passes to see transformations
2. **Check pass pipeline:** Use `pm.enableIRPrinting()` to see what each pass does
3. **Verify dialect loading:** Make sure all required dialects are loaded in MLIRContext
4. **Test incrementally:** Build and test after adding each major component

### References for Math Dialect

**Current implementation uses BOTH passes:**
```cpp
pm.addPass(createConvertMathToLibmPass());  // Try libm first
pm.addPass(createConvertMathToLLVMPass());   // Fallback for remaining ops
```

**Two lowering strategies for math operations:**

1. **Math-to-Libm** (`createConvertMathToLibmPass()`):
   - Creates external function calls to C standard library (libm)
   - Example: `math.exp %x` → `llvm.call @expf(%x)`
   - Links to system's libm at runtime (need `-lm` linker flag)
   - **Pros**: High accuracy (hardware/library optimized implementations)
   - **Cons**: Function call overhead, requires linking libm
   - **Best for**: Production code, numerical applications requiring precision

2. **Math-to-LLVM** (`createConvertMathToLLVMPass()`):
   - Converts to LLVM intrinsics or inline polynomial approximations
   - Example: `math.exp %x` → inline polynomial expansion or `llvm.exp.*` intrinsic
   - Everything inlined, no external dependencies
   - **Pros**: Faster (no function calls), self-contained binary
   - **Cons**: Lower accuracy (polynomial approximations), larger code size
   - **Best for**: Performance-critical code, embedded systems without libm

**Experiment suggestion:** In `lowering.cpp`, try commenting out `math-to-libm`:
```cpp
// pm.addPass(createConvertMathToLibmPass());  // Comment this out
pm.addPass(createConvertMathToLLVMPass());     // Use LLVM intrinsics only
```

Then rebuild and compare:
- Execution speed (LLVM version might be faster)
- Numerical accuracy (compare with NumPy reference)
- Binary size (check compiled module size)

**Why we use both:** Robustness! If `math-to-libm` can't handle an operation (or on platforms without libm), `math-to-llvm` provides a fallback.