# Chapter 6: Softmax with Math Dialect

This chapter demonstrates the **Math dialect** for mathematical functions and implements the softmax activation function using a numerically stable three-pass algorithm.

## What You'll Learn

- **Math Dialect**: Using `math.exp` and other mathematical operations
- **Multi-Pass Algorithms**: Breaking complex operations into sequential passes
- **Numerical Stability**: Avoiding overflow with max subtraction technique
- **Reduction Patterns**: Finding maximum values and computing sums
- **Loop-Carried Variables**: Using `scf.for` with iteration arguments

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

## Three-Pass Algorithm

### Pass 1: Find Maximum Value
```mlir
%max = scf.for %i = %c0 to %size step %c1 
       iter_args(%current_max = %neg_inf) -> (f32) {
  %val = memref.load %input[%i] : memref<?xf32>
  %new_max = arith.maximumf %current_max, %val : f32
  scf.yield %new_max : f32
}
```

### Pass 2: Compute exp(x - max) and Sum
```mlir
%sum = scf.for %i = %c0 to %size step %c1 
       iter_args(%current_sum = %c0_f32) -> (f32) {
  %val = memref.load %input[%i] : memref<?xf32>
  %shifted = arith.subf %val, %max : f32
  %exp_val = math.exp %shifted : f32      // ← Math dialect!
  memref.store %exp_val, %temp[%i] : memref<?xf32>
  %new_sum = arith.addf %current_sum, %exp_val : f32
  scf.yield %new_sum : f32
}
```

### Pass 3: Normalize
```mlir
scf.for %i = %c0 to %size step %c1 {
  %exp_val = memref.load %temp[%i] : memref<?xf32>
  %normalized = arith.divf %exp_val, %sum : f32
  memref.store %normalized, %output[%i] : memref<?xf32>
}
```

## Key MLIR Concepts

### Loop-Carried Variables

`scf.for` can carry values across iterations:

```mlir
%result = scf.for %i = %start to %end step %step 
          iter_args(%arg = %init) -> (f32) {
  // Use %arg (value from previous iteration)
  %new_val = ... compute using %arg ...
  scf.yield %new_val : f32  // Pass to next iteration
}
// %result contains the final value
```

This is how we implement reductions (max, sum) without mutation.

### Math Dialect Lowering

**math-to-libm pass** converts math operations to C library calls:

```mlir
// Before lowering
%exp = math.exp %x : f32

// After math-to-libm
%exp = llvm.call @expf(%x) : (f32) -> f32
```

At link time, this calls the standard C library's `expf()` function.

**Alternative**: `math-to-llvm` pass uses polynomial approximations (inline, faster, less accurate).

## Lowering Pipeline

```
High-Level MLIR
  (scf.for, math.exp)
    ↓ canonicalize
  (simplify)
    ↓ math-to-libm
  (math.exp → llvm.call @expf)
    ↓ scf-to-cf
  (scf.for → cf.br branches)
    ↓ convert-to-llvm
  (all dialects → llvm.*)
    ↓ mlir-translate
LLVM IR
    ↓ JIT compile
Native Code (links with libm)
```

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
- Performance benchmark comparison

## Comparison with Chapter 5

| Aspect | Chapter 5 (SAXPY) | Chapter 6 (Softmax) |
|--------|-------------------|---------------------|
| Operation | Element-wise linear | Non-linear with reduction |
| Dialects | SCF, Arith | SCF, Arith, **Math** |
| Algorithm | Single pass | Three passes |
| Key Feature | Simple loops | Loop-carried variables |
| Math Ops | +, × | +, ×, ÷, **exp**, max |
| Numerical Concern | None | Overflow prevention |
| Temp Storage | None | Needed for exp values |

## Why Three Passes?

**Can't we do it in one pass?**

No, because we need the maximum value *before* computing exponentials:
1. First pass: Find max (need to see all values)
2. Second pass: Now we can safely compute exp(x - max)
3. Third pass: Normalize using the sum

This demonstrates an important pattern: some algorithms require multiple passes over data.

## Exercises

- **Experiment**: Try using `math-to-llvm` **only** (remove `math-to-libm`) to see performance/accuracy trade-offs
- **Experiment**: Try other math functions (log, sin, cos, sqrt, etc.)
- **Challenge**: Implement log-softmax (more numerically stable for ML applications)

---

## Common Issues and Solutions

### Issue 1: "failed to legalize operation 'math.exp'"

**Error message:**
```
error: failed to legalize operation 'math.exp' that was explicitly marked illegal
Pass manager failed
```

**What went wrong:**
The initial lowering pipeline only included `math-to-libm` pass, but this pass alone doesn't fully convert math operations to LLVM. The pass manager was trying to convert everything to LLVM dialect, but `math.exp` operations were still present and marked as illegal in the LLVM conversion target.

**Why it failed:**
MLIR's conversion infrastructure uses a "target legality" system. When we run `convert-*-to-llvm` passes, they mark certain operations as illegal (must be converted). However:
1. `math-to-libm` creates function declarations for libm calls but doesn't always succeed for all math ops
2. Without a fallback, unconverted math operations remain illegal
3. The conversion fails because no pass can handle these remaining operations

**The fix:**
Add **both** `math-to-llvm` and `math-to-libm` passes in sequence:

```cpp
// Phase 2: Math to Libm
pm.addPass(createConvertMathToLibmPass());

// Phase 3: Math to LLVM (for remaining math ops)
pm.addPass(createConvertMathToLLVMPass());
```

This creates a two-tier strategy:
- First, try to lower to libm calls (more accurate, uses standard library)
- Then, convert any remaining math ops to LLVM intrinsics or polynomial approximations

**Key lesson:** When lowering dialects, always provide a complete lowering path. If one pass doesn't handle all cases, add fallback passes.

### Issue 2: Missing headers in bindings.cpp

**Error messages:**
```
error: variable has incomplete type 'mlir::MLIRContext'
error: no member named 'func' in namespace 'mlir'
error: no member named 'createSoftmaxModule' in namespace 'mlir'
```

**What went wrong:**
The bindings file had forward declarations but missing the actual header includes for MLIR types and dialects.

**The fix:**
Add all necessary MLIR includes:

```cpp
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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