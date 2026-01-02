#!/usr/bin/env python3
"""
Chapter 9: Custom Dialect with TableGen - Complete Test Suite

Demonstrates industrial-strength MLIR dialect development:
- TableGen-defined custom dialect
- OpBuilder-based IR construction (no string generation!)
- Pythonic API with operator overloading
- Same approach as Torch-MLIR, JAX, and IREE
"""

import sys
import os
import numpy as np

# Add build directory to path
build_paths = [
    '../build/x64-release/ch.9.TableGen-dialect',
    '../build/x64-debug/ch.9.TableGen-dialect',
    'build/x64-release/ch.9.TableGen-dialect',
    'build/x64-debug/ch.9.TableGen-dialect'
]

build_dir = None
for path in build_paths:
    if os.path.exists(path):
        build_dir = path
        break

if not build_dir:
    print("Error: Could not find build directory")
    sys.exit(1)

print(f"Using build directory: {build_dir}")
sys.path.insert(0, build_dir)

import ch9

print()
print("=" * 70)
print("Chapter 9: Custom Dialect with TableGen")
print("=" * 70)
print()

# Test 1: Element-wise addition with operator overloading
print("### Test 1: Tensor Addition (a + b) ###")
a = ch9.Tensor(np.array([1., 2., 3., 4.], dtype=np.float32))
b = ch9.Tensor(np.array([5., 6., 7., 8.], dtype=np.float32))
c = a + b
result = ch9.forward(c)
print(f"✓ [1. 2. 3. 4.] + [5. 6. 7. 8.] = {result}")
print()

# Test 2: Element-wise multiplication with operator overloading
print("### Test 2: Tensor Multiplication (a * b) ###")
a = ch9.Tensor(np.array([2., 3., 4., 5.], dtype=np.float32))
b = ch9.Tensor(np.array([10., 10., 10., 10.], dtype=np.float32))
c = a * b
result = ch9.forward(c)
print(f"✓ [2. 3. 4. 5.] * [10. 10. 10. 10.] = {result}")
print()

# Test 3: Matrix multiplication
print("### Test 3: Matrix Multiplication ###")
a = ch9.Tensor(np.array([[1., 2., 3.],
                         [4., 5., 6.]], dtype=np.float32))
b = ch9.Tensor(np.array([[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 1., 0.]], dtype=np.float32))
c = ch9.matmul(a, b)
result = ch9.forward(c)
print(f"✓ MatMul: (2, 3) @ (3, 4) = (2, 4)")
print(f"  Result shape: {result.shape}")
print(f"  First row: {result[0]}")
print()

# Test 4: ReLU activation
print("### Test 4: ReLU Activation ###")
a = ch9.Tensor(np.array([-1., 2., -3., 4.], dtype=np.float32))
b = ch9.relu(a)
result = ch9.forward(b)
print(f"✓ Input:  {a.numpy()}")
print(f"  Output: {result}")
print()

# Test 5: Chained operations (demonstrates graph building)
print("### Test 5: Chained Operations (a + b) * c ###")
a = ch9.Tensor(np.array([1., 2., 3., 4.], dtype=np.float32))
b = ch9.Tensor(np.array([1., 1., 1., 1.], dtype=np.float32))
c = ch9.Tensor(np.array([2., 3., 4., 5.], dtype=np.float32))
d = (a + b) * c
result = ch9.forward(d)
print(f"✓ ([1. 2. 3. 4.] + [1. 1. 1. 1.]) * [2. 3. 4. 5.] = {result}")
print()

# Test 6: More complex graph
print("### Test 6: Complex Graph: relu((a + b) * c) ###")
a = ch9.Tensor(np.array([1., -2., 3., -4.], dtype=np.float32))
b = ch9.Tensor(np.array([-2., 3., -4., 5.], dtype=np.float32))
c = ch9.Tensor(np.array([2., 1., 2., 1.], dtype=np.float32))
d = ch9.relu((a + b) * c)
result = ch9.forward(d)
print(f"✓ Input a: {a.numpy()}")
print(f"  Input b: {b.numpy()}")
print(f"  Input c: {c.numpy()}")
print(f"  Result:  {result}")
print()

# Test 7: Softmax
print("### Test 7: Softmax ###")
a = ch9.Tensor(np.array([[1., 2., 3.]], dtype=np.float32))
b = ch9.softmax(a)
result = ch9.forward(b)
print(f"✓ Input: {a.numpy()}")
print(f"  Output: {result}")
print()

# Test 8: Linear
print("### Test 8: Linear (x @ w.T + b) ###")
x = ch9.Tensor(np.array([[1., 2.]], dtype=np.float32))
w = ch9.Tensor(np.array([[1., 0.], [0., 1.]], dtype=np.float32))
b = ch9.Tensor(np.array([1., 1.], dtype=np.float32))
l = ch9.linear(x, w, b)
result = ch9.forward(l)
print(f"✓ Input: {x.numpy()}")
print(f"  Weight: {w.numpy()}")
print(f"  Bias: {b.numpy()}")
print(f"  Result: {result}")