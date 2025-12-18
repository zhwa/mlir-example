#!/usr/bin/env python3
"""Test Pure MLIR JIT compilation"""

import numpy as np
import sys
import os

build_paths = [
    '../build/x64-release/ch.12.Transformer',
    'build/x64-release/ch.12.Transformer',
]

for path in build_paths:
    if os.path.exists(path):
        sys.path.insert(0, path)
        break

import ch12

print("Testing Pure MLIR JIT Compilation")
print("=" * 70)

# Test 1: Simple Add
print("\n### Test 1: Add ###")
a = ch12.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
b = ch12.Tensor(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
c = a + b  # Graph construction

result = ch12.forward(c)  # JIT compile and execute
expected = np.array([[6.0, 8.0], [10.0, 12.0]], dtype=np.float32)

print(f"Result:\n{result}")
print(f"Expected:\n{expected}")

np.testing.assert_allclose(result, expected, rtol=1e-5)
print("✓ Add test PASSED!")

# Test 2: LayerNorm
print("\n### Test 2: LayerNorm ###")
x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
gamma = np.ones(4, dtype=np.float32)
beta = np.zeros(4, dtype=np.float32)

x_tensor = ch12.Tensor(x)
output = ch12.layer_norm(x_tensor, gamma, beta)
result = ch12.forward(output)

# Compute expected output with numpy
mean = np.mean(x, axis=1, keepdims=True)
var = np.var(x, axis=1, keepdims=True)
expected = (x - mean) / np.sqrt(var + 1e-5)

print("Input:", x)
print("Result:", result)
print("Expected:", expected)

np.testing.assert_allclose(result, expected, atol=1e-5)
print("✓ LayerNorm test PASSED!")

print("\n" + "=" * 70)
print("SUCCESS: All Pure MLIR JIT tests passed!")
print("=" * 70)
