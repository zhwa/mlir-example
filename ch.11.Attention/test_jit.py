#!/usr/bin/env python3
"""
Chapter 11: Attention Mechanism - JIT Compilation Tests

Tests pure MLIR JIT compilation approach with Tensor API.
"""

import sys
import os
import numpy as np

# Add build directory to path
build_paths = [
    '../build/x64-release/ch.11.Attention',
    '../build/x64-debug/ch.11.Attention',
    'build/x64-release/ch.11.Attention',
    'build/x64-debug/ch.11.Attention'
]

build_dir = None
for path in build_paths:
    if os.path.exists(path):
        build_dir = path
        break

if build_dir:
    print(f"Using build directory: {build_dir}")
    sys.path.insert(0, build_dir)
else:
    print("Warning: Build directory not found, attempting to import anyway")

try:
    import ch11
except ImportError as e:
    print(f"Error: Could not import ch11 module: {e}")
    print("Please build Chapter 11 first:")
    print("  cmake --build build/x64-release --target ch11")
    sys.exit(1)

print()
print("=" * 70)
print("Chapter 11: Attention Mechanism Tests (Pure MLIR JIT)")
print("=" * 70)
print()

# Test 1: Basic Add Operation
print("### Test 1: Add Operation ###")
np.random.seed(42)
a_data = np.random.randn(3, 4).astype(np.float32)
b_data = np.random.randn(3, 4).astype(np.float32)

a = ch11.Tensor(a_data)
b = ch11.Tensor(b_data)
result = ch11.forward(a + b)

expected = a_data + b_data
print(f"Expected shape: {expected.shape}, Actual: {result.shape}")
print(f"Expected (first row): {expected[0, :3]}")
print(f"Actual (first row): {result[0, :3]}")

try:
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)
    print("✓ Test 1 PASSED: Add operation matches NumPy!")
except AssertionError as e:
    print(f"✗ Test 1 FAILED: {e}")

print()

# Test 2: Matrix Multiplication
print("### Test 2: Matmul Operation ###")
np.random.seed(123)
x_data = np.random.randn(4, 3).astype(np.float32)
y_data = np.random.randn(3, 5).astype(np.float32)

x = ch11.Tensor(x_data)
y = ch11.Tensor(y_data)
result = ch11.forward(ch11.matmul(x, y))

expected = np.matmul(x_data, y_data)
print(f"Expected shape: {expected.shape}, Actual: {result.shape}")
print(f"Expected (first row): {expected[0, :3]}")
print(f"Actual (first row): {result[0, :3]}")

try:
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)
    print("✓ Test 2 PASSED: Matmul matches NumPy!")
except AssertionError as e:
    print(f"✗ Test 2 FAILED: {e}")

print()

# Test 3: Transpose
print("### Test 3: Transpose Operation ###")
np.random.seed(456)
data = np.random.randn(4, 5).astype(np.float32)

t = ch11.Tensor(data)
result = ch11.forward(ch11.transpose(t))

expected = data.T
print(f"Expected shape: {expected.shape}, Actual: {result.shape}")
print(f"Expected (first row): {expected[0, :3]}")
print(f"Actual (first row): {result[0, :3]}")

try:
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)
    print("✓ Test 3 PASSED: Transpose matches NumPy!")
except AssertionError as e:
    print(f"✗ Test 3 FAILED: {e}")

print()

# Test 4: Softmax
print("### Test 4: Softmax Operation ###")
np.random.seed(789)
data = np.random.randn(3, 4).astype(np.float32)

t = ch11.Tensor(data)
result = ch11.forward(ch11.softmax(t))

# NumPy reference softmax (row-wise)
data_max = np.max(data, axis=-1, keepdims=True)
exp_data = np.exp(data - data_max)
expected = exp_data / np.sum(exp_data, axis=-1, keepdims=True)

print(f"Expected shape: {expected.shape}, Actual: {result.shape}")
print(f"Expected (first row): {expected[0]}")
print(f"Actual (first row): {result[0]}")
print(f"Expected row sums: {np.sum(expected, axis=-1)}")
print(f"Actual row sums: {np.sum(result, axis=-1)}")

try:
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)
    print("✓ Test 4 PASSED: Softmax matches NumPy!")
except AssertionError as e:
    print(f"✗ Test 4 FAILED: {e}")

print()

# Test 5: Scale
print("### Test 5: Scale Operation ###")
np.random.seed(1011)
data = np.random.randn(3, 4).astype(np.float32)
scale_factor = 0.5

t = ch11.Tensor(data)
result = ch11.forward(ch11.scale(t, scale_factor))

expected = data * scale_factor
print(f"Expected shape: {expected.shape}, Actual: {result.shape}")
print(f"Expected (first row): {expected[0]}")
print(f"Actual (first row): {result[0]}")

try:
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)
    print("✓ Test 5 PASSED: Scale matches NumPy!")
except AssertionError as e:
    print(f"✗ Test 5 FAILED: {e}")

print()

# Test 6: Simple Attention (no multi-head)
print("### Test 6: Simple Attention Mechanism ###")
np.random.seed(2023)
seq_len = 4
d_k = 8

Q_data = np.random.randn(seq_len, d_k).astype(np.float32)
K_data = np.random.randn(seq_len, d_k).astype(np.float32)
V_data = np.random.randn(seq_len, d_k).astype(np.float32)

Q = ch11.Tensor(Q_data)
K = ch11.Tensor(K_data)
V = ch11.Tensor(V_data)

result = ch11.forward(ch11.attention(Q, K, V))

# NumPy reference attention
scale_factor = 1.0 / np.sqrt(d_k)
scores = np.matmul(Q_data, K_data.T)  # (seq_len, seq_len)
scaled_scores = scores * scale_factor
scores_max = np.max(scaled_scores, axis=-1, keepdims=True)
scores_exp = np.exp(scaled_scores - scores_max)
attn_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
expected = np.matmul(attn_weights, V_data)

print(f"Expected shape: {expected.shape}, Actual: {result.shape}")
print(f"Expected (first row): {expected[0, :4]}")
print(f"Actual (first row): {result[0, :4]}")

try:
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)
    print("✓ Test 6 PASSED: Attention matches NumPy reference!")
except AssertionError as e:
    print(f"✗ Test 6 FAILED: {e}")

print()

print("=" * 70)
print("Test suite complete")
print("=" * 70)