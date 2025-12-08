#!/usr/bin/env python3
"""Chapter 10: Optimized Compilation with Fusion + Vectorization"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build/x64-release/ch.10.Optimizations'))

import ch10
import numpy as np

print("\n" + "="*70)
print(" Chapter 10: Optimized Compilation")
print(" Fusion + Vectorization with vector dialect")
print("="*70 + "\n")

# Test 1: MatMul
print("Test 1: Matrix Multiplication (128x256 @ 256x128)")
A = ch10.Tensor(np.random.randn(128, 256).astype(np.float32))
B = ch10.Tensor(np.random.randn(256, 128).astype(np.float32))
C = ch10.matmul(A, B)

result = ch10.forward(C)
expected = A.numpy() @ B.numpy()
np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)
print("✓ MatMul correctness verified")

# Test 2: Fused MatMul + ReLU
print("\nTest 2: Fused MatMul + ReLU (256x512 @ 512x256)")
W = ch10.Tensor(np.random.randn(256, 512).astype(np.float32))
X = ch10.Tensor(np.random.randn(512, 256).astype(np.float32))
Y = ch10.relu(ch10.matmul(W, X))

result = ch10.forward(Y)
expected = np.maximum(0, W.numpy() @ X.numpy())
np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)
print("✓ Fused MatMul+ReLU correctness verified")

# Test 3: Element-wise fusion
print("\nTest 3: Element-wise Fusion (A+B)*(C+D)")
A = ch10.Tensor(np.random.randn(512, 512).astype(np.float32))
B = ch10.Tensor(np.random.randn(512, 512).astype(np.float32))
C = ch10.Tensor(np.random.randn(512, 512).astype(np.float32))
D = ch10.Tensor(np.random.randn(512, 512).astype(np.float32))

result_tensor = (A + B) * (C + D)
result = ch10.forward(result_tensor)
expected = (A.numpy() + B.numpy()) * (C.numpy() + D.numpy())
np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)
print("✓ Element-wise fusion correctness verified")

print("\n" + "="*70)
print(" All tests passed!")
print(" Optimizations: Linalg fusion + vectorization (vector dialect)")
print("="*70)
