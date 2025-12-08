#!/usr/bin/env python3
"""Chapter 10: Baseline vs Optimized Compilation"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build/x64-release/ch.10.Optimizations'))

import ch10
import numpy as np
import time

print("\n" + "="*70)
print(" Chapter 10: Baseline vs Optimized Compilation")
print("="*70 + "\n")

# Test 1: MatMul
A = ch10.Tensor(np.random.randn(128, 256).astype(np.float32))
B = ch10.Tensor(np.random.randn(256, 128).astype(np.float32))
C = ch10.matmul(A, B)

t0 = time.time()
for _ in range(50):
    r1 = ch10.forward(C)
t_base = (time.time() - t0) / 50

t0 = time.time()
for _ in range(50):
    r2 = ch10.forward_optimized(C)
t_opt = (time.time() - t0) / 50

np.testing.assert_allclose(r1, r2, rtol=1e-5)
print(f"✓ MatMul (128x256 @ 256x128):")
print(f"  Baseline:  {t_base*1000:.2f} ms")
print(f"  Optimized: {t_opt*1000:.2f} ms")
print(f"  Speedup:   {t_base/t_opt:.2f}x\n")

# Test 2: Fused ops
W = ch10.Tensor(np.random.randn(256, 512).astype(np.float32))
X = ch10.Tensor(np.random.randn(512, 256).astype(np.float32))
Y = ch10.relu(ch10.matmul(W, X))

t0 = time.time()
for _ in range(50):
    r1 = ch10.forward(Y)
t_base = (time.time() - t0) / 50

t0 = time.time()
for _ in range(50):
    r2 = ch10.forward_optimized(Y)
t_opt = (time.time() - t0) / 50

np.testing.assert_allclose(r1, r2, rtol=1e-5)
print(f"✓ MatMul+ReLU (256x512 @ 512x256):")
print(f"  Baseline:  {t_base*1000:.2f} ms")
print(f"  Optimized: {t_opt*1000:.2f} ms")
print(f"  Speedup:   {t_base/t_opt:.2f}x\n")

print("="*70)
print(" All tests passed!")
print(" Optimizations: Linalg fusion + loop invariant code motion")
print("="*70)
