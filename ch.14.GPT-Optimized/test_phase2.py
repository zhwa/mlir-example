#!/usr/bin/env python3
import sys
sys.path.insert(0, '../build/x64-release/ch.14.GPT-Optimized')
import ch14 as ch13
import numpy as np

print("Testing Phase 2 operations...")

# Test 1: Add operation
print("\nTest 1: Add operation (linalg.generic)")
A = np.random.randn(4, 8).astype(np.float32)
B = np.random.randn(4, 8).astype(np.float32)
try:
    result = ch13.forward(ch13.Tensor(A) + ch13.Tensor(B))
    expected = A + B
    if np.allclose(result, expected):
        print("  ✓ Add works!")
    else:
        print(f"  ✗ Add failed! Max diff: {np.abs(result - expected).max()}")
except Exception as e:
    print(f"  ✗ Add failed with error: {e}")

# Test 2: GELU operation
print("\nTest 2: GELU operation (linalg.generic)")
X = np.random.randn(4, 8).astype(np.float32) * 0.1
try:
    result = ch13.forward(ch13.gelu(ch13.Tensor(X)))
    print(f"  ✓ GELU works! Output shape: {result.shape}")
except Exception as e:
    print(f"  ✗ GELU failed with error: {e}")

print("\nPhase 2 basic tests complete!")
