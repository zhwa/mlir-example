#!/usr/bin/env python3
"""
Test script for MLIR-JIT compiled GEMM.

This script verifies that our JIT-compiled matrix multiplication works correctly
by comparing results against NumPy's highly optimized implementation.

Tests:
  1. Ones matrix - Easy to verify: each result should be sum of 32 ones = 32.0
  2. Random matrices - Compare against NumPy to verify numerical correctness
"""

import sys
import os
import numpy as np

# Auto-detect where the compiled Python module is located
# The unified build system places modules in: ../build/x64-release/ch.2.Dynamic-size/
build_paths = [
    '../build/x64-release/ch.2.Dynamic-size',
    '../build/x64-debug/ch.2.Dynamic-size',
    'build/x64-release/ch.2.Dynamic-size',  # If running from root
    'build/x64-debug/ch.2.Dynamic-size',
]

for path in build_paths:
    if os.path.exists(path) and any(f.startswith('ch2_dynamic_size') for f in os.listdir(path) 
                                     if not os.path.isdir(os.path.join(path, f))):
        sys.path.insert(0, path)
        print(f"Using build directory: {path}")
        break
else:
    print("ERROR: Could not find ch2_dynamic_size module!")
    print("Make sure you've built the project first:")
    print("  cmake --preset x64-release")
    print("  cmake --build --preset x64-release")
    sys.exit(1)

import ch2_dynamic_size as gemm

print("Testing JIT-compiled GEMM...")
print("=" * 60)

# Test 1: Ones matrix - easy to verify manually
print("\n=== Test 1: Ones matrix (sanity check) ===")
print("Computing: C = ones(8,32) @ ones(32,16)")
A = np.ones((8, 32), dtype=np.float32)
B = np.ones((32, 16), dtype=np.float32)

print("Calling gemm(A, B)...")
try:
    C = gemm.gemm(A, B)
    print(f"✓ Success! Result shape: {C.shape}")
    print(f"✓ C[0,0] = {C[0,0]:.1f} (expected: 32.0)")
    
    # Each element should be 32.0 (sum of 32 ones)
    if np.allclose(C, 32.0):
        print("✓ All values correct!")
    else:
        print(f"✗ FAILED: Expected all 32.0, got range [{C.min():.1f}, {C.max():.1f}]")
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Random matrices - verify against NumPy
print("\n=== Test 2: Random matrices (numerical accuracy) ===")
print("Computing: C = random(8,32) @ random(32,16)")
np.random.seed(42)  # Reproducible results
A = np.random.randn(8, 32).astype(np.float32)
B = np.random.randn(32, 16).astype(np.float32)

print("Calling gemm(A, B)...")
try:
    C_jit = gemm.gemm(A, B)
    C_numpy = A @ B  # NumPy's reference implementation
    
    max_error = np.max(np.abs(C_jit - C_numpy))
    print(f"✓ Success! Max error vs NumPy: {max_error:.2e}")
    
    if np.allclose(C_jit, C_numpy, rtol=1e-5):
        print("✓ Results match NumPy (within float32 precision)!")
    else:
        print(f"✗ FAILED: Errors too large! Max error: {max_error}")
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Different matrix dimensions - verify dynamic shapes work
print("\n=== Test 3: Different matrix sizes (dynamic shapes) ===")
print("Computing: C = random(16,64) @ random(64,32)")
np.random.seed(123)  # Different seed for variety
A3 = np.random.randn(16, 64).astype(np.float32)
B3 = np.random.randn(64, 32).astype(np.float32)

print("Calling gemm(A3, B3) with different dimensions...")
try:
    C3_jit = gemm.gemm(A3, B3)
    C3_numpy = A3 @ B3
    
    max_error = np.max(np.abs(C3_jit - C3_numpy))
    print(f"✓ Success! Max error vs NumPy: {max_error:.2e}")
    
    if np.allclose(C3_jit, C3_numpy, rtol=1e-5):
        print("✓ Different sizes work with same compiled function!")
        print(f"  Shape: ({A3.shape[0]}×{A3.shape[1]}) @ ({B3.shape[0]}×{B3.shape[1]}) = ({C3_jit.shape[0]}×{C3_jit.shape[1]})")
    else:
        print(f"✗ FAILED: Errors too large! Max error: {max_error}")
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("=== All tests complete ===")
print("\nTry these commands to explore the MLIR IR:")
print("  python3 -c 'import gemm; print(gemm.test_ir_generation())'")
print("  python3 -c 'import gemm; print(gemm.test_optimized_ir())'")