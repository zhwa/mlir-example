#!/usr/bin/env python3
"""Test SAXPY implementation using MLIR JIT."""

import sys
import os
import numpy as np

# Auto-detect build directory
build_paths = [
    '../build/x64-release/ch.5.Vector-ops',
    '../build/x64-debug/ch.5.Vector-ops',
    'build/x64-release/ch.5.Vector-ops',
    'build/x64-debug/ch.5.Vector-ops'
]

build_dir = None
for path in build_paths:
    if os.path.exists(path):
        build_dir = path
        break

if not build_dir:
    print("Error: Could not find build directory")
    print(f"Searched: {build_paths}")
    sys.exit(1)

print(f"Using build directory: {build_dir}")
sys.path.insert(0, build_dir)

import ch5_vector_ops

def test_saxpy_basic():
    """Test basic SAXPY operation."""
    print("=== Test: Basic SAXPY ===")

    alpha = 2.0
    A = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    B = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

    # Expected: [2*1+5, 2*2+6, 2*3+7, 2*4+8] = [7, 10, 13, 16]
    expected = np.array([7.0, 10.0, 13.0, 16.0], dtype=np.float32)

    # JIT-compiled MLIR version
    result = ch5_vector_ops.saxpy(alpha, A, B)

    print(f"alpha = {alpha}")
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"Result = {result}")
    print(f"Expected = {expected}")

    if np.allclose(result, expected):
        print("✓ Test passed!")
    else:
        print("✗ Test failed!")
        print(f"Difference: {result - expected}")

def test_saxpy_large():
    """Test with larger vectors."""
    print("\n=== Test: Large Vector ===")

    size = 1000
    alpha = 3.14
    A = np.random.randn(size).astype(np.float32)
    B = np.random.randn(size).astype(np.float32)

    # NumPy reference
    expected = alpha * A + B

    # MLIR JIT
    result = ch5_vector_ops.saxpy(alpha, A, B)

    print(f"Vector size: {size}")
    print(f"Max difference: {np.max(np.abs(result - expected))}")

    if np.allclose(result, expected):
        print("✓ Test passed!")
    else:
        print("✗ Test failed!")

def test_ir_generation():
    """Show generated MLIR IR."""
    print("\n=== Generated MLIR IR (High-Level) ===")
    ir = ch5_vector_ops.test_ir_generation()
    print(ir)

def test_lowered_ir():
    """Show lowered MLIR IR."""
    print("\n=== Lowered MLIR IR (LLVM Dialect) ===")
    ir = ch5_vector_ops.test_lowered_ir()
    # Only show first 1000 chars to avoid clutter
    print(ir[:1000])
    print("... (truncated)")

if __name__ == "__main__":
    test_saxpy_basic()
    test_saxpy_large()
    test_ir_generation()
    test_lowered_ir()