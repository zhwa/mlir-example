#!/usr/bin/env python3
"""Test Softmax implementation using MLIR JIT."""

import sys
import os
import numpy as np

# Auto-detect build directory
build_paths = [
    '../build/x64-release/ch.6.Softmax',
    '../build/x64-debug/ch.6.Softmax',
    'build/x64-release/ch.6.Softmax',
    'build/x64-debug/ch.6.Softmax'
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

import ch6_softmax

def test_softmax_basic():
    """Test basic softmax operation."""
    print("=== Test: Basic Softmax ===")

    input_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # NumPy reference implementation
    def numpy_softmax(x):
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)

    expected = numpy_softmax(input_arr)

    # MLIR JIT version
    result = ch6_softmax.softmax(input_arr)

    print(f"Input = {input_arr}")
    print(f"Result = {result}")
    print(f"Expected = {expected}")
    print(f"Sum = {np.sum(result)} (should be ~1.0)")

    if np.allclose(result, expected, rtol=1e-5):
        print("✓ Test passed!")
    else:
        print("✗ Test failed!")
        print(f"Difference: {result - expected}")

def test_softmax_large_values():
    """Test with large values (numerical stability test)."""
    print("\n=== Test: Large Values (Numerical Stability) ===")

    # Large values that would overflow without max subtraction
    input_arr = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)

    def numpy_softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    expected = numpy_softmax(input_arr)
    result = ch6_softmax.softmax(input_arr)

    print(f"Input = {input_arr}")
    print(f"Result = {result}")
    print(f"Expected = {expected}")

    if np.allclose(result, expected, rtol=1e-5):
        print("✓ Test passed! (No overflow)")
    else:
        print("✗ Test failed!")

def test_softmax_zeros():
    """Test with all zeros (edge case)."""
    print("\n=== Test: All Zeros ===")

    input_arr = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    result = ch6_softmax.softmax(input_arr)
    expected = np.array([1/3, 1/3, 1/3], dtype=np.float32)

    print(f"Input = {input_arr}")
    print(f"Result = {result}")
    print(f"Expected = {expected} (uniform distribution)")

    if np.allclose(result, expected, rtol=1e-5):
        print("✓ Test passed!")
    else:
        print("✗ Test failed!")

def test_softmax_random():
    """Test with random values."""
    print("\n=== Test: Random Values ===")

    size = 100
    input_arr = np.random.randn(size).astype(np.float32)

    def numpy_softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    expected = numpy_softmax(input_arr)
    result = ch6_softmax.softmax(input_arr)

    sum_result = np.sum(result)
    max_diff = np.max(np.abs(result - expected))

    print(f"Vector size: {size}")
    print(f"Sum of probabilities: {sum_result} (should be ~1.0)")
    print(f"Max difference from NumPy: {max_diff}")

    if np.allclose(result, expected, rtol=1e-5) and np.isclose(sum_result, 1.0):
        print("✓ Test passed!")
    else:
        print("✗ Test failed!")

def test_ir_generation():
    """Show generated MLIR IR."""
    print("\n=== Generated MLIR IR (High-Level) ===")
    ir = ch6_softmax.test_ir_generation()
    print(ir[:1500])  # Print first 1500 chars
    print("... (truncated)")

def test_lowered_ir():
    """Show lowered MLIR IR."""
    print("\n=== Lowered MLIR IR (LLVM Dialect) ===")
    ir = ch6_softmax.test_lowered_ir()
    print(ir[:1000])  # Print first 1000 chars
    print("... (truncated)")

if __name__ == "__main__":
    test_softmax_basic()
    test_softmax_large_values()
    test_softmax_zeros()
    test_softmax_random()
    test_ir_generation()
    test_lowered_ir()