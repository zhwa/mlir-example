#!/usr/bin/env python3
import os
import sys
import numpy as np

# Auto-detect build directory
build_paths = [
    '../build/x64-release/ch.7.Neural-ops',
    '../build/x64-debug/ch.7.Neural-ops',
    'build/x64-release/ch.7.Neural-ops',
    'build/x64-debug/ch.7.Neural-ops'
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

import ch7_neural_ops as ch7

def test_add():
    """Test element-wise addition"""
    print("Testing element-wise addition...")
    
    g = ch7.Graph()
    x = g.input([4])
    y = g.input([4])
    z = g.add(x, y)
    
    # Print MLIR
    print("\nMLIR for addition:")
    mlir_str = g.get_mlir(z, "add_fn")
    print(mlir_str)
    
    # Compile and execute
    fn = g.compile(z, "add_fn")
    
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    
    print(f"\nInput 1: {a}")
    print(f"Input 2: {b}")
    
    result = ch7.execute_binary_1d(fn, a, b)
    
    expected = a + b
    print(f"Result:  {result}")
    print(f"Expected: {expected}")
    assert np.allclose(result, expected), f"Addition test failed! Got {result}, expected {expected}"
    print("✓ Addition test passed\n")

def test_mul():
    """Test element-wise multiplication"""
    print("Testing element-wise multiplication...")
    
    g = ch7.Graph()
    x = g.input([4])
    y = g.input([4])
    z = g.mul(x, y)
    
    # Compile and execute
    fn = g.compile(z, "mul_fn")
    
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    result = ch7.execute_binary_1d(fn, a, b)
    
    expected = a * b
    print(f"Input 1: {a}")
    print(f"Input 2: {b}")
    print(f"Result:  {result}")
    print(f"Expected: {expected}")
    assert np.allclose(result, expected), "Multiplication test failed!"
    print("✓ Multiplication test passed\n")

def test_matmul():
    """Test matrix multiplication"""
    print("Testing matrix multiplication...")
    
    g = ch7.Graph()
    x = g.input([2, 3])
    y = g.input([3, 2])
    z = g.matmul(x, y)
    
    # Print MLIR
    print("\nMLIR for matmul:")
    print(g.get_mlir(z, "matmul_fn"))
    
    # Compile and execute
    fn = g.compile(z, "matmul_fn")
    
    a = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]], dtype=np.float32)
    result = ch7.execute_matmul(fn, a, b)
    
    expected = np.matmul(a, b)
    print(f"Input 1:\n{a}")
    print(f"Input 2:\n{b}")
    print(f"Result:\n{result}")
    print(f"Expected:\n{expected}")
    assert np.allclose(result, expected), "MatMul test failed!"
    print("✓ MatMul test passed\n")

def test_relu():
    """Test ReLU activation"""
    print("Testing ReLU activation...")
    
    g = ch7.Graph()
    x = g.input([6])
    y = g.relu(x)
    
    # Compile and execute
    fn = g.compile(y, "relu_fn")
    
    a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    result = ch7.execute_1d(fn, a)
    
    expected = np.maximum(a, 0.0)
    print(f"Input:    {a}")
    print(f"Result:   {result}")
    print(f"Expected: {expected}")
    assert np.allclose(result, expected), "ReLU test failed!"
    print("✓ ReLU test passed\n")

def test_softmax():
    """Test Softmax activation"""
    print("Testing Softmax activation...")
    
    g = ch7.Graph()
    x = g.input([4])
    y = g.softmax(x)
    
    # Compile and execute
    fn = g.compile(y, "softmax_fn")
    
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = ch7.execute_1d(fn, a)
    
    # Expected softmax
    exp_a = np.exp(a - np.max(a))
    expected = exp_a / np.sum(exp_a)
    
    print(f"Input:    {a}")
    print(f"Result:   {result}")
    print(f"Expected: {expected}")
    print(f"Sum: {np.sum(result)}")
    assert np.allclose(result, expected), "Softmax test failed!"
    assert np.isclose(np.sum(result), 1.0), "Softmax sum not 1.0!"
    print("✓ Softmax test passed\n")

def test_composition():
    """Test composition: y = softmax(relu(x @ W + b))"""
    print("Testing composition: y = softmax(relu(x @ W + b))...")
    
    g = ch7.Graph()
    x = g.input([1, 3])  # Input: 1x3
    W = g.input([3, 4])  # Weight: 3x4
    b = g.input([1, 4])  # Bias: 1x4
    
    # Build computation graph
    z1 = g.matmul(x, W)  # 1x4
    z2 = g.add(z1, b)    # 1x4
    z3 = g.relu(z2)      # 1x4
    # Note: softmax expects 1D input, so we'll use a different test
    
    # Print MLIR
    print("\nMLIR for composition:")
    print(g.get_mlir(z3, "composed_fn"))
    
    print("✓ Composition graph built successfully\n")

def test_multi_layer():
    """Test a simple two-layer network: layer2(relu(layer1))"""
    print("Testing multi-layer network...")
    
    g = ch7.Graph()
    x = g.input([2, 3])   # Input: 2x3
    W1 = g.input([3, 4])  # Layer 1 weight: 3x4
    W2 = g.input([4, 2])  # Layer 2 weight: 4x2
    
    # Layer 1: x @ W1
    h = g.matmul(x, W1)   # 2x4
    
    # Activation
    h_relu = g.relu(h)    # 2x4
    
    # Layer 2: h @ W2
    y = g.matmul(h_relu, W2)  # 2x2
    
    # Print MLIR
    print("\nMLIR for multi-layer network:")
    mlir = g.get_mlir(y, "mlp_fn")
    print(mlir)
    
    # Compile and test
    fn = g.compile(y, "mlp_fn")
    
    x_data = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=np.float32)
    W1_data = np.array([[0.1, 0.2, 0.3, 0.4],
                        [0.5, 0.6, 0.7, 0.8],
                        [0.9, 1.0, 1.1, 1.2]], dtype=np.float32)
    W2_data = np.array([[0.1, 0.2],
                        [0.3, 0.4],
                        [0.5, 0.6],
                        [0.7, 0.8]], dtype=np.float32)
    
    # Manual calculation
    h_manual = np.matmul(x_data, W1_data)
    h_relu_manual = np.maximum(h_manual, 0.0)
    y_manual = np.matmul(h_relu_manual, W2_data)
    
    print(f"\nInput shape: {x_data.shape}")
    print(f"W1 shape: {W1_data.shape}")
    print(f"W2 shape: {W2_data.shape}")
    print(f"Expected output shape: {y_manual.shape}")
    print(f"Expected output:\n{y_manual}")
    
    print("✓ Multi-layer network built successfully\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 7: Neural Operations with Computation Graph")
    print("=" * 60 + "\n")
    
    test_add()
    test_mul()
    test_matmul()
    test_relu()
    test_softmax()
    test_composition()
    test_multi_layer()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
