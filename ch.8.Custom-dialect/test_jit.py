#!/usr/bin/env python3
"""
Chapter 8: Custom Dialect - Complete Test Suite

Tests all operations end-to-end: Python API → nn dialect → lowering → compilation → execution
Uses libffi-based universal execute() function that handles ANY signature.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
from graph_builder import Graph
from lowering import MLIRLowering

# Auto-detect build directory
build_paths = [
    '../build/x64-release/ch.8.Custom-dialect',
    '../build/x64-debug/ch.8.Custom-dialect',
    'build/x64-release/ch.8.Custom-dialect',
    'build/x64-debug/ch.8.Custom-dialect'
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

import ch8

def test_add():
    """Test element-wise addition"""
    print("### Test 1: Element-wise Addition ###")

    # Build graph
    g = Graph()
    x = g.variable([4])
    y = g.variable([4])
    z = g.add(x, y)

    # Lower to standard MLIR
    lowering = MLIRLowering()
    mlir_text = lowering.lower_graph(g, z, "add")

    # Execute
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    result = ch8.execute(mlir_text, "add", [a, b], (4,))

    expected = a + b
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print(f"✓ Input: {a} + {b}")
    print(f"✓ Output: {result}")
    print()

def test_mul():
    """Test element-wise multiplication"""
    print("### Test 2: Element-wise Multiplication ###")

    g = Graph()
    x = g.variable([4])
    y = g.variable([4])
    z = g.mul(x, y)

    lowering = MLIRLowering()
    mlir_text = lowering.lower_graph(g, z, "mul")

    a = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    b = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    result = ch8.execute(mlir_text, "mul", [a, b], (4,))

    expected = a * b
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print(f"✓ Input: {a} * {b}")
    print(f"✓ Output: {result}")
    print()

def test_matmul():
    """Test matrix multiplication"""
    print("### Test 3: Matrix Multiplication ###")

    g = Graph()
    a_var = g.variable([2, 3])
    b_var = g.variable([3, 4])
    c = g.matmul(a_var, b_var)

    lowering = MLIRLowering()
    mlir_text = lowering.lower_graph(g, c, "matmul")

    a = np.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0, 0.0, 1.0],
                   [0.0, 1.0, 0.0, 1.0],
                   [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)

    result = ch8.execute(mlir_text, "matmul", [a, b], (2, 4))

    expected = a @ b
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print(f"✓ Input shapes: {a.shape} @ {b.shape}")
    print(f"✓ Output shape: {result.shape}")
    print(f"✓ Correct result: {result[0]}")
    print()

def test_relu():
    """Test ReLU activation"""
    print("### Test 4: ReLU Activation ###")

    g = Graph()
    x = g.variable([2, 4])
    y = g.relu(x)

    lowering = MLIRLowering()
    mlir_text = lowering.lower_graph(g, y, "relu")

    input_data = np.array([[-1.0, 2.0, -3.0, 4.0],
                            [5.0, -6.0, 7.0, -8.0]], dtype=np.float32)

    result = ch8.execute(mlir_text, "relu", [input_data], (2, 4))

    expected = np.maximum(0, input_data)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print(f"✓ Input: {input_data[0]}")
    print(f"✓ Output: {result[0]} (negatives zeroed)")
    print()

def test_multi_layer():
    """Test multi-layer neural network"""
    print("### Test 5: Multi-layer Network (3 inputs, 28 params) ###")

    g = Graph()
    x = g.variable([2, 3])
    W1 = g.variable([3, 4])
    W2 = g.variable([4, 2])
    h = g.matmul(x, W1)
    h_relu = g.relu(h)
    y = g.matmul(h_relu, W2)

    lowering = MLIRLowering()
    mlir_text = lowering.lower_graph(g, y, "mlp")

    x_data = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]], dtype=np.float32)
    W1_data = np.array([[0.1, 0.2, 0.3, 0.4],
                         [0.5, 0.6, 0.7, 0.8],
                         [0.9, 1.0, 1.1, 1.2]], dtype=np.float32)
    W2_data = np.array([[1.0, 0.0],
                         [0.0, 1.0],
                         [1.0, 1.0],
                         [0.5, 0.5]], dtype=np.float32)

    result = ch8.execute(mlir_text, "mlp", [x_data, W1_data, W2_data], (2, 2))

    # Compute expected result
    h_expected = x_data @ W1_data
    h_relu_expected = np.maximum(0, h_expected)
    y_expected = h_relu_expected @ W2_data

    assert np.allclose(result, y_expected, rtol=1e-5), f"Expected {y_expected}, got {result}"
    print(f"✓ Input shape: {x_data.shape}")
    print(f"✓ Hidden layer: {h_expected.shape} → ReLU → {h_relu_expected.shape}")
    print(f"✓ Output: {result}")
    print(f"✓ Expected: {y_expected}")
    print()

def test_raw_mlir():
    """Test direct MLIR text execution (demonstrates libffi flexibility)"""
    print("### Test 6: Raw MLIR Text (libffi flexibility) ###")

    # Direct MLIR without high-level API - tests universal execution
    mlir_add = """
module {
  func.func @add(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>) {
    linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, 
                         affine_map<(d0) -> (d0)>, 
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
    } ins(%arg0, %arg1 : memref<4xf32>, memref<4xf32>) 
      outs(%arg2 : memref<4xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
    }
    return
  }
}
"""

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    result = ch8.execute(mlir_add, "add", [a, b], (4,))

    expected = a + b
    assert np.allclose(result, expected)
    print(f"✓ Direct MLIR: {a} + {b} = {result}")
    print(f"✓ libffi handles raw MLIR text seamlessly")
    print()

if __name__ == "__main__":
    print("="*60)
    print("Chapter 8: Custom Dialect - Full Test Suite")
    print("Using libffi-based universal execute()")
    print("="*60)
    print()

    test_add()
    test_mul()
    test_matmul()
    test_relu()
    test_multi_layer()
    test_raw_mlir()

    print("="*60)
    print("All 6 tests passed! ✓")
    print("="*60)
    print()
    print("Summary:")
    print("- High-level nn dialect API works perfectly")
    print("- Python lowering to standard MLIR works")
    print("- C++ compilation and JIT execution works")
    print("- All operations produce correct results")
    print("- libffi-based execute() handles ANY signature universally")
    print()
    print("Key Achievement:")
    print("libffi eliminates ALL explicit parameter count cases.")
    print("Single universal execute() handles arbitrary signatures through")
    print("dynamic FFI dispatch - production-grade flexibility!")