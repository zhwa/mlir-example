#!/usr/bin/env python3
"""
Test generic execute() API - demonstrates shape-generic binding
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
from graph_builder import Graph
from lowering import MLIRLowering

# Auto-detect build directory
build_dir = None
for path in ['../build/x64-release/ch.8.Custom-dialect', 'build/x64-release/ch.8.Custom-dialect']:
    if os.path.exists(path):
        build_dir = path
        break

if not build_dir:
    print("Error: Could not find build directory")
    sys.exit(1)

sys.path.insert(0, build_dir)
import ch8

print("="*60)
print("Testing Generic execute() API")
print("="*60)
print()

# Test 1: Binary 1D (add)
print("Test 1: Binary 1D addition using generic API")
g = Graph()
x = g.variable([4])
y = g.variable([4])
z = g.add(x, y)

lowering = MLIRLowering()
mlir_text = lowering.lower_graph(g, z, "add")

a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

# Use generic API
result = ch8.execute(mlir_text, "add", [a, b], (4,))
expected = a + b
assert np.allclose(result, expected), f"Expected {expected}, got {result}"
print(f"✓ {a} + {b} = {result}")
print()

# Test 2: Matmul using generic API
print("Test 2: Matrix multiplication using generic API")
g = Graph()
a_var = g.variable([2, 3])
b_var = g.variable([3, 4])
c = g.matmul(a_var, b_var)

mlir_text = lowering.lower_graph(g, c, "matmul")

a = np.array([[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0]], dtype=np.float32)
b = np.array([[1.0, 0.0, 0.0, 1.0],
               [0.0, 1.0, 0.0, 1.0],
               [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)

# Use generic API
result = ch8.execute(mlir_text, "matmul", [a, b], (2, 4))
expected = a @ b
assert np.allclose(result, expected), f"Expected {expected}, got {result}"
print(f"✓ Matmul result shape: {result.shape}")
print(f"✓ First row: {result[0]}")
print()

# Test 3: ReLU using generic API
print("Test 3: ReLU using generic API")
g = Graph()
x = g.variable([2, 4])
y = g.relu(x)

mlir_text = lowering.lower_graph(g, y, "relu")

input_data = np.array([[-1.0, 2.0, -3.0, 4.0],
                        [5.0, -6.0, 7.0, -8.0]], dtype=np.float32)

# Use generic API
result = ch8.execute(mlir_text, "relu", [input_data], (2, 4))
expected = np.maximum(0, input_data)
assert np.allclose(result, expected), f"Expected {expected}, got {result}"
print(f"✓ ReLU({input_data[0]}) = {result[0]}")
print()

# Test 4: Multi-layer network using generic API
print("Test 4: Multi-layer network using generic API")
g = Graph()
x = g.variable([2, 3])
W1 = g.variable([3, 4])
W2 = g.variable([4, 2])
h = g.matmul(x, W1)
h_relu = g.relu(h)
y = g.matmul(h_relu, W2)

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

# Use generic API
result = ch8.execute(mlir_text, "mlp", [x_data, W1_data, W2_data], (2, 2))

h_expected = x_data @ W1_data
h_relu_expected = np.maximum(0, h_expected)
y_expected = h_relu_expected @ W2_data

assert np.allclose(result, y_expected, rtol=1e-5), f"Expected {y_expected}, got {result}"
print(f"✓ MLP output shape: {result.shape}")
print(f"✓ Output: {result}")
print()

print("="*60)
print("All generic API tests passed! ✓")
print("="*60)
print()
print("Key Achievement:")
print("- Single execute() function handles all operations")
print("- No need for execute_binary_1d, execute_matmul, etc.")
print("- Runtime shape introspection")
print("- True shape-generic binding layer!")
