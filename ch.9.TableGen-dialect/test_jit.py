#!/usr/bin/env python3
"""
Chapter 9: Custom Dialect with TableGen - Test Suite

Demonstrates the NN dialect defined in C++ with TableGen, lowered to standard MLIR.
"""

import sys
import os
import numpy as np

# Add build directory to path
build_paths = [
    '../build/x64-release/ch.9.TableGen-dialect',
    '../build/x64-debug/ch.9.TableGen-dialect',
    'build/x64-release/ch.9.TableGen-dialect',
    'build/x64-debug/ch.9.TableGen-dialect'
]

build_dir = None
for path in build_paths:
    if os.path.exists(path):
        build_dir = path
        break

if not build_dir:
    print("Error: Could not find build directory")
    sys.exit(1)

print(f"Using build directory: {build_dir}")
sys.path.insert(0, build_dir)

import ch9

def test_nn_add():
    """Test nn.add operation"""
    print("### Test 1: NN Add ###")

    mlir_code = """
    module {
      func.func @add(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>) {
        nn.add %arg0, %arg1, %arg2 : memref<4xf32>, memref<4xf32>, memref<4xf32>
        return
      }
    }
    """

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    result = ch9.execute(mlir_code, "add", [a, b], (4,))

    expected = a + b
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print(f"✓ {a} + {b} = {result}")
    print()

def test_nn_mul():
    """Test nn.mul operation"""
    print("### Test 2: NN Mul ###")

    mlir_code = """
    module {
      func.func @mul(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>) {
        nn.mul %arg0, %arg1, %arg2 : memref<4xf32>, memref<4xf32>, memref<4xf32>
        return
      }
    }
    """

    a = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    b = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    result = ch9.execute(mlir_code, "mul", [a, b], (4,))

    expected = a * b
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print(f"✓ {a} * {b} = {result}")
    print()

def test_nn_matmul():
    """Test nn.matmul operation"""
    print("### Test 3: NN MatMul ###")

    mlir_code = """
    module {
      func.func @matmul(%arg0: memref<2x3xf32>, %arg1: memref<3x4xf32>, %arg2: memref<2x4xf32>) {
        nn.matmul %arg0, %arg1, %arg2 : memref<2x3xf32>, memref<3x4xf32>, memref<2x4xf32>
        return
      }
    }
    """

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0, 1.0],
                  [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    result = ch9.execute(mlir_code, "matmul", [a, b], (2, 4))

    expected = a @ b
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print(f"✓ MatMul: {a.shape} @ {b.shape} = {result.shape}")
    print(f"  Result: {result[0]}")
    print()

def test_nn_relu():
    """Test nn.relu operation"""
    print("### Test 4: NN ReLU ###")

    mlir_code = """
    module {
      func.func @relu(%arg0: memref<2x4xf32>, %arg1: memref<2x4xf32>) {
        nn.relu %arg0, %arg1 : memref<2x4xf32>, memref<2x4xf32>
        return
      }
    }
    """

    input_data = np.array([[-1.0, 2.0, -3.0, 4.0],
                           [5.0, -6.0, 7.0, -8.0]], dtype=np.float32)
    result = ch9.execute(mlir_code, "relu", [input_data], (2, 4))

    expected = np.maximum(0, input_data)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print(f"✓ Input:  {input_data[0]}")
    print(f"  Output: {result[0]}")
    print()

if __name__ == "__main__":
    print("="*60)
    print("Chapter 9: Custom Dialect with TableGen")
    print("="*60)
    print()

    test_nn_add()
    test_nn_mul()
    test_nn_matmul()
    test_nn_relu()

    print("="*60)
    print("All tests passed! ✓")
    print("="*60)
    print()
    print("Summary:")
    print("- NN dialect defined with TableGen (ODS)")
    print("- Operations: add, mul, matmul, relu")
    print("- Lowering to linalg/arith works correctly")
    print("- Full compilation pipeline functional")
    print()
    print("Key Achievement:")
    print("Production-grade custom dialect using MLIR's TableGen!")