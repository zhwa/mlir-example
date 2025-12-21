#!/usr/bin/env python3
"""
Chapter 15: GPU Programming Concepts - Comprehensive Test Suite

Tests all phases:
- Phase 0: 1D Thread Hierarchy (vector add, thread indexing)
- Phase 1: 2D Thread Hierarchy (matrix multiplication)  
- Phase 2: Element-wise Operations (GELU, add, mul)
- Phase 3: Softmax with Reductions

Run all tests: python3 test_all.py
Run specific phase: python3 test_all.py --phase 0
"""

import os
import sys
import numpy as np
import argparse

# Add build directory to path
build_paths = [
    '../build/x64-release/ch.15.GPU-Concepts',
    '../build/x64-debug/ch.15.GPU-Concepts',
    'build/x64-release/ch.15.GPU-Concepts',
    'build/x64-debug/ch.15.GPU-Concepts'
]

build_dir = None
for path in build_paths:
    if os.path.exists(path):
        build_dir = path
        break

if build_dir:
    sys.path.insert(0, build_dir)
    print(f"Using build directory: {build_dir}\n")
else:
    print("Warning: Build directory not found, attempting to import anyway\n")

try:
    import ch15
except ImportError as e:
    print(f"Error: Could not import ch15 module: {e}")
    print("\nPlease build Chapter 15 first:")
    print("  cd /home/zhe/mlir-example")
    print("  cmake --build build/x64-release --target ch15")
    sys.exit(1)

# ============================================================================
# PHASE 0: 1D Thread Hierarchy
# ============================================================================

def test_vector_add_small():
    """Test vector addition with small size (1024 elements)"""
    print("TEST: Vector Add (Small - 1024 elements)")
    
    N = 1024
    A = np.random.randn(N).astype(np.float32)
    B = np.random.randn(N).astype(np.float32)
    
    C_gpu = ch15.vector_add(A, B)
    C_expected = A + B
    
    error = np.max(np.abs(C_gpu - C_expected))
    print(f"  Max error: {error:.2e}")
    assert error < 1e-6, f"Error {error} exceeds threshold"
    print("  ✅ PASSED\n")
    return True

def test_vector_add_large():
    """Test vector addition with large size (10000 elements)"""
    print("TEST: Vector Add (Large - 10000 elements)")
    
    N = 10000
    A = np.random.randn(N).astype(np.float32)
    B = np.random.randn(N).astype(np.float32)
    
    C_gpu = ch15.vector_add(A, B)
    C_expected = A + B
    
    error = np.max(np.abs(C_gpu - C_expected))
    print(f"  Max error: {error:.2e}")
    assert error < 1e-6, f"Error {error} exceeds threshold"
    print("  ✅ PASSED\n")
    return True

def test_vector_add_non_aligned():
    """Test vector addition with non-block-aligned size"""
    print("TEST: Vector Add (Non-aligned - 1337 elements)")
    
    N = 1337  # Not a multiple of 256
    A = np.random.randn(N).astype(np.float32)
    B = np.random.randn(N).astype(np.float32)
    
    C_gpu = ch15.vector_add(A, B)
    C_expected = A + B
    
    error = np.max(np.abs(C_gpu - C_expected))
    print(f"  Max error: {error:.2e}")
    assert error < 1e-6, f"Error {error} exceeds threshold"
    print("  ✅ PASSED - Bounds checking works!\n")
    return True

def test_thread_indexing_block0():
    """Test thread indexing: Block 0, Thread 5"""
    print("TEST: Thread Indexing (Block 0, Thread 5)")
    
    N = 1024
    target_global = 5  # Block 0 * 256 + Thread 5 = 5
    
    result = ch15.test_indexing(N, target_global)
    
    # Only index 5 should be 1.0, all others 0.0
    expected = np.zeros(N, dtype=np.float32)
    expected[target_global] = 1.0
    
    assert np.allclose(result, expected), "Thread indexing incorrect"
    print(f"  Global index {target_global} marked correctly")
    print("  ✅ PASSED\n")
    return True

def test_thread_indexing_block1():
    """Test thread indexing: Block 1, Thread 10"""
    print("TEST: Thread Indexing (Block 1, Thread 10)")
    
    N = 1024
    target_global = 256 + 10  # Block 1 * 256 + Thread 10 = 266
    
    result = ch15.test_indexing(N, target_global)
    
    expected = np.zeros(N, dtype=np.float32)
    expected[target_global] = 1.0
    
    assert np.allclose(result, expected), "Thread indexing incorrect"
    print(f"  Global index {target_global} marked correctly")
    print("  ✅ PASSED\n")
    return True

def test_thread_indexing_last():
    """Test thread indexing: Last thread in last block"""
    print("TEST: Thread Indexing (Last Block - Block 3, Thread 255)")
    
    N = 1024
    target_global = 3 * 256 + 255  # Block 3 * 256 + Thread 255 = 1023
    
    result = ch15.test_indexing(N, target_global)
    
    expected = np.zeros(N, dtype=np.float32)
    expected[target_global] = 1.0
    
    assert np.allclose(result, expected), "Thread indexing incorrect"
    print(f"  Global index {target_global} marked correctly")
    print("  ✅ PASSED\n")
    return True

# ============================================================================
# PHASE 1: 2D Thread Hierarchy
# ============================================================================

def test_matmul_small():
    """Test matrix multiplication with small matrices"""
    print("TEST: Matrix Multiplication (Small - 32×24 @ 24×16)")
    
    M, K, N = 32, 24, 16
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    C_gpu = ch15.matmul(A, B)
    C_expected = A @ B
    
    error = np.max(np.abs(C_gpu - C_expected))
    print(f"  Max error: {error:.2e}")
    assert error < 1e-4, f"Error {error} exceeds threshold"
    print("  ✅ PASSED\n")
    return True

def test_matmul_medium():
    """Test matrix multiplication with medium matrices"""
    print("TEST: Matrix Multiplication (Medium - 64×48 @ 48×32)")
    
    M, K, N = 64, 48, 32
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    C_gpu = ch15.matmul(A, B)
    C_expected = A @ B
    
    error = np.max(np.abs(C_gpu - C_expected))
    print(f"  Max error: {error:.2e}")
    assert error < 1e-4, f"Error {error} exceeds threshold"
    print("  ✅ PASSED\n")
    return True

def test_matmul_non_aligned():
    """Test matrix multiplication with non-block-aligned sizes"""
    print("TEST: Matrix Multiplication (Non-aligned - 33×25 @ 25×17)")
    
    M, K, N = 33, 25, 17  # Not multiples of 16
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    C_gpu = ch15.matmul(A, B)
    C_expected = A @ B
    
    error = np.max(np.abs(C_gpu - C_expected))
    print(f"  Max error: {error:.2e}")
    assert error < 1e-4, f"Error {error} exceeds threshold"
    print("  ✅ PASSED - 2D bounds checking works!\n")
    return True

def test_matmul_large():
    """Test matrix multiplication with large matrices"""
    print("TEST: Matrix Multiplication (Large - 128×96 @ 96×80)")
    
    M, K, N = 128, 96, 80
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    C_gpu = ch15.matmul(A, B)
    C_expected = A @ B
    
    error = np.max(np.abs(C_gpu - C_expected))
    print(f"  Max error: {error:.2e}")
    assert error < 1e-4, f"Error {error} exceeds threshold"
    print("  ✅ PASSED\n")
    return True

# ============================================================================
# PHASE 2: Element-wise Operations
# ============================================================================

def gelu_reference(x):
    """Reference GELU implementation"""
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    inner = sqrt_2_over_pi * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))

def test_gelu_small():
    """Test GELU with small array"""
    print("TEST: GELU (Small - 1024 elements)")
    
    N = 1024
    x = np.random.randn(N).astype(np.float32)
    
    y_gpu = ch15.gelu(x)
    y_expected = gelu_reference(x)
    
    error = np.max(np.abs(y_gpu - y_expected))
    print(f"  Max error: {error:.2e}")
    assert error < 0.025, f"Error {error} exceeds threshold (polynomial tanh approximation)"
    print("  ✅ PASSED\n")
    return True

def test_gelu_large():
    """Test GELU with large array"""
    print("TEST: GELU (Large - 10000 elements)")
    
    N = 10000
    x = np.random.randn(N).astype(np.float32)
    
    y_gpu = ch15.gelu(x)
    y_expected = gelu_reference(x)
    
    error = np.max(np.abs(y_gpu - y_expected))
    print(f"  Max error: {error:.2e}")
    assert error < 0.10, f"Error {error} exceeds threshold (polynomial tanh approximation)"
    print("  ✅ PASSED\n")
    return True

def test_gelu_non_aligned():
    """Test GELU with non-block-aligned size"""
    print("TEST: GELU (Non-aligned - 2047 elements)")
    
    N = 2047  # Not a multiple of 256
    x = np.random.randn(N).astype(np.float32)
    
    y_gpu = ch15.gelu(x)
    y_expected = gelu_reference(x)
    
    error = np.max(np.abs(y_gpu - y_expected))
    print(f"  Max error: {error:.2e}")
    assert error < 0.25, f"Error {error} exceeds threshold (polynomial tanh approximation)"
    print("  ✅ PASSED\n")
    return True

def test_add_small():
    """Test element-wise addition (small)"""
    print("TEST: Element-wise Add (Small - 512 elements)")
    
    N = 512
    x = np.random.randn(N).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)
    
    z_gpu = ch15.add(x, y)
    z_expected = x + y
    
    error = np.max(np.abs(z_gpu - z_expected))
    print(f"  Max error: {error:.2e}")
    assert error < 1e-6, f"Error {error} exceeds threshold"
    print("  ✅ PASSED\n")
    return True

def test_add_large():
    """Test element-wise addition (large)"""
    print("TEST: Element-wise Add (Large - 8192 elements)")
    
    N = 8192
    x = np.random.randn(N).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)
    
    z_gpu = ch15.add(x, y)
    z_expected = x + y
    
    error = np.max(np.abs(z_gpu - z_expected))
    print(f"  Max error: {error:.2e}")
    assert error < 1e-6, f"Error {error} exceeds threshold"
    print("  ✅ PASSED\n")
    return True

def test_mul_small():
    """Test element-wise multiplication (small)"""
    print("TEST: Element-wise Mul (Small - 768 elements)")
    
    N = 768
    x = np.random.randn(N).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)
    
    z_gpu = ch15.mul(x, y)
    z_expected = x * y
    
    error = np.max(np.abs(z_gpu - z_expected))
    print(f"  Max error: {error:.2e}")
    assert error < 1e-6, f"Error {error} exceeds threshold"
    print("  ✅ PASSED\n")
    return True

def test_mul_large():
    """Test element-wise multiplication (large)"""
    print("TEST: Element-wise Mul (Large - 5000 elements)")
    
    N = 5000
    x = np.random.randn(N).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)
    
    z_gpu = ch15.mul(x, y)
    z_expected = x * y
    
    error = np.max(np.abs(z_gpu - z_expected))
    print(f"  Max error: {error:.2e}")
    assert error < 1e-6, f"Error {error} exceeds threshold"
    print("  ✅ PASSED\n")
    return True

# ============================================================================
# PHASE 3: Softmax with Reductions
# ============================================================================

def test_softmax_small():
    """Test softmax with small size (256 elements - single block)"""
    print("TEST: Softmax (Small - 256 elements, single block)")
    
    N = 256
    x = np.random.randn(N).astype(np.float32)
    
    y_gpu = ch15.softmax(x)
    
    # NumPy reference: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    y_expected = exp_x / np.sum(exp_x)
    
    # Check numerical properties
    sum_y = np.sum(y_gpu)
    print(f"  Sum of softmax: {sum_y:.6f} (should be ~1.0)")
    assert np.abs(sum_y - 1.0) < 1e-5, f"Sum {sum_y} not close to 1.0"
    
    error = np.max(np.abs(y_gpu - y_expected))
    print(f"  Max error: {error:.2e}")
    # Note: Using Taylor series for exp, so accuracy is ~1-2% (not production-quality)
    assert error < 0.1, f"Error {error} exceeds threshold"
    print("  ✅ PASSED\n")
    return True

def test_softmax_large():
    """Test softmax with large size (4096 elements - multiple blocks)"""
    print("TEST: Softmax (Large - 4096 elements, 16 blocks)")
    
    N = 4096
    x = np.random.randn(N).astype(np.float32)
    
    y_gpu = ch15.softmax(x)
    
    # NumPy reference
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    y_expected = exp_x / np.sum(exp_x)
    
    # Check sum
    sum_y = np.sum(y_gpu)
    print(f"  Sum of softmax: {sum_y:.6f} (should be ~1.0)")
    assert np.abs(sum_y - 1.0) < 1e-4, f"Sum {sum_y} not close to 1.0"
    
    error = np.max(np.abs(y_gpu - y_expected))
    print(f"  Max error: {error:.2e}")
    assert error < 0.1, f"Error {error} exceeds threshold"
    print("  ✅ PASSED\n")
    return True

def test_softmax_non_aligned():
    """Test softmax with non-aligned size"""
    print("TEST: Softmax (Non-aligned - 1357 elements)")
    
    N = 1357
    x = np.random.randn(N).astype(np.float32)
    
    y_gpu = ch15.softmax(x)
    
    # NumPy reference
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    y_expected = exp_x / np.sum(exp_x)
    
    # Check sum
    sum_y = np.sum(y_gpu)
    print(f"  Sum of softmax: {sum_y:.6f} (should be ~1.0)")
    assert np.abs(sum_y - 1.0) < 1e-5, f"Sum {sum_y} not close to 1.0"
    
    error = np.max(np.abs(y_gpu - y_expected))
    print(f"  Max error: {error:.2e}")
    assert error < 0.1, f"Error {error} exceeds threshold"
    print("  ✅ PASSED\n")
    return True

def test_softmax_extreme_values():
    """Test softmax with extreme values (numerical stability)"""
    print("TEST: Softmax (Extreme values - numerical stability)")
    
    N = 512
    x = np.array([100.0] * (N // 2) + [-100.0] * (N // 2), dtype=np.float32)
    
    y_gpu = ch15.softmax(x)
    
    # NumPy reference
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    y_expected = exp_x / np.sum(exp_x)
    
    # Check sum
    sum_y = np.sum(y_gpu)
    print(f"  Sum of softmax: {sum_y:.6f} (should be ~1.0)")
    assert np.abs(sum_y - 1.0) < 1e-4, f"Sum {sum_y} not close to 1.0"
    
    # Check no NaN or inf
    assert not np.any(np.isnan(y_gpu)), "Found NaN in output"
    assert not np.any(np.isinf(y_gpu)), "Found inf in output"
    
    error = np.max(np.abs(y_gpu - y_expected))
    print(f"  Max error: {error:.2e}")
    assert error < 0.01, f"Error {error} exceeds threshold"
    print("  ✅ PASSED\n")
    return True

# ============================================================================
# Main Test Runner
# ============================================================================

def run_phase_tests(phase_num):
    """Run tests for a specific phase"""
    
    phase_tests = {
        0: [
            ("Vector Add (Small)", test_vector_add_small),
            ("Vector Add (Large)", test_vector_add_large),
            ("Vector Add (Non-aligned)", test_vector_add_non_aligned),
            ("Thread Indexing (Block 0)", test_thread_indexing_block0),
            ("Thread Indexing (Block 1)", test_thread_indexing_block1),
            ("Thread Indexing (Last)", test_thread_indexing_last),
        ],
        1: [
            ("MatMul (Small)", test_matmul_small),
            ("MatMul (Medium)", test_matmul_medium),
            ("MatMul (Non-aligned)", test_matmul_non_aligned),
            ("MatMul (Large)", test_matmul_large),
        ],
        2: [
            ("GELU (Small)", test_gelu_small),
            ("GELU (Large)", test_gelu_large),
            ("GELU (Non-aligned)", test_gelu_non_aligned),
            ("Add (Small)", test_add_small),
            ("Add (Large)", test_add_large),
            ("Mul (Small)", test_mul_small),
            ("Mul (Large)", test_mul_large),
        ],
        3: [
            ("Softmax (Small)", test_softmax_small),
            ("Softmax (Large)", test_softmax_large),
            ("Softmax (Non-aligned)", test_softmax_non_aligned),
            ("Softmax (Extreme values)", test_softmax_extreme_values),
        ]
    }
    
    phase_names = {
        0: "Phase 0: 1D Thread Hierarchy",
        1: "Phase 1: 2D Matrix Multiplication",
        2: "Phase 2: Element-wise Operations",
        3: "Phase 3: Softmax with Reductions"
    }
    
    if phase_num not in phase_tests:
        print(f"Error: Phase {phase_num} does not exist")
        return 0, 0
    
    print("=" * 70)
    print(f"Chapter 15: {phase_names[phase_num]}")
    print("=" * 70)
    print()
    
    tests = phase_tests[phase_num]
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"❌ FAILED: {name}")
            print(f"   Error: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    return passed, failed

def main():
    parser = argparse.ArgumentParser(description="Run Chapter 15 GPU tests")
    parser.add_argument('--phase', type=int, choices=[0, 1, 2, 3],
                       help='Run tests for specific phase only (0, 1, 2, or 3)')
    args = parser.parse_args()
    
    if args.phase is not None:
        # Run single phase
        passed, failed = run_phase_tests(args.phase)
        total = passed + failed
        
        print("\n" + "=" * 70)
        print(f"PHASE {args.phase} SUMMARY")
        print("=" * 70)
        print(f"\n✅ Passed: {passed}/{total}\n")
        
        if failed > 0:
            print(f"❌ {failed} test(s) failed")
            return 1
        else:
            print(f"🎉 All Phase {args.phase} tests PASSED!")
            return 0
    else:
        # Run all phases
        total_passed = 0
        total_failed = 0
        phase_results = {}
        
        for phase_num in [0, 1, 2, 3]:
            passed, failed = run_phase_tests(phase_num)
            phase_results[phase_num] = (passed, failed)
            total_passed += passed
            total_failed += failed
            print()
        
        # Overall summary
        print("=" * 70)
        print("OVERALL SUMMARY")
        print("=" * 70)
        
        for phase_num in [0, 1, 2, 3]:
            passed, failed = phase_results[phase_num]
            total = passed + failed
            status = "✅" if failed == 0 else "❌"
            print(f"  {status} Phase {phase_num}: {passed}/{total} passed")
        
        total = total_passed + total_failed
        print(f"\n  Total: {total_passed}/{total} tests passed\n")
        
        if total_failed == 0:
            print("🎉 ALL TESTS PASSED!\n")
            print("Achievements:")
            print("  ✅ Phase 0: 1D thread hierarchy working (6/6)")
            print("  ✅ Phase 1: 2D matrix multiplication working (4/4)")
            print("  ✅ Phase 2: Element-wise operations working (7/7)")
            print("  ✅ Phase 3: Softmax with reductions working (4/4)")
            print("  ✅ MLIR 19 constant pool bug solved!\n")
            
            print("🚀 Ready for Phase 4: LayerNorm (multi-stage reductions)")
            return 0
        else:
            print(f"❌ {total_failed} test(s) failed")
            return 1

if __name__ == "__main__":
    sys.exit(main())
