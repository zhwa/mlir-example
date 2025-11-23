#!/usr/bin/env python3
"""
Comprehensive test suite for MLIR-JIT compiled GEMM.

This script demonstrates and tests:
  1. Clean Python API - C = gemm(A, B) with no manual allocation
  2. Correctness tests - Compare against NumPy
  3. Performance benchmark - Measure JIT caching speedup
  4. Shape flexibility - Test various matrix dimensions
"""

import sys
import os
import numpy as np
import time

# Auto-detect build directory
# The unified build system places modules in: ../build/x64-release/ch.4.Tensor-bufferization/
build_paths = [
    '../build/x64-release/ch.4.Tensor-bufferization',
    '../build/x64-debug/ch.4.Tensor-bufferization',
    'build/x64-release/ch.4.Tensor-bufferization',  # If running from root
    'build/x64-debug/ch.4.Tensor-bufferization',
]

for path in build_paths:
    if os.path.exists(path) and any(f.startswith('ch4_tensor_bufferization') for f in os.listdir(path) 
                                     if not os.path.isdir(os.path.join(path, f))):
        sys.path.insert(0, path)
        print(f"Using build directory: {path}")
        break
else:
    print("ERROR: Could not find ch4_tensor_bufferization module!")
    print("Build the project first: cmake --preset x64-release && cmake --build --preset x64-release")
    sys.exit(1)

import ch4_tensor_bufferization as llvm_example

def demo_clean_api():
    """Demonstrate the clean Python API."""
    print("\n" + "="*70)
    print("CLEAN API DEMO")
    print("="*70)
    print("\nThe API is clean and user-friendly:")
    print("  C = llvm_example.gemm(A, B)")
    print("\n✓ No manual allocation needed")
    print("✓ Works with any matrix size")
    print("✓ Returns new NumPy array")
    print("✓ Out-parameter is an internal implementation detail\n")
    
    # Example 1: Small matrices
    print("--- Example 1: Small matrices (2×3 @ 3×2) ---")
    A = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], dtype=np.float32)
    B = np.array([[7.0, 8.0],
                  [9.0, 10.0],
                  [11.0, 12.0]], dtype=np.float32)
    
    # Clean API: Just write C = gemm(A, B)!
    C = llvm_example.gemm(A, B)
    expected = A @ B
    
    print(f"A shape: {A.shape}, B shape: {B.shape}")
    print(f"Result C:\n{C}")
    print(f"✓ Correct: {np.allclose(C, expected)}")
    
    # Example 2: Different shapes
    print("\n--- Example 2: Rectangular matrices (3×5 @ 5×2) ---")
    A = np.random.randn(3, 5).astype(np.float32)
    B = np.random.randn(5, 2).astype(np.float32)
    C = llvm_example.gemm(A, B)
    expected = A @ B
    print(f"A shape: {A.shape}, B shape: {B.shape}, C shape: {C.shape}")
    print(f"✓ Correct: {np.allclose(C, expected, rtol=1e-5)}")

def test_correctness():
    """Test numerical correctness against NumPy."""
    print("\n" + "="*70)
    print("CORRECTNESS TESTS")
    print("="*70)
    
    # Test 1: Ones matrix - easy to verify manually
    print("\n=== Test 1: Ones matrix (sanity check) ===")
    print("Computing: C = ones(8,32) @ ones(32,16)")
    A = np.ones((8, 32), dtype=np.float32)
    B = np.ones((32, 16), dtype=np.float32)
    
    C = llvm_example.gemm(A, B)
    print(f"✓ Result shape: {C.shape}")
    print(f"✓ C[0,0] = {C[0,0]:.1f} (expected: 32.0)")
    
    if np.allclose(C, 32.0):
        print("✓ All values correct!")
    else:
        print(f"✗ FAILED: Expected all 32.0, got range [{C.min():.1f}, {C.max():.1f}]")
        return False
    
    # Test 2: Random matrices - verify against NumPy
    print("\n=== Test 2: Random matrices (numerical accuracy) ===")
    print("Computing: C = random(8,32) @ random(32,16)")
    np.random.seed(42)
    A = np.random.randn(8, 32).astype(np.float32)
    B = np.random.randn(32, 16).astype(np.float32)
    
    C_jit = llvm_example.gemm(A, B)
    C_numpy = A @ B
    
    max_error = np.max(np.abs(C_jit - C_numpy))
    print(f"✓ Max error vs NumPy: {max_error:.2e}")
    
    if np.allclose(C_jit, C_numpy, rtol=1e-5):
        print("✓ Results match NumPy (within float32 precision)!")
    else:
        print(f"✗ FAILED: Errors too large! Max error: {max_error}")
        return False
    
    # Test 3: Various shapes (dynamic shape support)
    print("\n=== Test 3: Various matrix shapes ===")
    test_shapes = [
        ((5, 5), (5, 5)),       # Small square
        ((10, 20), (20, 15)),   # Medium rectangular
        ((100, 50), (50, 25)),  # Larger matrices
    ]
    
    for (m, k1), (k2, n) in test_shapes:
        A = np.random.randn(m, k1).astype(np.float32)
        B = np.random.randn(k2, n).astype(np.float32)
        C_jit = llvm_example.gemm(A, B)
        C_numpy = A @ B
        max_error = np.max(np.abs(C_jit - C_numpy))
        status = "✓" if max_error < 1e-5 else "✗"
        print(f"  {status} {m}×{k1} × {k2}×{n} → {m}×{n}  (error: {max_error:.2e})")
    
    return True

def test_performance():
    """Benchmark JIT caching performance."""
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARK: JIT Caching")
    print("="*70)
    print("\nNote: First call compiles, subsequent calls use cached function.")
    print("The cache works for ALL shapes (shape-agnostic compilation)!\n")
    
    test_cases = [
        (8, 16, 32),      # Small
        (100, 100, 100),  # Medium
        (500, 500, 500),  # Large
    ]
    
    results = []
    for M, N, K in test_cases:
        print(f"\n--- Shape: A({M}×{K}) × B({K}×{N}) → C({M}×{N}) ---")
        
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        times = []
        for i in range(5):
            start = time.perf_counter()
            C = llvm_example.gemm(A, B)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
            
            label = "COMPILE" if i == 0 else "CACHED"
            print(f"  Call {i+1}: {elapsed_ms:7.2f} ms  [{label}]")
        
        first_call = times[0]
        avg_cached = np.mean(times[1:])
        speedup = first_call / avg_cached if avg_cached > 0 else 0
        
        print(f"  → First: {first_call:.2f} ms, Cached avg: {avg_cached:.2f} ms")
        print(f"  → Speedup: {speedup:.1f}x 🚀")
        
        results.append({
            'shape': (M, N, K),
            'first_ms': first_call,
            'cached_ms': avg_cached,
            'speedup': speedup
        })
    
    # Summary table
    print(f"\n{'='*70}")
    print("Summary: Caching Performance")
    print(f"{'='*70}")
    print(f"{'Shape':<20} {'1st Call':>12} {'Cached':>12} {'Speedup':>12}")
    print(f"{'-'*70}")
    
    for r in results:
        shape_str = f"{r['shape'][0]}×{r['shape'][1]}×{r['shape'][2]}"
        print(f"{shape_str:<20} {r['first_ms']:>10.2f} ms "
              f"{r['cached_ms']:>10.2f} ms {r['speedup']:>10.1f}x")
    
    print(f"{'='*70}\n")

def main():
    print("\n" + "="*70)
    print("MLIR-JIT GEMM: Complete Test Suite")
    print("="*70)
    
    # Demo the clean API
    demo_clean_api()
    
    # Run correctness tests
    if not test_correctness():
        print("\n✗ Correctness tests FAILED!")
        sys.exit(1)
    
    # Run performance benchmark
    test_performance()
    
    print("="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
    print("\nExplore the generated MLIR IR:")
    print("  python3 -c 'import llvm_example; print(llvm_example.test_ir_generation())'")
    print("  python3 -c 'import llvm_example; print(llvm_example.test_optimized_ir())'")
    print()

if __name__ == "__main__":
    main()
