#!/usr/bin/env python3
"""
Chapter 14: Performance Baseline Measurements

Run this before optimization to establish baseline performance.
After each optimization phase, re-run to measure speedup.
"""

import sys
sys.path.insert(0, '../build/x64-release/ch.14.GPT-Optimized')

import numpy as np
import time
import ch14

def benchmark_operation(op_func, *args, num_runs=100, warmup=10):
    """Measure average execution time with warmup"""
    # Warmup runs
    for _ in range(warmup):
        result = op_func(*args)
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(num_runs):
        result = op_func(*args)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / num_runs * 1000
    return avg_time_ms, result

print("=" * 70)
print("Chapter 14: Phase 0 - Baseline Performance Measurements")
print("=" * 70)
print()
print("Measuring current performance (Chapter 13 code, no optimizations)")
print()

# ============================================================================
# Benchmark 1: MatMul (core operation)
# ============================================================================

print("### Benchmark 1: Matrix Multiplication ###")
sizes = [(64, 128, 64), (128, 256, 128), (256, 512, 256)]

for M, K, N in sizes:
    # Create test matrices
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    # Benchmark
    time_ms, result = benchmark_operation(
        lambda: ch14.forward(ch14.matmul(ch14.Tensor(A), ch14.Tensor(B))),
        num_runs=50
    )
    
    # GFLOPS calculation: 2*M*N*K operations
    gflops = (2 * M * N * K) / (time_ms * 1e6)
    
    print(f"  {M}x{K} @ {K}x{N}: {time_ms:.3f} ms ({gflops:.2f} GFLOPS)")

print()

# ============================================================================
# Benchmark 2: LayerNorm
# ============================================================================

print("### Benchmark 2: Layer Normalization ###")
configs = [(128, 64), (256, 128), (512, 256)]

for batch, features in configs:
    # Create test data
    x = np.random.randn(batch, features).astype(np.float32)
    gamma = np.ones(features, dtype=np.float32)
    beta = np.zeros(features, dtype=np.float32)
    
    # Benchmark
    time_ms, result = benchmark_operation(
        lambda: ch14.forward(ch14.layer_norm(ch14.Tensor(x), gamma, beta)),
        num_runs=100
    )
    
    print(f"  ({batch}, {features}): {time_ms:.3f} ms")

print()

# ============================================================================
# Benchmark 3: Attention (without generation)
# ============================================================================

print("### Benchmark 3: GPT Attention Block ###")
seq_lens = [8, 16, 32]
d_model = 64

for seq_len in seq_lens:
    # Create test data
    hidden = np.random.randn(seq_len, d_model).astype(np.float32)
    
    # Create projection weights
    w_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    b_q = np.zeros(d_model, dtype=np.float32)
    w_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    b_k = np.zeros(d_model, dtype=np.float32)
    w_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    b_v = np.zeros(d_model, dtype=np.float32)
    w_o = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    b_o = np.zeros(d_model, dtype=np.float32)
    
    # Benchmark
    time_ms, result = benchmark_operation(
        lambda: ch14.forward(ch14.gpt_attention(
            ch14.Tensor(hidden), w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o
        )),
        num_runs=50
    )
    
    print(f"  seq_len={seq_len}: {time_ms:.3f} ms")

print()

# ============================================================================
# Benchmark 4: Full GPT Forward Pass
# ============================================================================

print("### Benchmark 4: Full GPT Forward Pass ###")

def create_gpt_weights(d_model, d_ff, num_layers):
    """Create random weights for GPT model"""
    weights = []
    for _ in range(num_layers):
        # Q, K, V, O projections
        for _ in range(4):
            weights.append(np.random.randn(d_model, d_model).astype(np.float32) * 0.02)
            weights.append(np.zeros(d_model, dtype=np.float32))
        
        # FFN
        weights.append(np.random.randn(d_model, d_ff).astype(np.float32) * 0.02)
        weights.append(np.zeros(d_ff, dtype=np.float32))
        weights.append(np.random.randn(d_ff, d_model).astype(np.float32) * 0.02)
        weights.append(np.zeros(d_model, dtype=np.float32))
        
        # Layer norms
        weights.append(np.ones(d_model, dtype=np.float32))
        weights.append(np.zeros(d_model, dtype=np.float32))
        weights.append(np.ones(d_model, dtype=np.float32))
        weights.append(np.zeros(d_model, dtype=np.float32))
    
    return weights

vocab_size = 256
d_model = 64
d_ff = 256
num_layers = 2
seq_lens = [8, 16, 32]

embedding_table = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02
all_weights = create_gpt_weights(d_model, d_ff, num_layers)
final_gamma = np.ones(d_model, dtype=np.float32)
final_beta = np.zeros(d_model, dtype=np.float32)

for seq_len in seq_lens:
    indices = np.random.randint(0, vocab_size, size=seq_len, dtype=np.int32)
    
    # Benchmark
    time_ms, result = benchmark_operation(
        lambda: ch14.forward(ch14.gpt_forward(
            indices, embedding_table, all_weights, final_gamma, final_beta
        )),
        num_runs=50
    )
    
    print(f"  seq_len={seq_len}, 2 layers: {time_ms:.3f} ms")

print()

# ============================================================================
# Benchmark 5: Generation Loop (autoregressive)
# ============================================================================

print("### Benchmark 5: Autoregressive Generation ###")

from generation import generate_greedy

prompt = np.array([72, 101, 108, 108, 111], dtype=np.int32)  # "Hello"
max_new_tokens_list = [10, 20, 50]

for max_new_tokens in max_new_tokens_list:
    # Benchmark (fewer runs since generation is expensive)
    time_ms, result = benchmark_operation(
        lambda: generate_greedy(
            prompt, embedding_table, all_weights, final_gamma, final_beta,
            max_new_tokens=max_new_tokens
        ),
        num_runs=10,
        warmup=3
    )
    
    tokens_per_sec = max_new_tokens / (time_ms / 1000)
    
    print(f"  Generate {max_new_tokens} tokens: {time_ms:.3f} ms ({tokens_per_sec:.1f} tokens/sec)")

print()
print("=" * 70)
print("Baseline measurements complete!")
print("=" * 70)
print()
print("Next steps:")
print("  1. Save these numbers for comparison")
print("  2. Implement Phase 1: Linalg-based MatMul lowering")
print("  3. Re-run this script to measure speedup")
print()
print("Expected improvements after all optimizations:")
print("  - MatMul: 2-3x faster")
print("  - LayerNorm: 1.5-2x faster")
print("  - Full forward pass: 3-5x faster")
print("  - Generation (with KV cache): 10-100x faster")
