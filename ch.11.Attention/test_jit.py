#!/usr/bin/env python3
"""
Chapter 11: Attention Mechanism - Simple Tests

Tests the C++ attention implementation directly with NumPy arrays.
"""

import sys
import os
import numpy as np

# Add build directory to path
build_paths = [
    '../build/x64-release/ch.11.Attention',
    '../build/x64-debug/ch.11.Attention',
    'build/x64-release/ch.11.Attention',
    'build/x64-debug/ch.11.Attention'
]

build_dir = None
for path in build_paths:
    if os.path.exists(path):
        build_dir = path
        break

if build_dir:
    print(f"Using build directory: {build_dir}")
    sys.path.insert(0, build_dir)
else:
    print("Warning: Build directory not found, attempting to import anyway")

try:
    import ch11
except ImportError as e:
    print(f"Error: Could not import ch11 module: {e}")
    print("Please build Chapter 11 first:")
    print("  cmake --build build/x64-release --target ch11")
    sys.exit(1)

print()
print("=" * 70)
print("Chapter 11: Attention Mechanism Tests")
print("=" * 70)
print()

def reference_attention_with_proj(x, w_q, w_k, w_v, w_o, num_heads):
    """
    Reference implementation of multi-head attention with projections
    """
    B, T, C = x.shape
    head_dim = C // num_heads
    
    # Linear projections: Q = x @ w_q^T, etc.
    q = np.matmul(x, w_q.T)  # (B, T, C)
    k = np.matmul(x, w_k.T)
    v = np.matmul(x, w_v.T)
    
    # Reshape and transpose for multi-head: (B, T, C) -> (B, num_heads, T, head_dim)
    q = q.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    
    # Compute attention for each head
    outputs = []
    for h in range(num_heads):
        q_h = q[:, h, :, :]  # (B, T, head_dim)
        k_h = k[:, h, :, :]
        v_h = v[:, h, :, :]
        
        # Attention scores
        scores = np.matmul(q_h, k_h.transpose(0, 2, 1))  # (B, T, T)
        scores = scores / np.sqrt(head_dim)
        
        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        attn = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        
        # Output
        out = np.matmul(attn, v_h)  # (B, T, head_dim)
        outputs.append(out)
    
    # Stack and concatenate heads
    stacked = np.stack(outputs, axis=1)  # (B, num_heads, T, head_dim)
    transposed = stacked.transpose(0, 2, 1, 3)  # (B, T, num_heads, head_dim)
    concat = transposed.reshape(B, T, C)  # (B, T, C)
    
    # Output projection
    final = np.matmul(concat, w_o.T)  # (B, T, C)
    
    return final


# Test 1: Single-Head Attention
print("### Test 1: Single-Head Self-Attention ###")
seq_len = 4
d_model = 8
num_heads = 1
head_dim = d_model

np.random.seed(42)
x = np.random.randn(seq_len, d_model).astype(np.float32)

# Identity weights (no projection)
w_q = np.eye(d_model, dtype=np.float32)
w_k = np.eye(d_model, dtype=np.float32)
w_v = np.eye(d_model, dtype=np.float32)
w_o = np.eye(d_model, dtype=np.float32)

print(f"Input shape: {x.shape}")
print(f"Configuration: num_heads={num_heads}, head_dim={head_dim}")

# Compute reference (add batch dimension for reference function)
x_batched = x[np.newaxis, :, :]
expected_batched = reference_attention_with_proj(x_batched, w_q, w_k, w_v, w_o, num_heads)
expected = expected_batched[0]  # Remove batch dimension

# Compute with ch11
output = np.zeros_like(x)
ch11.attention(x, output, w_q, w_k, w_v, w_o, num_heads, head_dim)

print(f"\nExpected output (first 2 tokens, 4 dims):\n{expected[:2, :4]}")
print(f"Actual output (first 2 tokens, 4 dims):\n{output[:2, :4]}")

try:
    np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-4)
    print("\n✓ Test 1 PASSED: Single-head attention matches reference!")
except AssertionError as e:
    print(f"\n✗ Test 1 FAILED: {e}")

print()


# Test 2: Multi-Head Attention
print("### Test 2: Multi-Head Attention (2 heads) ###")
num_heads = 2
head_dim = d_model // num_heads

np.random.seed(123)
x_multi = np.random.randn(seq_len, d_model).astype(np.float32)

print(f"Input shape: {x_multi.shape}")
print(f"Configuration: num_heads={num_heads}, head_dim={head_dim}")

# Compute reference (add batch dimension for reference function)
x_multi_batched = x_multi[np.newaxis, :, :]
expected_multi_batched = reference_attention_with_proj(x_multi_batched, w_q, w_k, w_v, w_o, num_heads)
expected_multi = expected_multi_batched[0]  # Remove batch dimension

# Compute with ch11
output_multi = np.zeros_like(x_multi)
ch11.attention(x_multi, output_multi, w_q, w_k, w_v, w_o, num_heads, head_dim)

print(f"\nExpected output (first 2 tokens, 4 dims):\n{expected_multi[:2, :4]}")
print(f"Actual output (first 2 tokens, 4 dims):\n{output_multi[:2, :4]}")

try:
    np.testing.assert_allclose(output_multi, expected_multi, rtol=1e-4, atol=1e-4)
    print("\n✓ Test 2 PASSED: Multi-head attention matches reference!")
except AssertionError as e:
    print(f"\n✗ Test 2 FAILED: {e}")

print()


# Test 3: Attention with Random Projections
print("### Test 3: Attention with Q/K/V/O Projections ###")
np.random.seed(456)
x_proj = np.random.randn(seq_len, d_model).astype(np.float32)

# Random weight matrices
# Note: C++ implementation expects weights in the form they'll be used (no transpose)
# Reference implementation does x @ w.T, so we need to transpose the weights before passing to C++
np.random.seed(789)
w_q_rand = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
w_k_rand = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
w_v_rand = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
w_o_rand = np.random.randn(d_model, d_model).astype(np.float32) * 0.1

print(f"Input shape: {x_proj.shape}")
print(f"Configuration: num_heads={num_heads}, head_dim={head_dim}")
print(f"Weight shapes: w_q={w_q_rand.shape}, w_k={w_k_rand.shape}, w_v={w_v_rand.shape}, w_o={w_o_rand.shape}")

# Compute reference (add batch dimension for reference function)
x_proj_batched = x_proj[np.newaxis, :, :]
expected_proj_batched = reference_attention_with_proj(x_proj_batched, w_q_rand, w_k_rand, w_v_rand, w_o_rand, num_heads)
expected_proj = expected_proj_batched[0]  # Remove batch dimension

# Compute with ch11 (transpose weights to match C++ expectation: input @ W instead of input @ W.T)
output_proj = np.zeros_like(x_proj)
ch11.attention(x_proj, output_proj, w_q_rand.T, w_k_rand.T, w_v_rand.T, w_o_rand.T, num_heads, head_dim)

print(f"\nExpected output (first 2 tokens, 4 dims):\n{expected_proj[:2, :4]}")
print(f"Actual output (first 2 tokens, 4 dims):\n{output_proj[:2, :4]}")

try:
    np.testing.assert_allclose(output_proj, expected_proj, rtol=1e-4, atol=1e-4)
    print("\n✓ Test 3 PASSED: Attention with projections matches reference!")
except AssertionError as e:
    print(f"\n✗ Test 3 FAILED: {e}")

print()
print("=" * 70)
print("Test suite complete")
print("=" * 70)
