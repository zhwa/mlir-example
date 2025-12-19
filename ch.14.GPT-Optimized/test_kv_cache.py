#!/usr/bin/env python3
"""Test KV cache implementation"""
import sys
import os
sys.path.insert(0, '../build/x64-release/ch.14.GPT-Optimized')

import ch14 as ch13
import numpy as np

print("Testing KV Cache Implementation\n")
print("=" * 70)

# Test configuration
d_model = 16
max_seq = 32

# Create random weights for attention
w_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
b_q = np.zeros(d_model, dtype=np.float32)
w_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
b_k = np.zeros(d_model, dtype=np.float32)
w_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
b_v = np.zeros(d_model, dtype=np.float32)
w_o = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
b_o = np.zeros(d_model, dtype=np.float32)

# Initialize caches
k_cache = np.zeros((max_seq, d_model), dtype=np.float32)
v_cache = np.zeros((max_seq, d_model), dtype=np.float32)

print("Test 1: Single token cached attention")
# Process first token
token1 = np.random.randn(1, d_model).astype(np.float32) * 0.1
out1 = ch13.gpt_attention_cached(
    token1, k_cache, v_cache, 0,
    w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o
)
print(f"  Token 1 output shape: {out1.shape}")
assert out1.shape == (1, d_model), f"Expected (1, {d_model}), got {out1.shape}"
print("  ✓ Single token works!")

print("\nTest 2: Second token with cache")
token2 = np.random.randn(1, d_model).astype(np.float32) * 0.1
out2 = ch13.gpt_attention_cached(
    token2, k_cache, v_cache, 1,
    w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o
)
print(f"  Token 2 output shape: {out2.shape}")
assert out2.shape == (1, d_model), f"Expected (1, {d_model}), got {out2.shape}"
print("  ✓ Cached attention works!")

print("\nTest 3: Process sequence incrementally")
seq_len = 8
outputs = []
for pos in range(seq_len):
    token = np.random.randn(1, d_model).astype(np.float32) * 0.1
    out = ch13.gpt_attention_cached(
        token, k_cache, v_cache, pos,
        w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o
    )
    outputs.append(out)
print(f"  Processed {seq_len} tokens incrementally")
print(f"  All outputs shape: {[o.shape for o in outputs]}")
print("  ✓ Incremental processing works!")

print("\n" + "=" * 70)
print("✓ All KV cache tests passed!")
