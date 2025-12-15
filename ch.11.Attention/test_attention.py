#!/usr/bin/env python3
"""
Chapter 11: Attention Mechanism - Test Suite

Start simple: implement single-head attention first
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
print("Chapter 11: Attention Mechanism")
print("=" * 70)
print()

# Test 1: Simple scaled dot-product attention (single head, small size)
print("### Test 1: Single-Head Attention (Minimal Case) ###")
print("Input shape: (batch=1, seq_len=4, d_model=8)")
print()

# Create simple test data
batch_size = 1
seq_len = 4
d_model = 8

np.random.seed(42)
x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

# PyTorch reference implementation
def reference_attention(x, scale=True):
    """
    Simple single-head attention for reference
    q = k = v = x (self-attention)
    """
    import numpy as np
    
    q = x  # (B, T, D)
    k = x
    v = x
    
    # Attention scores: Q @ K^T
    scores = np.matmul(q, k.transpose(0, 2, 1))  # (B, T, T)
    
    if scale:
        d_k = x.shape[-1]
        scores = scores / np.sqrt(d_k)
    
    # Softmax
    # For numerical stability, subtract max
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
    
    # Weighted sum: attn @ V
    output = np.matmul(attention_weights, v)  # (B, T, D)
    
    return output, attention_weights

expected_output, expected_attn = reference_attention(x)

print(f"Input:\n{x[0, :2, :4]}  (showing first 2 tokens, 4 dims)")
print(f"\nExpected output shape: {expected_output.shape}")
print(f"Expected attention weights shape: {expected_attn.shape}")
print(f"\nExpected output (first 2 tokens, 4 dims):\n{expected_output[0, :2, :4]}")

# Try to run with ch11 (will fail initially until we implement it)
try:
    x_tensor = ch11.Tensor(x)
    output_tensor = ch11.attention(x_tensor, num_heads=1, head_dim=d_model)
    result = ch11.forward(output_tensor)
    
    print(f"\nActual output shape: {result.shape}")
    print(f"Actual output (first 2 tokens, 4 dims):\n{result[0, :2, :4]}")
    
    # Check if results match
    np.testing.assert_allclose(result, expected_output, rtol=1e-4, atol=1e-4)
    print("\n✓ Test 1 PASSED: Single-head attention matches reference!")
    
except AttributeError as e:
    print(f"\n⚠ Test 1 SKIPPED: {e}")
    print("   (This is expected - we haven't implemented attention() yet)")
except Exception as e:
    print(f"\n✗ Test 1 FAILED: {e}")

print()

# Test 2: Multi-Head Attention
print("### Test 2: Multi-Head Attention ###")
num_heads = 2
d_model = 8
head_dim = d_model // num_heads  # 8 // 2 = 4

batch_size = 1
seq_len = 4

np.random.seed(123)
x_multi = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

print(f"Input shape: (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")
print(f"Configuration: num_heads={num_heads}, head_dim={head_dim}")

def reference_multihead_attention(x, num_heads):
    """
    Multi-head attention reference implementation
    For simplicity: Q=K=V=input (no projections yet)
    """
    B, T, C = x.shape
    head_dim = C // num_heads
    
    # Reshape: (B, T, C) -> (B, T, num_heads, head_dim)
    x_reshaped = x.reshape(B, T, num_heads, head_dim)
    
    # Transpose: (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
    x_heads = x_reshaped.transpose(0, 2, 1, 3)
    
    # Compute attention for each head independently
    outputs = []
    for h in range(num_heads):
        q = x_heads[:, h, :, :]  # (B, T, head_dim)
        k = x_heads[:, h, :, :]
        v = x_heads[:, h, :, :]
        
        # Attention scores
        scores = np.matmul(q, k.transpose(0, 2, 1))  # (B, T, T)
        scores = scores / np.sqrt(head_dim)
        
        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        attn = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        
        # Output
        out = np.matmul(attn, v)  # (B, T, head_dim)
        outputs.append(out)
    
    # Stack heads: list of (B, T, head_dim) -> (B, num_heads, T, head_dim)
    stacked = np.stack(outputs, axis=1)
    
    # Transpose back: (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim)
    transposed = stacked.transpose(0, 2, 1, 3)
    
    # Concatenate heads: (B, T, num_heads, head_dim) -> (B, T, C)
    concat = transposed.reshape(B, T, C)
    
    return concat

expected_multi = reference_multihead_attention(x_multi, num_heads)

print(f"\nExpected output shape: {expected_multi.shape}")
print(f"Expected output (first 2 tokens, 4 dims):\n{expected_multi[0, :2, :4]}")

try:
    x_tensor_multi = ch11.Tensor(x_multi)
    output_tensor_multi = ch11.attention(x_tensor_multi, num_heads=num_heads, head_dim=head_dim)
    result_multi = ch11.forward(output_tensor_multi)
    
    print(f"\nActual output shape: {result_multi.shape}")
    print(f"Actual output (first 2 tokens, 4 dims):\n{result_multi[0, :2, :4]}")
    
    np.testing.assert_allclose(result_multi, expected_multi, rtol=1e-4, atol=1e-4)
    print("\n✓ Test 2 PASSED: Multi-head attention matches reference!")
    
except NotImplementedError as e:
    print(f"\n⚠ Test 2 SKIPPED: {e}")
except Exception as e:
    print(f"\n✗ Test 2 FAILED: {e}")

print()

# Test 3: Attention with Linear Projections
print("### Test 3: Attention with Linear Projections (Q, K, V, Output) ###")
num_heads = 2
d_model = 8
head_dim = d_model // num_heads

batch_size = 1
seq_len = 4

np.random.seed(456)
x_proj = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

# Create random weight matrices (d_model x d_model)
np.random.seed(789)
w_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
w_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
w_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
w_o = np.random.randn(d_model, d_model).astype(np.float32) * 0.1

print(f"Input shape: (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")
print(f"Configuration: num_heads={num_heads}, head_dim={head_dim}")
print(f"Weight shapes: w_q={w_q.shape}, w_k={w_k.shape}, w_v={w_v.shape}, w_o={w_o.shape}")

def reference_attention_with_proj(x, w_q, w_k, w_v, w_o, num_heads):
    """
    Full attention with linear projections
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

expected_proj = reference_attention_with_proj(x_proj, w_q, w_k, w_v, w_o, num_heads)

print(f"\nExpected output shape: {expected_proj.shape}")
print(f"Expected output (first 2 tokens, 4 dims):\n{expected_proj[0, :2, :4]}")

try:
    x_tensor_proj = ch11.Tensor(x_proj)
    output_tensor_proj = ch11.attention_proj(x_tensor_proj, num_heads, head_dim, w_q, w_k, w_v, w_o)
    result_proj = ch11.forward(output_tensor_proj)
    
    print(f"\nActual output shape: {result_proj.shape}")
    print(f"Actual output (first 2 tokens, 4 dims):\n{result_proj[0, :2, :4]}")
    
    np.testing.assert_allclose(result_proj, expected_proj, rtol=1e-4, atol=1e-4)
    print("\n✓ Test 3 PASSED: Attention with projections matches reference!")
    
except AttributeError as e:
    print(f"\n⚠ Test 3 SKIPPED: {e}")
except Exception as e:
    print(f"\n✗ Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Test suite complete")
print("=" * 70)
