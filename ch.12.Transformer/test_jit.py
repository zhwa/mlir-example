#!/usr/bin/env python3
"""Test Pure MLIR JIT compilation"""

import numpy as np
import sys
import os

build_paths = [
    '../build/x64-release/ch.12.Transformer',
    'build/x64-release/ch.12.Transformer',
]

for path in build_paths:
    if os.path.exists(path):
        sys.path.insert(0, path)
        break

import ch12

print("Testing Pure MLIR JIT Compilation")
print("=" * 70)

# Test 1: Simple Add
print("\n### Test 1: Add ###")
a = ch12.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
b = ch12.Tensor(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
c = a + b  # Graph construction

result = ch12.forward(c)  # JIT compile and execute
expected = np.array([[6.0, 8.0], [10.0, 12.0]], dtype=np.float32)

print(f"Result shape: {result.shape}")
np.testing.assert_allclose(result, expected, rtol=1e-5)
print("✓ Add test PASSED!")

# Test 2: LayerNorm
print("\n### Test 2: LayerNorm ###")
x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
gamma = np.ones(4, dtype=np.float32)
beta = np.zeros(4, dtype=np.float32)

x_tensor = ch12.Tensor(x)
output = ch12.layer_norm(x_tensor, gamma, beta)
result = ch12.forward(output)

# Compute expected output with numpy
mean = np.mean(x, axis=1, keepdims=True)
var = np.var(x, axis=1, keepdims=True)
expected = (x - mean) / np.sqrt(var + 1e-5)

print(f"Result shape: {result.shape}")
np.testing.assert_allclose(result, expected, atol=1e-5)
print("✓ LayerNorm test PASSED!")

# Test 3: Linear
print("\n### Test 3: Linear ###")
x = np.array([[1.0, 2.0]], dtype=np.float32)  # (1, 2)
weight = np.array([[0.5, 0.3], [0.2, 0.4]], dtype=np.float32)  # (2, 2) -> output (1, 2)
bias = np.array([0.1, 0.2], dtype=np.float32)

x_tensor = ch12.Tensor(x)
output = ch12.linear(x_tensor, weight, bias)
result = ch12.forward(output)

# x @ weight.T + bias
expected = x @ weight.T + bias

print(f"Result shape: {result.shape}")
np.testing.assert_allclose(result, expected, rtol=1e-4)
print("✓ Linear test PASSED!")

# Test 4: GELU
print("\n### Test 4: GELU ###")
x = np.array([[0.0, 1.0, -1.0]], dtype=np.float32)
x_tensor = ch12.Tensor(x)
output = ch12.gelu(x_tensor)
result = ch12.forward(output)

# GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
def gelu_ref(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

expected = gelu_ref(x)
print(f"Result shape: {result.shape}")
np.testing.assert_allclose(result, expected, rtol=1e-4)
print("✓ GELU test PASSED!")

# Test 5: FFN (Feed-Forward Network)
print("\n### Test 5: FFN ###")
d_model = 4
d_ff = 8
x = np.random.randn(2, d_model).astype(np.float32)
w1 = np.random.randn(d_ff, d_model).astype(np.float32)
b1 = np.random.randn(d_ff).astype(np.float32)
w2 = np.random.randn(d_model, d_ff).astype(np.float32)
b2 = np.random.randn(d_model).astype(np.float32)

x_tensor = ch12.Tensor(x)
output = ch12.ffn(x_tensor, w1, b1, w2, b2)
result = ch12.forward(output)

# Reference: FFN(x) = Linear(GELU(Linear(x)))
hidden = x @ w1.T + b1
activated = gelu_ref(hidden)
expected = activated @ w2.T + b2

print(f"Result shape: {result.shape}")
np.testing.assert_allclose(result, expected, rtol=1e-4)
print("✓ FFN test PASSED!")

# Test 6: Matmul
print("\n### Test 6: Matmul ###")
a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

a_tensor = ch12.Tensor(a)
b_tensor = ch12.Tensor(b)
output = ch12.matmul(a_tensor, b_tensor)
result = ch12.forward(output)

expected = a @ b
print(f"Result shape: {result.shape}")
np.testing.assert_allclose(result, expected, rtol=1e-5)
print("✓ Matmul test PASSED!")

# Test 7: Transpose
print("\n### Test 7: Transpose ###")
a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
a_tensor = ch12.Tensor(a)
output = ch12.transpose(a_tensor)
result = ch12.forward(output)

expected = a.T
print(f"Result shape: {result.shape}")
np.testing.assert_allclose(result, expected, rtol=1e-5)
print("✓ Transpose test PASSED!")

# Test 8: Softmax
print("\n### Test 8: Softmax ###")
a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
a_tensor = ch12.Tensor(a)
output = ch12.softmax(a_tensor)
result = ch12.forward(output)

# Softmax along last axis
exp_a = np.exp(a - np.max(a, axis=-1, keepdims=True))
expected = exp_a / np.sum(exp_a, axis=-1, keepdims=True)
print(f"Result shape: {result.shape}")
np.testing.assert_allclose(result, expected, rtol=1e-5)
print("✓ Softmax test PASSED!")

# Test 9: Scale
print("\n### Test 9: Scale ###")
a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
scale_factor = 0.5
a_tensor = ch12.Tensor(a)
output = ch12.scale(a_tensor, scale_factor)
result = ch12.forward(output)

expected = a * scale_factor
print(f"Result shape: {result.shape}")
np.testing.assert_allclose(result, expected, rtol=1e-5)
print("✓ Scale test PASSED!")

# Test 10: Attention
print("\n### Test 10: Multi-Head Attention ###")
seq_len, d_model = 4, 8
x = np.random.randn(seq_len, d_model).astype(np.float32)

# Q, K, V projection weights
w_q = np.random.randn(d_model, d_model).astype(np.float32)
b_q = np.random.randn(d_model).astype(np.float32)
w_k = np.random.randn(d_model, d_model).astype(np.float32)
b_k = np.random.randn(d_model).astype(np.float32)
w_v = np.random.randn(d_model, d_model).astype(np.float32)
b_v = np.random.randn(d_model).astype(np.float32)
w_o = np.random.randn(d_model, d_model).astype(np.float32)
b_o = np.random.randn(d_model).astype(np.float32)

x_tensor = ch12.Tensor(x)
output = ch12.attention(x_tensor, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o)
result = ch12.forward(output)

# Compute expected attention
Q = x @ w_q.T + b_q
K = x @ w_k.T + b_k
V = x @ w_v.T + b_v
d_k = K.shape[1]
scores = Q @ K.T / np.sqrt(d_k)
attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
attn_out = attn_weights @ V
expected = attn_out @ w_o.T + b_o

print(f"Result shape: {result.shape}")
print(f"Result sample: {result[0, :3]}")
print(f"Expected sample: {expected[0, :3]}")
np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)
print("✓ Multi-Head Attention test PASSED!")

# Test 11: Transformer Block
print("\n### Test 11: Transformer Block ###")
seq_len, d_model = 4, 8
x = np.random.randn(seq_len, d_model).astype(np.float32)

# Attention weights
w_q = np.random.randn(d_model, d_model).astype(np.float32)
b_q = np.random.randn(d_model).astype(np.float32)
w_k = np.random.randn(d_model, d_model).astype(np.float32)
b_k = np.random.randn(d_model).astype(np.float32)
w_v = np.random.randn(d_model, d_model).astype(np.float32)
b_v = np.random.randn(d_model).astype(np.float32)
w_o = np.random.randn(d_model, d_model).astype(np.float32)
b_o = np.random.randn(d_model).astype(np.float32)

# LayerNorm 1
gamma1 = np.ones(d_model, dtype=np.float32)
beta1 = np.zeros(d_model, dtype=np.float32)

# FFN weights
d_ff = 16
w1 = np.random.randn(d_ff, d_model).astype(np.float32)
b1 = np.random.randn(d_ff).astype(np.float32)
w2 = np.random.randn(d_model, d_ff).astype(np.float32)
b2 = np.random.randn(d_model).astype(np.float32)

# LayerNorm 2
gamma2 = np.ones(d_model, dtype=np.float32)
beta2 = np.zeros(d_model, dtype=np.float32)

x_tensor = ch12.Tensor(x)
output = ch12.transformer_block(
    x_tensor,
    w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
    gamma1, beta1,
    w1, b1, w2, b2,
    gamma2, beta2
)
result = ch12.forward(output)

# Compute expected output step-by-step
# Step 1: LayerNorm(input)
mean1 = np.mean(x, axis=1, keepdims=True)
var1 = np.var(x, axis=1, keepdims=True)
normed1 = (x - mean1) / np.sqrt(var1 + 1e-5) * gamma1 + beta1

# Step 2: Multi-head attention
Q = normed1 @ w_q.T + b_q
K = normed1 @ w_k.T + b_k
V = normed1 @ w_v.T + b_v
d_k = K.shape[1]
scores = Q @ K.T / np.sqrt(d_k)
attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
attn_out = attn_weights @ V
attn_out = attn_out @ w_o.T + b_o

# Step 3: Residual connection
residual1 = x + attn_out

# Step 4: LayerNorm(residual1)
mean2 = np.mean(residual1, axis=1, keepdims=True)
var2 = np.var(residual1, axis=1, keepdims=True)
normed2 = (residual1 - mean2) / np.sqrt(var2 + 1e-5) * gamma2 + beta2

# Step 5: FFN
hidden = normed2 @ w1.T + b1
activated = gelu_ref(hidden)
ffn_out = activated @ w2.T + b2

# Step 6: Final residual
expected = residual1 + ffn_out

print(f"Result shape: {result.shape}")
np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)
print("✓ Transformer Block test PASSED!")

print("\n" + "=" * 70)
print("SUCCESS: All Pure MLIR JIT tests passed!")
print("=" * 70)