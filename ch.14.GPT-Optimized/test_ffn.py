#!/usr/bin/env python3
"""Test FFN in isolation"""
import numpy as np
import ch14 as ch13

print("Testing FFN with residual connection...")

d_model = 16
d_ff = 64
batch = 8

# Create input
x = np.random.randn(batch, d_model).astype(np.float32) * 0.1

# Create FFN weights
w1 = np.random.randn(d_model, d_ff).astype(np.float32) * 0.02
b1 = np.zeros(d_ff, dtype=np.float32)
w2 = np.random.randn(d_ff, d_model).astype(np.float32) * 0.02
b2 = np.zeros(d_model, dtype=np.float32)

print(f"Input shape: {x.shape}")
print(f"w1 shape: {w1.shape}, b1 shape: {b1.shape}")
print(f"w2 shape: {w2.shape}, b2 shape: {b2.shape}")

# Test FFN without residual
print("\n1. Testing FFN alone...")
try:
    hidden = ch13.forward(ch13.linear(ch13.Tensor(x), w1, b1))
    print(f"   After linear1: {hidden.shape}")
    
    activated = ch13.forward(ch13.gelu(ch13.Tensor(hidden)))
    print(f"   After GELU: {activated.shape}")
    
    output = ch13.forward(ch13.linear(ch13.Tensor(activated), w2, b2))
    print(f"   After linear2: {output.shape}")
    print("   ✓ FFN works!")
except Exception as e:
    print(f"   ✗ FFN failed: {e}")

# Test Add operation
print("\n2. Testing Add operation...")
try:
    a = np.random.randn(batch, d_model).astype(np.float32)
    b = np.random.randn(batch, d_model).astype(np.float32)
    result = ch13.forward(ch13.Tensor(a) + ch13.Tensor(b))
    print(f"   Add result shape: {result.shape}")
    print("   ✓ Add works!")
except Exception as e:
    print(f"   ✗ Add failed: {e}")

# Test FFN with residual (like in gpt_block)
print("\n3. Testing FFN with residual connection...")
try:
    # Simulate what gpt_block does
    input_tensor = ch13.Tensor(x)
    ffn_output = ch13.linear(ch13.gelu(ch13.linear(input_tensor, w1, b1)), w2, b2)
    residual = input_tensor + ffn_output
    result = ch13.forward(residual)
    print(f"   Result shape: {result.shape}")
    print("   ✓ FFN with residual works!")
except Exception as e:
    print(f"   ✗ FFN with residual failed: {e}")
