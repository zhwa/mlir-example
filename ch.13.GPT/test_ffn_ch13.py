#!/usr/bin/env python3
"""Test FFN in Chapter 13"""
import sys
sys.path.insert(0, '../build/x64-release/ch.13.GPT')
import ch13
import numpy as np

d_model = 16
d_ff = 64
batch = 8

x = np.random.randn(batch, d_model).astype(np.float32) * 0.1
w1 = np.random.randn(d_model, d_ff).astype(np.float32) * 0.02
b1 = np.zeros(d_ff, dtype=np.float32)
w2 = np.random.randn(d_ff, d_model).astype(np.float32) * 0.02
b2 = np.zeros(d_model, dtype=np.float32)

print('Testing Chapter 13 FFN...')
print(f'Input: {x.shape}')
print(f'w1: {w1.shape}, b1: {b1.shape}')
print(f'w2: {w2.shape}, b2: {b2.shape}')

hidden = ch13.forward(ch13.linear(ch13.Tensor(x), w1, b1))
print(f'After linear1: {hidden.shape}')

activated = ch13.forward(ch13.gelu(ch13.Tensor(hidden)))
print(f'After GELU: {activated.shape}')

output = ch13.forward(ch13.linear(ch13.Tensor(activated), w2, b2))
print(f'After linear2: {output.shape}')
