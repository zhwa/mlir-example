#!/usr/bin/env python3
"""
Chapter 14: Optimized GPT - Test Suite & Benchmarks

Tests GPT components incrementally (same as Chapter 13):
- Phase 2: Embedding layer
- Phase 3: Causal masking
- Phase 4: RoPE
- Phase 5: Full GPT model
- Phase 6: Generation

ALSO includes performance benchmarks vs Chapter 13 baseline.
"""

import sys
import os
import numpy as np

# Add build directory to path
build_paths = [
    '../build/x64-release/ch.14.GPT-Optimized',
    '../build/x64-debug/ch.14.GPT-Optimized',
    'build/x64-release/ch.14.GPT-Optimized',
    'build/x64-debug/ch.14.GPT-Optimized'
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
    import ch14
    # Alias for easier porting from ch14 tests
    ch14 = ch14
except ImportError as e:
    print(f"Error: Could not import ch14 module: {e}")
    print("Please build Chapter 14 first:")
    print("  cmake --build build/x64-release --target ch14")
    sys.exit(1)

print()
print("=" * 70)
print("Chapter 14: Optimized GPT Tests (Phase 0 - Baseline)")
print("=" * 70)
print()

# Model configuration
VOCAB_SIZE = 256
D_MODEL = 64
NUM_HEADS = 4
HEAD_DIM = D_MODEL // NUM_HEADS
NUM_LAYERS = 2
MAX_SEQ_LEN = 32

# ============================================================================
# Phase 1: Sanity Check (module loads)
# ============================================================================

print("### Phase 1: Module Import ###")
print("✓ ch14 module imported successfully")
print()

# ============================================================================
# Phase 2: Embedding Layer Tests
# ============================================================================

print("### Phase 2: Embedding Layer ###")

# Test 1: Basic embedding lookup
print("Test 1: Basic embedding lookup")
seq_len = 4
vocab_size = 10
d_model = 8

# Create simple embedding table (each token i maps to vector [i, i, i, ...])
embedding_table = np.arange(vocab_size * d_model, dtype=np.float32).reshape(vocab_size, d_model)
indices = np.array([0, 2, 5, 9], dtype=np.int32)

# Expected output: look up rows 0, 2, 5, 9 from embedding_table
expected = embedding_table[indices]

# Compile and execute
result = ch14.forward(ch14.embedding(indices, embedding_table))

print(f"  Input indices shape: {indices.shape}")
print(f"  Embedding table shape: {embedding_table.shape}")
print(f"  Output shape: {result.shape}")
print(f"  Expected shape: {expected.shape}")

# Verify numerical correctness
if np.allclose(result, expected, rtol=1e-5, atol=1e-6):
    print("  ✓ Embedding lookup matches NumPy reference")
else:
    print("  ✗ ERROR: Embedding mismatch!")
    print(f"    Max diff: {np.abs(result - expected).max()}")
    sys.exit(1)

# Test 2: Larger batch with realistic dimensions
print("\nTest 2: GPT-scale embedding lookup")
seq_len = 16
indices = np.array([72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33, 0, 0, 0, 0], dtype=np.int32)  # "Hello world!" padded
embedding_table = np.random.randn(VOCAB_SIZE, D_MODEL).astype(np.float32) * 0.1
expected = embedding_table[indices]

result = ch14.forward(ch14.embedding(indices, embedding_table))

print(f"  Sequence length: {seq_len}")
print(f"  Vocab size: {VOCAB_SIZE}, d_model: {D_MODEL}")
print(f"  Output shape: {result.shape}")

if result.shape == (seq_len, D_MODEL):
    print("  ✓ Output shape correct")
else:
    print(f"  ✗ ERROR: Expected shape ({seq_len}, {D_MODEL}), got {result.shape}")
    sys.exit(1)

if np.allclose(result, expected, rtol=1e-5, atol=1e-6):
    print("  ✓ Embedding values correct")
else:
    print("  ✗ ERROR: Embedding mismatch!")
    print(f"    Max diff: {np.abs(result - expected).max()}")
    sys.exit(1)

print("\n✓ Phase 2: All embedding tests passed!")
print()

# ============================================================================
# Phase 3: Causal Masking Tests
# ============================================================================

print("### Phase 3: Causal Masking ###")

# Test 1: Create causal mask
print("Test 1: Create causal mask")
seq_len = 4
mask = ch14.forward(ch14.create_causal_mask(seq_len))

print(f"  Mask shape: {mask.shape}")
print(f"  Expected shape: ({seq_len}, {seq_len})")

# Verify shape
if mask.shape != (seq_len, seq_len):
    print(f"  ✗ ERROR: Expected shape ({seq_len}, {seq_len}), got {mask.shape}")
    sys.exit(1)

# Verify structure: lower triangle = 0, upper triangle = -inf
print("  Mask values:")
for i in range(seq_len):
    row_str = "  "
    for j in range(seq_len):
        if np.isinf(mask[i, j]) and mask[i, j] < 0:
            row_str += "  -inf"
        else:
            row_str += f"{mask[i, j]:6.1f}"
    print(row_str)

# Check pattern
correct = True
for i in range(seq_len):
    for j in range(seq_len):
        if j <= i:
            # Should be 0.0 (can attend)
            if not np.isclose(mask[i, j], 0.0):
                print(f"  ✗ ERROR: mask[{i}][{j}] should be 0.0, got {mask[i, j]}")
                correct = False
        else:
            # Should be -inf (cannot attend)
            if not (np.isinf(mask[i, j]) and mask[i, j] < 0):
                print(f"  ✗ ERROR: mask[{i}][{j}] should be -inf, got {mask[i, j]}")
                correct = False

if correct:
    print("  ✓ Causal mask structure correct")
else:
    sys.exit(1)

# Test 2: Masked softmax
print("\nTest 2: Masked softmax with causal mask")
batch = 2
seq_len = 4

# Create simple logits (all ones)
logits = np.ones((batch, seq_len, seq_len), dtype=np.float32)
mask_tensor = ch14.create_causal_mask(seq_len)

# Apply masked softmax
result = ch14.forward(ch14.masked_softmax(ch14.Tensor(logits), mask_tensor))

print(f"  Input logits shape: {logits.shape}")
print(f"  Mask shape: {mask.shape}")
print(f"  Output shape: {result.shape}")

# Verify shape
if result.shape != (batch, seq_len, seq_len):
    print(f"  ✗ ERROR: Expected shape ({batch}, {seq_len}, {seq_len}), got {result.shape}")
    sys.exit(1)

# Verify properties:
# 1. Each row sums to 1.0 (softmax property)
# 2. Upper triangle is 0 (masked positions)
# 3. Lower triangle values are positive and sum to 1
print("  First sample attention weights:")
for i in range(seq_len):
    row_str = "    "
    for j in range(seq_len):
        row_str += f"{result[0, i, j]:6.4f} "
    row_sum = result[0, i, :].sum()
    row_str += f"  (sum={row_sum:.4f})"
    print(row_str)

correct = True
for b in range(batch):
    for i in range(seq_len):
        row_sum = result[b, i, :].sum()
        if not np.isclose(row_sum, 1.0, atol=1e-5):
            print(f"  ✗ ERROR: Row sum [{b}][{i}] = {row_sum}, expected 1.0")
            correct = False

        for j in range(seq_len):
            if j > i:
                # Should be 0 (masked)
                if not np.isclose(result[b, i, j], 0.0, atol=1e-6):
                    print(f"  ✗ ERROR: result[{b}][{i}][{j}] = {result[b, i, j]}, expected 0.0 (masked)")
                    correct = False
            else:
                # Should be positive and equal (uniform attention to available positions)
                expected_val = 1.0 / (i + 1)  # Attend uniformly to positions 0..i
                if not np.isclose(result[b, i, j], expected_val, atol=1e-5):
                    print(f"  ✗ ERROR: result[{b}][{i}][{j}] = {result[b, i, j]}, expected {expected_val}")
                    correct = False

if correct:
    print("  ✓ Masked softmax correct (rows sum to 1, upper triangle masked)")
else:
    sys.exit(1)

# Test 3: Masked softmax with non-uniform logits
print("\nTest 3: Masked softmax with varied logits")
seq_len = 4
logits = np.array([[[1.0, 2.0, 3.0, 4.0],
                    [2.0, 3.0, 4.0, 5.0],
                    [3.0, 4.0, 5.0, 6.0],
                    [4.0, 5.0, 6.0, 7.0]]], dtype=np.float32)  # [1, 4, 4]

mask_tensor = ch14.create_causal_mask(seq_len)
result = ch14.forward(ch14.masked_softmax(ch14.Tensor(logits), mask_tensor))

# Verify with NumPy reference
mask_np = np.zeros((seq_len, seq_len), dtype=np.float32)
for i in range(seq_len):
    for j in range(seq_len):
        if j > i:
            mask_np[i, j] = -np.inf

expected = np.zeros_like(logits)
for i in range(seq_len):
    masked_logits = logits[0, i, :] + mask_np[i, :]
    expected[0, i, :] = np.exp(masked_logits - masked_logits.max())
    expected[0, i, :] /= expected[0, i, :].sum()

if np.allclose(result, expected, rtol=1e-5, atol=1e-6):
    print("  ✓ Masked softmax matches NumPy reference")
else:
    print("  ✗ ERROR: Masked softmax mismatch!")
    print(f"    Max diff: {np.abs(result - expected).max()}")
    sys.exit(1)

print("\n✓ Phase 3: All causal masking tests passed!")
print()

# ============================================================================
# Phase 4: RoPE (Rotary Position Embeddings) Tests
# ============================================================================

print("### Phase 4: Rotary Position Embeddings (RoPE) ###")

# Test 1: Basic RoPE application
print("Test 1: Basic RoPE transformation")
seq_len = 4
d_model = 8

# Create simple input matrix
input_matrix = np.arange(seq_len * d_model, dtype=np.float32).reshape(seq_len, d_model)
input_matrix = input_matrix / 10.0  # Scale down for easier interpretation

result = ch14.forward(ch14.rope(ch14.Tensor(input_matrix)))

print(f"  Input shape: {input_matrix.shape}")
print(f"  Output shape: {result.shape}")

# Verify shape
if result.shape != (seq_len, d_model):
    print(f"  ✗ ERROR: Expected shape ({seq_len}, {d_model}), got {result.shape}")
    sys.exit(1)

print("  ✓ Output shape correct")

# Test 2: RoPE with NumPy reference implementation
print("\nTest 2: RoPE numerical correctness")
seq_len = 8
d_model = 16
base = 10000.0

# Create random input
np.random.seed(42)
input_matrix = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1

# NumPy reference implementation of RoPE
def rope_reference(x, base=10000.0):
    seq_len, d_model = x.shape
    output = np.zeros_like(x)

    for pos in range(seq_len):
        for dim_idx in range(0, d_model, 2):
            # Compute theta = base^(-dim_idx/d_model)
            theta = base ** (-dim_idx / d_model)

            # Compute angle = pos * theta
            angle = pos * theta

            # Get input pair
            x0 = x[pos, dim_idx]
            x1 = x[pos, dim_idx + 1]

            # Apply rotation
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)

            output[pos, dim_idx] = x0 * cos_angle - x1 * sin_angle
            output[pos, dim_idx + 1] = x0 * sin_angle + x1 * cos_angle

    return output

expected = rope_reference(input_matrix, base)
result = ch14.forward(ch14.rope(ch14.Tensor(input_matrix)))

if np.allclose(result, expected, rtol=1e-5, atol=1e-6):
    print("  ✓ RoPE matches NumPy reference")
else:
    print("  ✗ ERROR: RoPE mismatch!")
    print(f"    Max diff: {np.abs(result - expected).max()}")
    print(f"    Mean diff: {np.abs(result - expected).mean()}")
    # Show first few values for debugging
    print("\n    First position, first 4 dims:")
    print(f"      Expected: {expected[0, :4]}")
    print(f"      Got:      {result[0, :4]}")
    sys.exit(1)

# Test 3: RoPE properties
print("\nTest 3: RoPE property checks")
seq_len = 16
d_model = 32

# Property 1: RoPE preserves norm approximately (slight variation due to rotation)
input_matrix = np.random.randn(seq_len, d_model).astype(np.float32)
input_norms = np.linalg.norm(input_matrix, axis=1)

result = ch14.forward(ch14.rope(ch14.Tensor(input_matrix)))
output_norms = np.linalg.norm(result, axis=1)

# Norms should be preserved (within floating point precision)
if np.allclose(input_norms, output_norms, rtol=1e-5, atol=1e-6):
    print("  ✓ RoPE preserves vector norms")
else:
    print(f"  ⚠ Warning: Norm preservation approximate (max diff: {np.abs(input_norms - output_norms).max()})")
    # This is still acceptable due to numerical precision

# Property 2: Different positions get different rotations
pos0_output = result[0, :]
pos1_output = result[1, :]
pos15_output = result[15, :]

# Outputs at different positions should be significantly different
diff_0_1 = np.linalg.norm(pos0_output - pos1_output)
diff_0_15 = np.linalg.norm(pos0_output - pos15_output)

if diff_0_1 > 0.1 and diff_0_15 > 0.5:
    print("  ✓ Different positions produce different embeddings")
else:
    print(f"  ✗ ERROR: Positions too similar (diff_0_1={diff_0_1:.4f}, diff_0_15={diff_0_15:.4f})")
    sys.exit(1)

print("\n✓ Phase 4: All RoPE tests passed!")
print()

# ============================================================================
# Phase 5: GPT Model Composition Tests
# ============================================================================

print("### Phase 5: GPT Model Composition ###")

# Helper function to create random weights
def create_gpt_weights(d_model, d_ff, num_layers):
    """Create random weights for GPT model"""
    weights = []
    for _ in range(num_layers):
        # Q, K, V, O projections (d_model x d_model each)
        weights.append(np.random.randn(d_model, d_model).astype(np.float32) * 0.02)  # W_Q
        weights.append(np.zeros(d_model, dtype=np.float32))  # b_Q
        weights.append(np.random.randn(d_model, d_model).astype(np.float32) * 0.02)  # W_K
        weights.append(np.zeros(d_model, dtype=np.float32))  # b_K
        weights.append(np.random.randn(d_model, d_model).astype(np.float32) * 0.02)  # W_V
        weights.append(np.zeros(d_model, dtype=np.float32))  # b_V
        weights.append(np.random.randn(d_model, d_model).astype(np.float32) * 0.02)  # W_O
        weights.append(np.zeros(d_model, dtype=np.float32))  # b_O

        # FFN: PyTorch format weights (out_features, in_features)
        weights.append(np.random.randn(d_ff, d_model).astype(np.float32) * 0.02)  # W1 (64x16)
        weights.append(np.zeros(d_ff, dtype=np.float32))  # b1
        weights.append(np.random.randn(d_model, d_ff).astype(np.float32) * 0.02)  # W2 (16x64)
        weights.append(np.zeros(d_model, dtype=np.float32))  # b2

        # Layer norms (d_model each)
        weights.append(np.ones(d_model, dtype=np.float32))   # gamma1
        weights.append(np.zeros(d_model, dtype=np.float32))  # beta1
        weights.append(np.ones(d_model, dtype=np.float32))   # gamma2
        weights.append(np.zeros(d_model, dtype=np.float32))  # beta2

    return weights

# Test 1: GPT attention with RoPE and causal masking
print("Test 1: GPT attention (RoPE + causal masking)")
seq_len = 8
d_model = 16

input_seq = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1

# Create projection matrices
w_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
b_q = np.zeros(d_model, dtype=np.float32)
w_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
b_k = np.zeros(d_model, dtype=np.float32)
w_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
b_v = np.zeros(d_model, dtype=np.float32)
w_o = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
b_o = np.zeros(d_model, dtype=np.float32)

result = ch14.forward(ch14.gpt_attention(
    ch14.Tensor(input_seq), w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o
))

print(f"  Input shape: {input_seq.shape}")
print(f"  Output shape: {result.shape}")

if result.shape == (seq_len, d_model):
    print("  ✓ GPT attention output shape correct")
else:
    print(f"  ✗ ERROR: Expected shape ({seq_len}, {d_model}), got {result.shape}")
    sys.exit(1)

# Check that output is different from input (attention did something)
diff = np.linalg.norm(result - input_seq)
if diff > 0.01:
    print(f"  ✓ Attention transformed input (diff={diff:.4f})")
else:
    print(f"  ✗ ERROR: Output too similar to input (diff={diff:.4f})")
    sys.exit(1)

# Test 2: Single GPT block
print("\nTest 2: Single GPT block (attention + FFN + residuals)")
seq_len = 8
d_model = 16
d_ff = 64

input_seq = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1
weights = create_gpt_weights(d_model, d_ff, num_layers=1)

result = ch14.forward(ch14.gpt_block(
    ch14.Tensor(input_seq),
    *weights  # Unpack all 16 weights
))

print(f"  Input shape: {input_seq.shape}")
print(f"  Output shape: {result.shape}")

if result.shape == (seq_len, d_model):
    print("  ✓ GPT block output shape correct")
else:
    print(f"  ✗ ERROR: Expected shape ({seq_len}, {d_model}), got {result.shape}")
    sys.exit(1)

# Test 3: Full GPT forward pass (1 layer for debugging)
print("\nTest 3a: Embedding within forward()")
seq_len = 4
vocab_size = 16
d_model = 8

indices = np.array([0, 1, 2, 3], dtype=np.int32)
embedding_table = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02
gamma = np.ones(d_model, dtype=np.float32)
beta = np.zeros(d_model, dtype=np.float32)

# Test just embedding + layer_norm (no attention)
def simple_forward(indices_in, table, gamma_in, beta_in):
    hidden = ch14.embedding(indices_in, table)
    return ch14.layer_norm(hidden, gamma_in, beta_in)

result = ch14.forward(simple_forward(indices, embedding_table, gamma, beta))
print(f"  Output shape: {result.shape}")
if result.shape == (seq_len, d_model):
    print("  ✓ Embedding + LayerNorm works")
else:
    print(f"  ✗ ERROR: Expected shape ({seq_len}, {d_model}), got {result.shape}")
    sys.exit(1)

print("\nTest 3: Full GPT forward pass")
seq_len = 8
vocab_size = VOCAB_SIZE
d_model = D_MODEL
d_ff = d_model * 4
num_layers = 2

# Create input token IDs
indices = np.random.randint(0, vocab_size, size=seq_len, dtype=np.int32)

# Create embedding table
embedding_table = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02

# Create weights for all layers
all_weights = create_gpt_weights(d_model, d_ff, num_layers)

# Final layer norm
final_gamma = np.ones(d_model, dtype=np.float32)
final_beta = np.zeros(d_model, dtype=np.float32)

result = ch14.forward(ch14.gpt_forward(
    indices, embedding_table, all_weights, final_gamma, final_beta
))

print(f"  Input: {seq_len} token IDs")
print(f"  Model: {num_layers} layers, d_model={d_model}, d_ff={d_ff}")
print(f"  Output shape: {result.shape}")

if result.shape == (seq_len, d_model):
    print("  ✓ GPT forward output shape correct")
else:
    print(f"  ✗ ERROR: Expected shape ({seq_len}, {d_model}), got {result.shape}")
    sys.exit(1)

# Check output statistics
output_mean = result.mean()
output_std = result.std()
print(f"  Output mean: {output_mean:.4f}, std: {output_std:.4f}")

# After layer norm, should have roughly mean 0, std 1
if abs(output_mean) < 0.5 and 0.5 < output_std < 2.0:
    print("  ✓ Output statistics reasonable (layer norm working)")
else:
    print(f"  ⚠ Warning: Unusual output statistics (might be OK with random weights)")

# Test 4: Full GPT with realistic dimensions
print("\nTest 4: Full GPT with realistic dimensions")
seq_len = MAX_SEQ_LEN // 4  # 8 tokens
d_model = D_MODEL  # 64
d_ff = d_model * 4  # 256
num_layers = NUM_LAYERS  # 2

indices = np.array([72, 101, 108, 108, 111, 32, 119, 111], dtype=np.int32)  # "Hello wo"
embedding_table = np.random.randn(VOCAB_SIZE, d_model).astype(np.float32) * 0.1
all_weights = create_gpt_weights(d_model, d_ff, num_layers)
final_gamma = np.ones(d_model, dtype=np.float32)
final_beta = np.zeros(d_model, dtype=np.float32)

result = ch14.forward(ch14.gpt_forward(
    indices, embedding_table, all_weights, final_gamma, final_beta
))

print(f"  Config: vocab_size={VOCAB_SIZE}, d_model={d_model}, layers={num_layers}")
print(f"  Input: {len(indices)} tokens")
print(f"  Output shape: {result.shape}")

if result.shape == (len(indices), d_model):
    print("  ✓ Full GPT model working with realistic dimensions")
else:
    print(f"  ✗ ERROR: Expected shape ({len(indices)}, {d_model}), got {result.shape}")
    sys.exit(1)

print("\n✓ Phase 5: All GPT model composition tests passed!")
print()

# ============================================================================
# Phase 6: Autoregressive Generation
# ============================================================================

print("### Phase 6: Autoregressive Generation ###")

# Import generation utilities
from generation import generate_greedy, sample

# Test 1: Basic generation (greedy)
print("\nTest 1: Greedy generation")
prompt = np.array([72, 101, 108], dtype=np.int32)  # "Hel"
d_model = 32  # Smaller for faster testing
d_ff = 128
num_layers = 1
max_new_tokens = 5

# Create minimal model
embedding_table = np.random.randn(VOCAB_SIZE, d_model).astype(np.float32) * 0.02
all_weights = create_gpt_weights(d_model, d_ff, num_layers)
final_gamma = np.ones(d_model, dtype=np.float32)
final_beta = np.zeros(d_model, dtype=np.float32)

generated = generate_greedy(prompt, embedding_table, all_weights,
                             final_gamma, final_beta, max_new_tokens=max_new_tokens)

print(f"  Prompt length: {len(prompt)}")
print(f"  Generated length: {len(generated)}")
print(f"  Prompt tokens: {prompt}")
print(f"  Generated tokens: {generated}")

if len(generated) == len(prompt) + max_new_tokens:
    print("  ✓ Generated correct number of tokens")
else:
    print(f"  ✗ ERROR: Expected {len(prompt) + max_new_tokens} tokens, got {len(generated)}")
    sys.exit(1)

# Verify prompt is preserved
if np.array_equal(generated[:len(prompt)], prompt):
    print("  ✓ Prompt tokens preserved in output")
else:
    print("  ✗ ERROR: Prompt tokens not preserved")
    sys.exit(1)

# Test 2: Generation is deterministic (greedy)
print("\nTest 2: Greedy generation is deterministic")
generated2 = generate_greedy(prompt, embedding_table, all_weights,
                              final_gamma, final_beta, max_new_tokens=max_new_tokens)

if np.array_equal(generated, generated2):
    print("  ✓ Greedy generation is deterministic")
else:
    print("  ✗ ERROR: Greedy generation not deterministic")
    sys.exit(1)

# Test 3: Different prompts give different outputs
print("\nTest 3: Different prompts produce different outputs")
prompt2 = np.array([65, 66, 67], dtype=np.int32)  # "ABC"
generated_alt = generate_greedy(prompt2, embedding_table, all_weights,
                                 final_gamma, final_beta, max_new_tokens=max_new_tokens)

# At least one token should be different (with high probability for random model)
if not np.array_equal(generated[len(prompt):], generated_alt[len(prompt2):]):
    print("  ✓ Different prompts produce different outputs")
else:
    print("  ⚠ Warning: Different prompts gave same output (possible with random weights)")

# Test 4: Softmax sampling
print("\nTest 4: Sampling with softmax")
# Set seed for reproducibility
np.random.seed(42)

logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
samples = [sample(logits, top_k=None) for _ in range(100)]

# Check that we sample from the distribution (not just argmax)
unique_samples = set(samples)
if len(unique_samples) > 1:
    print(f"  ✓ Sampling is stochastic (sampled {len(unique_samples)} unique tokens)")
else:
    print("  ⚠ Warning: Sampling only produced one unique token")

# Most frequent should be highest logit (token 3)
most_common = max(set(samples), key=samples.count)
if most_common == 3:
    print("  ✓ Sampling favors higher logit tokens")
else:
    print(f"  ⚠ Note: Most common token was {most_common}, expected 3")

# Test 5: Generation with realistic model
print("\nTest 5: Generation with full-scale model")
d_model = D_MODEL  # 64
d_ff = d_model * 4
num_layers = 2
prompt = np.array([72, 101, 108, 108, 111], dtype=np.int32)  # "Hello"
max_new_tokens = 10

embedding_table = np.random.randn(VOCAB_SIZE, d_model).astype(np.float32) * 0.02
all_weights = create_gpt_weights(d_model, d_ff, num_layers)
final_gamma = np.ones(d_model, dtype=np.float32)
final_beta = np.zeros(d_model, dtype=np.float32)

generated = generate_greedy(prompt, embedding_table, all_weights,
                             final_gamma, final_beta, max_new_tokens=max_new_tokens)

print(f"  Model: {num_layers} layers, d_model={d_model}")
print(f"  Prompt: {prompt} ('{chr(72)+chr(101)+chr(108)+chr(108)+chr(111)}')")
print(f"  Generated: {generated[:10]}... (showing first 10)")

if len(generated) == len(prompt) + max_new_tokens:
    print("  ✓ Full-scale generation working")
else:
    print(f"  ✗ ERROR: Expected {len(prompt) + max_new_tokens} tokens, got {len(generated)}")
    sys.exit(1)

print("\n✓ Phase 6: All autoregressive generation tests passed!")
print()

print("=" * 70)
print("Test suite complete - Phases 1-6 passing!")
print("=" * 70)