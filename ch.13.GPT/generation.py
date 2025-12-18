"""
Autoregressive text generation for minimal GPT
"""

import numpy as np
import sys
import os

# Add build directory to path
build_paths = [
    '../build/x64-release/ch.13.GPT',
    '../build/x64-debug/ch.13.GPT',
    'build/x64-release/ch.13.GPT',
    'build/x64-debug/ch.13.GPT'
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
    import ch13
except ImportError as e:
    print(f"Error: Could not import ch13 module: {e}")
    print("Please build Chapter 13 first:")
    print("  cmake --build build/x64-release --target ch13")
    sys.exit(1)

def generate(prompt_tokens, embedding_table, all_weights, final_gamma, final_beta,
             max_new_tokens=20, temperature=1.0, top_k=None):
    """
    Generate text autoregressively from a prompt.

    Args:
        prompt_tokens: np.array of int32, shape [prompt_len]
        embedding_table: np.array of float32, shape [vocab_size, d_model]
        all_weights: list of weight arrays for transformer blocks
        final_gamma: np.array of float32, shape [d_model]
        final_beta: np.array of float32, shape [d_model]
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature (1.0 = no change, <1 = more confident, >1 = more random)
        top_k: if set, only sample from top k logits (None = sample from all)

    Returns:
        np.array of int32, shape [prompt_len + max_new_tokens]
    """
    vocab_size = embedding_table.shape[0]
    d_model = embedding_table.shape[1]

    # Start with prompt
    tokens = list(prompt_tokens)

    # Generate one token at a time
    for _ in range(max_new_tokens):
        # Current sequence
        current_tokens = np.array(tokens, dtype=np.int32)

        # Forward pass: get hidden states [seq_len, d_model]
        hidden = ch13.forward(ch13.gpt_forward(
            current_tokens, embedding_table, all_weights, final_gamma, final_beta
        ))

        # Extract last position hidden state [d_model]
        last_hidden = hidden[-1, :]

        # Project to vocabulary: logits = last_hidden @ embedding_table.T
        # (Using embedding table as output projection - "tied weights")
        logits = last_hidden @ embedding_table.T  # [vocab_size]

        # Apply temperature
        logits = logits / temperature

        # Sample next token
        next_token = sample(logits, top_k=top_k)

        # Append to sequence
        tokens.append(next_token)

    return np.array(tokens, dtype=np.int32)

def sample(logits, top_k=None):
    """
    Sample a token from logits.

    Args:
        logits: np.array of float32, shape [vocab_size]
        top_k: if set, only sample from top k logits

    Returns:
        int: sampled token index
    """
    if top_k is not None:
        # Get top k indices
        top_indices = np.argpartition(logits, -top_k)[-top_k:]
        # Mask other logits with -inf
        masked_logits = np.full_like(logits, -np.inf)
        masked_logits[top_indices] = logits[top_indices]
        logits = masked_logits

    # Softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    probs = exp_logits / np.sum(exp_logits)

    # Sample from categorical distribution
    token = np.random.choice(len(probs), p=probs)
    return int(token)

def generate_greedy(prompt_tokens, embedding_table, all_weights, final_gamma, final_beta,
                    max_new_tokens=20):
    """
    Generate text greedily (always pick most likely token).
    Faster and deterministic, good for testing.
    """
    return generate(prompt_tokens, embedding_table, all_weights, final_gamma, final_beta,
                    max_new_tokens=max_new_tokens, temperature=1e-8)  # Very low temp = greedy

if __name__ == "__main__":
    # Demo
    print("=== Minimal GPT Text Generation Demo ===\n")

    # Tiny model for demo
    vocab_size = 256  # Byte-level
    d_model = 64
    d_ff = 256
    num_layers = 2

    # Random weights (not trained)
    embedding_table = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02

    # Create random transformer weights
    all_weights = []
    for _ in range(num_layers):
        for _ in range(4):  # Q, K, V, O
            all_weights.append(np.random.randn(d_model, d_model).astype(np.float32) * 0.02)
            all_weights.append(np.zeros(d_model, dtype=np.float32))
        # FFN
        all_weights.append(np.random.randn(d_model, d_ff).astype(np.float32) * 0.02)
        all_weights.append(np.zeros(d_ff, dtype=np.float32))
        all_weights.append(np.random.randn(d_ff, d_model).astype(np.float32) * 0.02)
        all_weights.append(np.zeros(d_model, dtype=np.float32))
        # Layer norms
        all_weights.append(np.ones(d_model, dtype=np.float32))
        all_weights.append(np.zeros(d_model, dtype=np.float32))
        all_weights.append(np.ones(d_model, dtype=np.float32))
        all_weights.append(np.zeros(d_model, dtype=np.float32))

    final_gamma = np.ones(d_model, dtype=np.float32)
    final_beta = np.zeros(d_model, dtype=np.float32)

    # Prompt: "Hello" as bytes
    prompt = "Hello"
    prompt_tokens = np.array([ord(c) for c in prompt], dtype=np.int32)

    print(f"Prompt: '{prompt}'")
    print(f"Prompt tokens: {prompt_tokens}")
    print()

    # Generate (greedy for determinism in testing)
    print("Generating 10 tokens (greedy)...")
    generated = generate_greedy(prompt_tokens, embedding_table, all_weights,
                                 final_gamma, final_beta, max_new_tokens=10)

    # Decode (clamp to printable ASCII for demo)
    generated_bytes = [min(max(32, t), 126) for t in generated]
    generated_text = ''.join(chr(b) for b in generated_bytes)

    print(f"Generated sequence: {generated}")
    print(f"Decoded text: '{generated_text}'")
    print()
    print("Note: This is a randomly initialized model, so output is random!")
    print("A trained model would generate coherent text.")