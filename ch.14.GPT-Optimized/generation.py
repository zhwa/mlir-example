"""
Autoregressive text generation for minimal GPT
"""

import numpy as np
import sys
import os

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
    import ch14 as ch13  # Alias to minimize code changes
except ImportError as e:
    print(f"Error: Could not import ch14 module: {e}")
    print("Please build Chapter 14 first:")
    print("  cmake --build build/x64-release --target ch14")
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

def generate_cached(prompt_tokens, embedding_table, all_weights, final_gamma, final_beta,
                    max_new_tokens=20, max_seq_len=512, temperature=1.0, top_k=None):
    """
    Generate text with KV caching for 10-100x speedup.
    
    Only computes attention for new tokens, reusing cached K/V from previous tokens.
    """
    vocab_size = embedding_table.shape[0]
    d_model = embedding_table.shape[1]
    num_layers = len(all_weights) // 16
    
    # Initialize KV caches for each layer [max_seq_len, d_model]
    k_caches = [np.zeros((max_seq_len, d_model), dtype=np.float32) for _ in range(num_layers)]
    v_caches = [np.zeros((max_seq_len, d_model), dtype=np.float32) for _ in range(num_layers)]
    
    # Process prompt (fill caches)
    tokens = list(prompt_tokens)
    prompt_len = len(prompt_tokens)
    
    # For prompt, we still do full forward pass but cache the K/V
    current_tokens = np.array(tokens, dtype=np.int32)
    hidden = ch13.forward(ch13.gpt_forward(
        current_tokens, embedding_table, all_weights, final_gamma, final_beta
    ))
    
    # Extract K/V from prompt processing and fill caches
    # (In a real implementation, we'd need to extract K/V during forward pass)
    # For now, we'll do a simplified version: process prompt token-by-token to fill caches
    
    # Actually, let's process tokens incrementally from the start
    tokens = []
    
    # Process each prompt token incrementally
    for pos, token_id in enumerate(prompt_tokens):
        # Get embedding for this token [d_model]
        token_emb = embedding_table[token_id]
        hidden = token_emb.copy()
        
        # Apply each transformer layer with caching
        for layer in range(num_layers):
            base = layer * 16
            
            # Layer norm 1
            gamma1 = all_weights[base + 12]
            beta1 = all_weights[base + 13]
            normed = _layer_norm_cpu(hidden.reshape(1, -1), gamma1, beta1)[0]
            
            # Cached attention
            attn_out = ch13.gpt_attention_cached(
                normed.reshape(1, -1),
                k_caches[layer], v_caches[layer], pos,
                all_weights[base + 0], all_weights[base + 1],   # Q
                all_weights[base + 2], all_weights[base + 3],   # K
                all_weights[base + 4], all_weights[base + 5],   # V
                all_weights[base + 6], all_weights[base + 7]    # O
            )[0]
            
            hidden = hidden + attn_out  # Residual
            
            # Layer norm 2 + FFN (no caching needed)
            gamma2 = all_weights[base + 14]
            beta2 = all_weights[base + 15]
            normed2 = _layer_norm_cpu(hidden.reshape(1, -1), gamma2, beta2)[0]
            
            # FFN
            w1, b1 = all_weights[base + 8], all_weights[base + 9]
            w2, b2 = all_weights[base + 10], all_weights[base + 11]
            ffn_out = _ffn_cpu(normed2.reshape(1, -1), w1, b1, w2, b2)[0]
            
            hidden = hidden + ffn_out  # Residual
        
        # Final layer norm
        hidden = _layer_norm_cpu(hidden.reshape(1, -1), final_gamma, final_beta)[0]
        tokens.append(token_id)
    
    # Generate new tokens
    for gen_step in range(max_new_tokens):
        pos = prompt_len + gen_step
        
        # Project last hidden to vocabulary
        logits = hidden @ embedding_table.T
        logits = logits / temperature
        next_token = sample(logits, top_k=top_k)
        tokens.append(next_token)
        
        # Get embedding for new token
        token_emb = embedding_table[next_token]
        hidden = token_emb.copy()
        
        # Apply layers with caching
        for layer in range(num_layers):
            base = layer * 16
            
            gamma1 = all_weights[base + 12]
            beta1 = all_weights[base + 13]
            normed = _layer_norm_cpu(hidden.reshape(1, -1), gamma1, beta1)[0]
            
            # Cached attention (reuses all previous K/V)
            attn_out = ch13.gpt_attention_cached(
                normed.reshape(1, -1),
                k_caches[layer], v_caches[layer], pos,
                all_weights[base + 0], all_weights[base + 1],
                all_weights[base + 2], all_weights[base + 3],
                all_weights[base + 4], all_weights[base + 5],
                all_weights[base + 6], all_weights[base + 7]
            )[0]
            
            hidden = hidden + attn_out
            
            gamma2 = all_weights[base + 14]
            beta2 = all_weights[base + 15]
            normed2 = _layer_norm_cpu(hidden.reshape(1, -1), gamma2, beta2)[0]
            
            w1, b1 = all_weights[base + 8], all_weights[base + 9]
            w2, b2 = all_weights[base + 10], all_weights[base + 11]
            ffn_out = _ffn_cpu(normed2.reshape(1, -1), w1, b1, w2, b2)[0]
            
            hidden = hidden + ffn_out
        
        hidden = _layer_norm_cpu(hidden.reshape(1, -1), final_gamma, final_beta)[0]
    
    return np.array(tokens, dtype=np.int32)

def _layer_norm_cpu(x, gamma, beta, eps=1e-5):
    """CPU implementation of layer norm for cached generation"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

def _ffn_cpu(x, w1, b1, w2, b2):
    """CPU implementation of FFN for cached generation"""
    # Linear 1
    hidden = x @ w1 + b1
    # GELU
    hidden = 0.5 * hidden * (1 + np.tanh(np.sqrt(2 / np.pi) * (hidden + 0.044715 * hidden**3)))
    # Linear 2
    return hidden @ w2 + b2

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