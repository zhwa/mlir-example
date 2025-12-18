#!/usr/bin/env python3
"""
Minimal GPT - Interactive Text Generation Demo

Demonstrates autoregressive text generation with a minimal GPT model.
Uses random weights (not trained), so output is random but shows the mechanism.

Usage:
    python3 generate.py
"""

import sys
sys.path.insert(0, '../build/x64-release/ch.13.GPT')

import numpy as np
import ch13
from generation import generate, generate_greedy


def create_random_model(vocab_size=256, d_model=64, num_layers=2):
    """Initialize a GPT model with random weights"""
    d_ff = d_model * 4
    
    # Embedding table
    embedding_table = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02
    
    # Transformer weights
    all_weights = []
    for _ in range(num_layers):
        # Q, K, V, O projections
        for _ in range(4):
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
    
    # Final layer norm
    final_gamma = np.ones(d_model, dtype=np.float32)
    final_beta = np.zeros(d_model, dtype=np.float32)
    
    return embedding_table, all_weights, final_gamma, final_beta


def text_to_tokens(text):
    """Convert text to token IDs (byte-level)"""
    return np.array([ord(c) for c in text], dtype=np.int32)


def tokens_to_text(tokens):
    """Convert token IDs to text (clamped to printable ASCII)"""
    # Clamp to printable range for demo
    printable = [min(max(32, t), 126) for t in tokens]
    return ''.join(chr(b) for b in printable)


def main():
    print("=" * 70)
    print("Minimal GPT - Text Generation Demo")
    print("=" * 70)
    print()
    print("⚠️  IMPORTANT: This model has RANDOM WEIGHTS (not trained)")
    print("   Output will be random - this demonstrates the generation mechanism.")
    print("   Real GPT models are trained on billions of tokens of text data.")
    print()
    
    # Model config
    print("Model Configuration:")
    print("  • Vocabulary: 256 tokens (byte-level)")
    print("  • Architecture: 2-layer transformer with RoPE and causal masking")
    print("  • Hidden dimension: 64")
    print("  • Feed-forward dimension: 256")
    print("  • Attention heads: 4")
    print()
    
    # Initialize
    print("Initializing random model...")
    embedding_table, all_weights, final_gamma, final_beta = create_random_model()
    print("✓ Model ready")
    print()
    
    # Demo 1: Basic generation
    print("─" * 70)
    print("Demo 1: Basic Generation")
    print("─" * 70)
    
    prompt = "Hello"
    prompt_tokens = text_to_tokens(prompt)
    print(f"Prompt: '{prompt}'")
    print(f"Generating 20 tokens (greedy)...\n")
    
    generated = generate_greedy(prompt_tokens, embedding_table, all_weights,
                                 final_gamma, final_beta, max_new_tokens=20)
    output = tokens_to_text(generated)
    print(f"Output: '{output}'")
    print()
    
    # Demo 2: Temperature comparison
    print("─" * 70)
    print("Demo 2: Temperature Effects")
    print("─" * 70)
    print(f"Prompt: '{prompt}'\n")
    
    for temp in [0.5, 1.0, 1.5]:
        np.random.seed(42)  # For reproducibility
        generated = generate(prompt_tokens, embedding_table, all_weights,
                             final_gamma, final_beta, max_new_tokens=15,
                             temperature=temp)
        output = tokens_to_text(generated)
        print(f"T={temp}: '{output}'")
    
    print()
    
    # Demo 3: Multiple prompts
    print("─" * 70)
    print("Demo 3: Different Prompts")
    print("─" * 70)
    
    prompts = ["AI", "GPT", "MLIR", "Code"]
    for p in prompts:
        p_tokens = text_to_tokens(p)
        generated = generate_greedy(p_tokens, embedding_table, all_weights,
                                     final_gamma, final_beta, max_new_tokens=10)
        output = tokens_to_text(generated)
        print(f"'{p}' → '{output}'")
    
    print()
    
    # Explanation
    print("=" * 70)
    print("Understanding the Results")
    print("=" * 70)
    print()
    print("Why is the output random/repetitive?")
    print("  • Model weights are randomly initialized (not trained)")
    print("  • No language patterns learned from text data")
    print("  • No optimization or gradient descent performed")
    print()
    print("What does this demonstrate?")
    print("  ✓ Complete GPT architecture implementation")
    print("  ✓ Token embeddings with RoPE positional encoding")
    print("  ✓ Causal self-attention (no future peeking)")
    print("  ✓ Feed-forward networks with GELU activation")
    print("  ✓ Pre-norm transformer blocks with residual connections")
    print("  ✓ Autoregressive generation loop")
    print("  ✓ Vocabulary projection and sampling")
    print()
    print("For real text generation, you would need:")
    print("  • Training on large text corpus (e.g., books, Wikipedia)")
    print("  • Proper tokenization (BPE/WordPiece, not byte-level)")
    print("  • Larger model (billions of parameters)")
    print("  • Training infrastructure (GPUs, distributed training)")
    print("  • Optimization (Adam, learning rate schedules)")
    print()
    print("This is an educational minimal GPT demonstrating the core concepts!")
    print("=" * 70)


if __name__ == "__main__":
    main()
