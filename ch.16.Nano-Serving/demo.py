#!/usr/bin/env python3
"""
Nano LLM Serving - Interactive Text Generation Demo

Demonstrates production-grade LLM serving with continuous batching,
KV cache management, chunked prefill, and radix caching.

Uses random weights (not trained), so output is random but shows the
complete serving pipeline with all optimizations.

Usage:
    python3 demo.py
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from nano_engine import NanoServingEngine, SamplingParams
from executor import ModelConfig

def text_to_tokens(text):
    """Convert text to token IDs (byte-level)"""
    return [ord(c) for c in text]

def tokens_to_text(tokens):
    """Convert token IDs to text (clamped to printable ASCII)"""
    # Clamp to printable range for demo
    printable = [min(max(32, t), 126) for t in tokens]
    return ''.join(chr(b) for b in printable)

def main():
    print("=" * 70)
    print("Nano LLM Serving - Production-Grade Inference Demo")
    print("=" * 70)
    print()
    print("⚠️  IMPORTANT: This model has RANDOM WEIGHTS (not trained)")
    print("   Output will be random - this demonstrates the serving pipeline.")
    print("   Real LLM serving uses trained models with learned weights.")
    print()

    # Model config
    vocab_size = 256
    d_model = 64
    num_layers = 2
    num_heads = 4
    max_seq_len = 128

    print("Model Configuration:")
    print(f"  • Vocabulary: {vocab_size} tokens (byte-level)")
    print(f"  • Architecture: {num_layers}-layer transformer")
    print(f"  • Hidden dimension: {d_model}")
    print(f"  • Attention heads: {num_heads}")
    print(f"  • Max sequence length: {max_seq_len}")
    print()

    print("Serving Features:")
    print("  ✓ Continuous batching (dynamic request scheduling)")
    print("  ✓ KV cache management (efficient memory pooling)")
    print("  ✓ Chunked prefill (interrupt-friendly prefill)")
    print("  ✓ Radix caching (prefix sharing across requests)")
    print()

    # Initialize model
    print("Initializing random model...")
    
    # Initialize serving engine
    config = ModelConfig(
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_embd=d_model,
        max_seq_len=max_seq_len
    )

    # Create weights dictionary
    weights = {
        'token_emb': np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02,
        'position_emb': np.random.randn(max_seq_len, d_model).astype(np.float32) * 0.02,
        'ln_f.gamma': np.ones(d_model, dtype=np.float32),
        'ln_f.beta': np.zeros(d_model, dtype=np.float32),
        'lm_head': np.random.randn(d_model, vocab_size).astype(np.float32) * 0.02,
    }

    engine = NanoServingEngine(
        config=config,
        weights=weights,
        kv_cache_pages=256,
        max_chunk_size=256,
        max_batch_size=32,
        eos_token_id=0
    )

    print("✓ Serving engine ready")
    print()

    # Demo 1: Single request
    print("─" * 70)
    print("Demo 1: Single Request Generation")
    print("─" * 70)

    prompt = "Hello"
    prompt_tokens = text_to_tokens(prompt)
    print(f"Prompt: '{prompt}'")
    print(f"Generating with max_tokens=20...\n")

    params = SamplingParams(max_tokens=20, temperature=1.0, ignore_eos=True)
    results = engine.generate([prompt_tokens], [params])

    output_tokens = results[0].output_tokens
    output_text = tokens_to_text(output_tokens)
    print(f"Output: '{output_text}'")
    print()

    # Demo 2: Batch requests (demonstrates continuous batching)
    print("─" * 70)
    print("Demo 2: Batched Requests (Continuous Batching)")
    print("─" * 70)
    print("Processing multiple requests in parallel...")
    print()

    prompts = ["AI", "GPT", "MLIR", "Code"]
    prompt_token_lists = [text_to_tokens(p) for p in prompts]
    sampling_params = [
        SamplingParams(max_tokens=10, temperature=0.8, ignore_eos=True)
        for _ in prompts
    ]

    results = engine.generate(prompt_token_lists, sampling_params)

    for i, (prompt, result) in enumerate(zip(prompts, results)):
        output = tokens_to_text(result.output_tokens)
        print(f"Request {i+1}: '{prompt}' → '{output}'")

    print()

    # Demo 3: Prefix sharing (demonstrates radix cache)
    print("─" * 70)
    print("Demo 3: Prefix Sharing (Radix Cache)")
    print("─" * 70)
    print("Shared prefix: 'Hello World'")
    print("Different suffixes demonstrate cache reuse...\n")

    base_prompt = "Hello World"
    suffixes = [" AI", " GPT", " MLIR"]

    for suffix in suffixes:
        full_prompt = base_prompt + suffix
        prompt_tokens = text_to_tokens(full_prompt)
        params = SamplingParams(max_tokens=8, temperature=0.8, ignore_eos=True)

        results = engine.generate([prompt_tokens], [params])
        output = tokens_to_text(results[0].output_tokens)
        print(f"'{full_prompt}' → '{output}'")

    print()

    # Show statistics
    print("─" * 70)
    print("Serving Statistics")
    print("─" * 70)
    stats = engine.get_stats()
    print(f"Total requests processed: {stats['total_requests_served']}")
    print(f"Total tokens generated: {stats['total_tokens_generated']}")
    print(f"KV cache utilization: {stats['memory_utilization']:.1%} ({stats['num_total_pages'] - stats['num_free_pages']}/{stats['num_total_pages']} pages)")
    print(f"Radix cache hit rate: {stats.get('cache_hit_rate', 0):.1%} ({stats.get('cache_hits', 0)} hits, {stats.get('cache_misses', 0)} misses)")
    print()

    # Explanation
    print("=" * 70)
    print("Understanding Production LLM Serving")
    print("=" * 70)
    print()
    print("Why is the output random?")
    print("  • Model weights are randomly initialized (not trained)")
    print("  • Real serving systems use trained model checkpoints")
    print()
    print("What does this demonstrate?")
    print("  ✓ Continuous batching: Dynamic scheduling of multiple requests")
    print("  ✓ KV cache pool: Efficient memory management with block allocation")
    print("  ✓ Chunked prefill: Interruptible prefill for responsiveness")
    print("  ✓ Radix cache: Prefix sharing reduces redundant computation")
    print("  ✓ Request scheduling: Prefill/decode separation with priorities")
    print()
    print("Production Optimizations:")
    print("  • Memory: Block-based KV cache eliminates fragmentation")
    print("  • Throughput: Continuous batching maximizes GPU utilization")
    print("  • Latency: Chunked prefill prevents decode starvation")
    print("  • Efficiency: Radix cache exploits common prefixes (e.g., system prompts)")
    print()
    print("This demonstrates a complete production LLM serving pipeline!")
    print("=" * 70)

if __name__ == "__main__":
    main()
