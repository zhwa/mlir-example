#!/usr/bin/env python3
"""
Test to verify KV-caching is working correctly

This test demonstrates that:
1. Prefill phase creates and stores K/V caches
2. Decode phase reuses cached K/V tensors
3. Cache grows as new tokens are generated
"""

import sys
sys.path.insert(0, 'python')

from python.executor import ModelExecutor, ModelConfig
from python.kv_pool import KVCachePool
from python.request import Request
from python.batch import Batch
import numpy as np

def test_kv_caching():
    """Verify that KV-caching is properly integrated"""

    print("=" * 70)
    print("KV-Caching Integration Test")
    print("=" * 70)

    # Setup
    config = ModelConfig(vocab_size=50257, n_layer=2, n_head=4, n_embd=128)
    weights = {'token_emb': np.random.randn(config.vocab_size, config.n_embd).astype(np.float32) * 0.02}
    kv_pool = KVCachePool(
        num_pages=100,
        page_size=16,
        num_layers=config.n_layer,
        num_heads=config.n_head,
        head_dim=config.head_dim
    )
    executor = ModelExecutor(config, weights, kv_pool)

    # Create a test request
    req = Request(
        req_id=1,
        prompt_tokens=[1, 2, 3, 4, 5],
        max_tokens=3
    )

    print("\nStep 1: Prefill Phase")
    print(f"  Processing prompt: {req.prompt_tokens}")

    # Create prefill batch
    prefill_batch = Batch.from_prefill([req])
    prefill_logits = executor.execute_prefill(prefill_batch)

    print(f"  ✓ Prefill completed")
    print(f"  ✓ Generated logits shape: {prefill_logits.shape}")

    # Check that K/V caches were created
    assert req.k_caches is not None, "K caches should be created"
    assert req.v_caches is not None, "V caches should be created"
    assert len(req.k_caches) == config.n_layer, f"Should have {config.n_layer} K caches"
    assert len(req.v_caches) == config.n_layer, f"Should have {config.n_layer} V caches"

    print(f"  ✓ KV caches created: {len(req.k_caches)} layers")

    # Check cache shapes
    for layer_idx, (k_cache, v_cache) in enumerate(zip(req.k_caches, req.v_caches)):
        print(f"    Layer {layer_idx}: K shape {k_cache.shape}, V shape {v_cache.shape}")
        expected_seq_len = len(req.prompt_tokens)
        assert k_cache.shape[0] == expected_seq_len, f"K cache seq_len should be {expected_seq_len}"
        assert v_cache.shape[0] == expected_seq_len, f"V cache seq_len should be {expected_seq_len}"
        assert k_cache.shape[1] == config.n_embd, f"K cache d_model should be {config.n_embd}"
        assert v_cache.shape[1] == config.n_embd, f"V cache d_model should be {config.n_embd}"

    # Generate tokens using decode
    print("\nStep 2: Decode Phase (3 tokens)")

    for step in range(3):
        # Add generated token
        new_token = 10 + step  # Dummy token
        req.output_tokens.append(new_token)

        # Create decode batch
        decode_batch = Batch.from_decode([req])

        # Store cache size before decode
        prev_cache_len = req.k_caches[0].shape[0] if req.k_caches else 0

        # Execute decode
        decode_logits = executor.execute_decode(decode_batch)

        # Check that caches grew
        new_cache_len = req.k_caches[0].shape[0]

        print(f"  Step {step + 1}:")
        print(f"    Generated token: {new_token}")
        print(f"    KV cache grew: {prev_cache_len} → {new_cache_len} tokens")
        print(f"    Logits shape: {decode_logits.shape}")

        # Verify cache grew by 1
        assert new_cache_len == prev_cache_len + 1, \
            f"Cache should grow by 1 token (was {prev_cache_len}, now {new_cache_len})"

    print("\n" + "=" * 70)
    print("KV-Caching Integration Test PASSED!")
    print("=" * 70)
    return True

if __name__ == '__main__':
    try:
        success = test_kv_caching()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)