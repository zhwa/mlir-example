"""
Model Executor - Wrapper for Chapter 14 MLIR JIT

Executes GPT model forward passes using Chapter 14's optimized MLIR JIT.
"""

import sys
import os
import numpy as np
from typing import Dict, Any, Optional
sys.path.insert(0, '.')

# Import Chapter 14's MLIR JIT module
try:
    build_paths = [
        '../build/x64-release/ch.14.GPT-Optimized',
        '../build/x64-debug/ch.14.GPT-Optimized',
        '../../build/x64-release/ch.14.GPT-Optimized',
        '../../build/x64-debug/ch.14.GPT-Optimized',
    ]

    ch14 = None
    for path in build_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            try:
                import ch14
                break
            except ImportError:
                sys.path.pop(0)

    if ch14 is None:
        import ch14

except ImportError as e:
    print(f"Warning: Could not import ch14 module: {e}")
    print("Phase 2 executor requires Chapter 14 to be built.")
    print("  cmake --build --preset x64-release --target ch14")
    ch14 = None

from python.batch import Batch
from python.kv_pool import KVCachePool

class ModelConfig:
    """GPT model configuration"""
    def __init__(
        self,
        vocab_size: int = 50257,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        max_seq_len: int = 1024
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.max_seq_len = max_seq_len
        self.head_dim = n_embd // n_head

class ModelExecutor:
    """
    Executes GPT model using Chapter 14's MLIR JIT

    Handles both prefill (full prompt) and decode (single token) phases.
    Uses cached attention for decode to avoid recomputing KV for past tokens.
    """

    def __init__(self, config: ModelConfig, weights: Dict[str, np.ndarray], kv_pool: KVCachePool):
        """
        Initialize model executor

        Args:
            config: Model configuration
            weights: Model weights (token_emb, position_emb, blocks, ln_f, lm_head)
            kv_pool: KV cache pool for attention caching
        """
        if ch14 is None:
            raise RuntimeError("Chapter 14 module not available. Build ch14 first.")

        self.config = config
        self.weights = weights
        self.kv_pool = kv_pool

        # Validate weights
        required_keys = ['token_emb', 'position_emb', 'ln_f.gamma', 'ln_f.beta', 'lm_head']
        for key in required_keys:
            if key not in weights:
                raise ValueError(f"Missing weight: {key}")

    def execute_prefill(self, batch: Batch) -> np.ndarray:
        """
        Execute prefill phase

        Processes all prompt tokens in parallel to:
        1. Compute KV cache for all prompt tokens
        2. Generate logits for next token prediction

        Args:
            batch: Prefill batch with concatenated prompt tokens

        Returns:
            Logits array [total_tokens, vocab_size]
            (Use last token's logits for each request)
        """
        if not batch.is_prefill:
            raise ValueError("Batch is not a prefill batch")

        # For simplicity, we'll call ch14.gpt_forward
        # In production, this would:
        # 1. Call attention layers with full context
        # 2. Store K/V in cache pool
        # 3. Return logits

        # Mock implementation for now - returns random logits
        # TODO: Integrate actual MLIR JIT execution with KV cache storage
        total_tokens = len(batch.input_ids)
        logits = np.random.randn(total_tokens, self.config.vocab_size).astype(np.float32)

        return logits

    def execute_decode(self, batch: Batch) -> np.ndarray:
        """
        Execute decode phase

        Generates one token for each request using cached KV.
        Only processes the last token of each request.

        Args:
            batch: Decode batch with one token per request

        Returns:
            Logits array [batch_size, vocab_size]
        """
        if batch.is_prefill:
            raise ValueError("Batch is not a decode batch")

        # For simplicity, mock implementation
        # In production, this would:
        # 1. Call ch14.gpt_attention_cached with K/V cache
        # 2. Only compute attention for new token
        # 3. Return logits for next token

        # Mock implementation - returns random logits
        # TODO: Integrate cached attention from Chapter 14
        batch_size = len(batch.input_ids)
        logits = np.random.randn(batch_size, self.config.vocab_size).astype(np.float32)

        return logits

    def sample(self, logits: np.ndarray, temperature: float = 1.0, top_k: int = 50) -> int:
        """
        Sample next token from logits

        Args:
            logits: Logits for single position [vocab_size]
            temperature: Sampling temperature (higher = more random)
            top_k: Only sample from top k tokens

        Returns:
            Sampled token ID
        """
        # Apply temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            indices = np.argpartition(logits, -top_k)[-top_k:]
            top_logits = logits[indices]
            # Softmax over top-k
            probs = np.exp(top_logits - np.max(top_logits))
            probs = probs / np.sum(probs)
            # Sample from top-k
            token_idx = np.random.choice(len(indices), p=probs)
            return int(indices[token_idx])
        else:
            # Softmax over all tokens
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
            return int(np.random.choice(len(logits), p=probs))