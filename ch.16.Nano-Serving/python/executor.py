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
    from python.cpp_modules import ch14
except ImportError as e:
    print(f"Warning: Could not import ch14 module: {e}")
    print("Phase 2 executor requires Chapter 14 to be built.")
    print("  cmake --build --preset x64-release --target ch14")
    ch14 = None

# Import KVCachePool as alias
KVCachePool = ch14.KVCachePool if ch14 else None

from python.batch import Batch

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

def _create_ch14_compatible_weights(config: ModelConfig):
    """
    Create weight arrays in format expected by ch14.gpt_forward

    Uses random initialization for educational purposes.
    This demonstrates the serving infrastructure without requiring trained models.

    Returns:
        all_weights: List of 16 arrays per layer
            Per layer: W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O,
                      W1, b1, W2, b2, gamma1, beta1, gamma2, beta2
    """
    d_model = config.n_embd
    d_ff = 4 * d_model  # Standard transformer expansion
    num_layers = config.n_layer

    weights = []
    for _ in range(num_layers):
        # Attention projections: Q, K, V, O (each d_model x d_model)
        weights.append(np.random.randn(d_model, d_model).astype(np.float32) * 0.02)  # W_Q
        weights.append(np.zeros(d_model, dtype=np.float32))  # b_Q
        weights.append(np.random.randn(d_model, d_model).astype(np.float32) * 0.02)  # W_K
        weights.append(np.zeros(d_model, dtype=np.float32))  # b_K
        weights.append(np.random.randn(d_model, d_model).astype(np.float32) * 0.02)  # W_V
        weights.append(np.zeros(d_model, dtype=np.float32))  # b_V
        weights.append(np.random.randn(d_model, d_model).astype(np.float32) * 0.02)  # W_O
        weights.append(np.zeros(d_model, dtype=np.float32))  # b_O

        # FFN: W1 (d_ff, d_model), W2 (d_model, d_ff)
        weights.append(np.random.randn(d_ff, d_model).astype(np.float32) * 0.02)  # W1
        weights.append(np.zeros(d_ff, dtype=np.float32))  # b1
        weights.append(np.random.randn(d_model, d_ff).astype(np.float32) * 0.02)  # W2
        weights.append(np.zeros(d_model, dtype=np.float32))  # b2

        # Layer norms
        weights.append(np.ones(d_model, dtype=np.float32))   # gamma1
        weights.append(np.zeros(d_model, dtype=np.float32))  # beta1
        weights.append(np.ones(d_model, dtype=np.float32))   # gamma2
        weights.append(np.zeros(d_model, dtype=np.float32))  # beta2

    return weights

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

        # Organize weights for ch14.gpt_forward
        # Using random weights for educational demonstration of serving infrastructure
        self.embedding_table = weights.get('token_emb', 
            np.random.randn(config.vocab_size, config.n_embd).astype(np.float32) * 0.02)
        self.all_weights = _create_ch14_compatible_weights(config)
        self.final_gamma = weights.get('ln_f.gamma', 
            np.ones(config.n_embd, dtype=np.float32))
        self.final_beta = weights.get('ln_f.beta', 
            np.zeros(config.n_embd, dtype=np.float32))

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

        # Call ch14.gpt_forward_prefill to get hidden states + KV caches
        token_ids = np.array(batch.input_ids, dtype=np.int32)

        result = ch14.gpt_forward_prefill(
            token_ids,
            self.embedding_table,
            self.all_weights,
            self.final_gamma,
            self.final_beta
        )

        hidden_states = result.hidden_states
        k_caches = result.k_caches
        v_caches = result.v_caches

        # Project to vocabulary: logits = hidden @ embedding_table.T
        logits = hidden_states @ self.embedding_table.T

        # Safety: Check for NaN/Inf and clip to reasonable range
        if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
            print(f"Warning: NaN/Inf detected in prefill logits, clipping values")
            logits = np.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)

        logits = np.clip(logits, -100.0, 100.0)

        # Store K/V caches in requests for decode phase
        for req in batch.requests:
            seq_len = len(req.prompt_tokens)
            # Store per-request K/V caches
            # Each request gets the caches corresponding to its tokens
            req.k_caches = k_caches
            req.v_caches = v_caches
            req.cached_len = seq_len

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

        # Get past K/V caches from the first request
        # All requests in the batch share the same cache structure
        first_req = batch.requests[0]
        if first_req.k_caches is None or first_req.v_caches is None:
            # Fallback: no cached K/V available, do full forward pass
            token_ids = np.array(batch.input_ids, dtype=np.int32)
            hidden_states = ch14.forward(ch14.gpt_forward(
                token_ids,
                self.embedding_table,
                self.all_weights,
                self.final_gamma,
                self.final_beta
            ))
        else:
            # Use KV-cached decode
            token_ids = np.array(batch.input_ids, dtype=np.int32)

            result = ch14.gpt_forward_decode(
                token_ids,
                self.embedding_table,
                self.all_weights,
                self.final_gamma,
                self.final_beta,
                first_req.k_caches,
                first_req.v_caches
            )

            hidden_states = result.hidden_states
            
            # Update caches for all requests
            for req in batch.requests:
                req.k_caches = result.k_caches
                req.v_caches = result.v_caches

        # For decode, we only care about the last position
        last_hidden = hidden_states[-len(batch.requests):, :]  # Last token per request
        logits = last_hidden @ self.embedding_table.T

        # Safety: Check for NaN/Inf and clip to reasonable range
        if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
            print(f"Warning: NaN/Inf detected in decode logits, clipping values")
            logits = np.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)

        logits = np.clip(logits, -100.0, 100.0)

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