"""
Nano LLM Serving Engine - Complete Integration

Brings together all optimizations:
- Radix cache for prefix sharing
- Continuous batching for dynamic scheduling
- Chunked prefill for long contexts
- MLIR JIT execution from Chapter 14
"""

from typing import List, Optional, Dict, Any
import sys
sys.path.insert(0, '.')
import numpy as np

from python.request import Request
from python.kv_pool import KVCachePool
from python.radix_manager import RadixCacheManager
from python.chunked_prefill import ChunkedPrefillManager
from python.decode_manager import DecodeManager
from python.executor import ModelExecutor, ModelConfig
from python.continuous_batcher import ContinuousBatcher

class SamplingParams:
    """Parameters for text generation sampling"""

    def __init__(self,
                 max_tokens: int = 16,
                 temperature: float = 1.0,
                 ignore_eos: bool = False):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.ignore_eos = ignore_eos

class NanoServingEngine:
    """
    Complete LLM serving engine with all optimizations

    Features:
    - Radix cache: 40-60% cache hit rate from prefix sharing
    - Continuous batching: Dynamic request scheduling
    - Chunked prefill: Handle long contexts efficiently
    - MLIR JIT: Fast model execution
    """

    def __init__(self,
                 config: ModelConfig,
                 weights: Dict[str, np.ndarray],
                 kv_cache_pages: int = 256,
                 max_chunk_size: int = 256,
                 max_batch_size: int = 32,
                 eos_token_id: int = 0):
        """
        Initialize serving engine

        Args:
            config: Model configuration
            weights: Model weights dictionary
            kv_cache_pages: Number of KV cache pages
            max_chunk_size: Maximum tokens per prefill chunk
            max_batch_size: Maximum batch size for decode
            eos_token_id: End-of-sequence token ID
        """
        self.config = config
        self.weights = weights
        self.eos_token_id = eos_token_id

        # Initialize KV cache pool
        page_size = 16
        self.kv_pool = KVCachePool(
            num_pages=kv_cache_pages,
            page_size=page_size,
            num_layers=config.n_layer,
            num_heads=config.n_head,
            head_dim=config.head_dim
        )

        # Initialize radix cache for prefix sharing
        self.radix_mgr = RadixCacheManager(self.kv_pool)

        # Initialize schedulers
        self.prefill_mgr = ChunkedPrefillManager(
            self.kv_pool,
            max_chunk_size=max_chunk_size
        )
        self.decode_mgr = DecodeManager(max_batch_size=max_batch_size)

        # Initialize model executor (MLIR JIT)
        self.executor = ModelExecutor(config, weights, self.kv_pool)

        # Initialize continuous batcher
        self.batcher = ContinuousBatcher(
            executor=self.executor,
            radix_mgr=self.radix_mgr,
            prefill_mgr=self.prefill_mgr,
            decode_mgr=self.decode_mgr,
            eos_token_id=eos_token_id
        )

        # Statistics
        self.total_requests = 0
        self.total_tokens_generated = 0

    def generate(self,
                 prompt_tokens_list: List[List[int]],
                 sampling_params: Optional[List[SamplingParams]] = None
                ) -> List[Request]:
        """
        Main serving API - generate completions for prompts

        Args:
            prompt_tokens_list: List of tokenized prompts
            sampling_params: Optional sampling parameters per request

        Returns:
            List of completed Request objects with output_tokens
        """
        if not prompt_tokens_list:
            return []

        # Create requests
        requests = []
        for i, prompt_tokens in enumerate(prompt_tokens_list):
            params = sampling_params[i] if sampling_params else SamplingParams()

            req = Request(
                req_id=self.total_requests + i,
                prompt_tokens=prompt_tokens,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                ignore_eos=params.ignore_eos
            )
            requests.append(req)

        self.total_requests += len(requests)

        # Run continuous batching until all requests finish
        finished = self.batcher.run_until_complete(requests)

        # Update statistics
        for req in finished:
            self.total_tokens_generated += len(req.output_tokens)

        return finished

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive serving statistics

        Returns:
            Dictionary with performance metrics
        """
        radix_stats = {
            'cache_hit_rate': self.radix_mgr.cache_hit_rate,
            'cache_hits': self.radix_mgr.cache_hits,
            'cache_misses': self.radix_mgr.cache_misses,
            'total_evictions': self.radix_mgr.total_evictions,
        }

        batcher_stats = self.batcher.get_stats()

        kv_cache_stats = {
            'num_free_pages': self.kv_pool.num_free_pages,
            'num_total_pages': self.kv_pool.num_pages,
            'memory_utilization': 1.0 - (self.kv_pool.num_free_pages / self.kv_pool.num_pages),
        }

        return {
            **radix_stats,
            **batcher_stats,
            **kv_cache_stats,
            'total_requests_served': self.total_requests,
            'total_tokens_generated': self.total_tokens_generated,
        }

    def reset_stats(self):
        """Reset all statistics counters"""
        self.batcher.total_tokens_generated = 0
        self.batcher.total_steps = 0
        self.radix_mgr.cache_hits = 0
        self.radix_mgr.cache_misses = 0
        self.radix_mgr.total_evictions = 0
        self.total_requests = 0
        self.total_tokens_generated = 0

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"NanoServingEngine("
                f"requests={stats['total_requests_served']}, "
                f"tokens={stats['total_tokens_generated']}, "
                f"cache_hit_rate={stats['cache_hit_rate']:.1%}, "
                f"mem_util={stats['memory_utilization']:.1%})")