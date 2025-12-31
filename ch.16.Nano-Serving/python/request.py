"""
Request representation for LLM serving

A Request represents a single generation task with its state.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class Request:
    """Represents a single generation request"""

    req_id: int
    prompt_tokens: List[int]
    max_tokens: int
    temperature: float = 1.0
    ignore_eos: bool = False

    # State tracking
    cached_len: int = 0              # Tokens already in KV cache
    output_tokens: List[int] = field(default_factory=list)
    is_finished: bool = False

    # KV cache management (Phase 1: Page-based allocation)
    kv_pages: List[int] = field(default_factory=list)  # Physical pages allocated
    
    # KV cache data (Chapter 16: Actual K/V tensors per layer)
    k_caches: Optional[List[np.ndarray]] = None  # List of K tensors, one per layer
    v_caches: Optional[List[np.ndarray]] = None  # List of V tensors, one per layer

    @property
    def total_len(self) -> int:
        """Total tokens processed (prompt + output)"""
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def extend_len(self) -> int:
        """Tokens to process in next step"""
        if len(self.output_tokens) == 0:
            # Prefill: process from cached_len to end of prompt
            return len(self.prompt_tokens) - self.cached_len
        else:
            # Decode: only process last generated token
            return 1

    @property
    def device_len(self) -> int:
        """Total tokens in KV cache (cached + new)"""
        return self.cached_len + self.extend_len

    def __repr__(self) -> str:
        return (f"Request(id={self.req_id}, "
                f"prompt_len={len(self.prompt_tokens)}, "
                f"output_len={len(self.output_tokens)}, "
                f"cached={self.cached_len})")