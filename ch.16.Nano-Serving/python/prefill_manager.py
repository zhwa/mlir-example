"""
Prefill Manager - Scheduling for Prompt Processing

Handles the selection and batching of requests for prefill phase.
"""

from typing import List, Optional
import sys
sys.path.insert(0, '.')

from python.request import Request
from python.batch import Batch
from python.kv_pool import KVCachePool

class PrefillManager:
    """
    Manages prefill scheduling

    Prefill phase processes all prompt tokens in parallel to:
    1. Generate KV cache for the prompt
    2. Produce the first output token

    Batching strategy: FCFS with token budget
    """

    def __init__(self, kv_pool: KVCachePool, max_prefill_tokens: int = 2048):
        """
        Initialize prefill manager

        Args:
            kv_pool: KV cache pool for page allocation
            max_prefill_tokens: Maximum tokens to process in one prefill batch
        """
        self.kv_pool = kv_pool
        self.max_prefill_tokens = max_prefill_tokens

    def schedule(self, waiting_requests: List[Request]) -> Optional[Batch]:
        """
        Select requests for prefill batch

        Strategy: First-Come-First-Served (FCFS) until token budget exhausted

        Args:
            waiting_requests: List of requests waiting for prefill

        Returns:
            Batch of selected requests, or None if no requests can fit
        """
        if not waiting_requests:
            return None

        selected = []
        total_tokens = 0
        pages_allocated = 0  # Track pages allocated in this batch

        for req in waiting_requests:
            prompt_len = len(req.prompt_tokens)

            # Check if adding this request exceeds budget
            if total_tokens + prompt_len <= self.max_prefill_tokens:
                # Check if KV pool has enough pages
                num_pages_needed = (prompt_len + self.kv_pool.page_size - 1) // self.kv_pool.page_size

                # Check remaining free pages after previous allocations
                if self.kv_pool.num_free_pages - pages_allocated >= num_pages_needed:
                    selected.append(req)
                    total_tokens += prompt_len
                    pages_allocated += num_pages_needed
                else:
                    # Out of memory - stop selecting
                    break
            else:
                # Token budget exceeded
                break

        if not selected:
            return None

        # Allocate KV cache pages for selected requests
        for req in selected:
            if req.kv_pages is None or len(req.kv_pages) == 0:  # Not already allocated
                req.kv_pages = self.kv_pool.allocate(len(req.prompt_tokens))

        return Batch.from_prefill(selected)

    def can_schedule(self, waiting_requests: List[Request]) -> bool:
        """
        Check if any waiting request can be scheduled

        Args:
            waiting_requests: Requests waiting for prefill

        Returns:
            True if at least one request can fit in budget and memory
        """
        if not waiting_requests:
            return False

        # Check if first request fits
        first_req = waiting_requests[0]
        prompt_len = len(first_req.prompt_tokens)

        if prompt_len > self.max_prefill_tokens:
            # Request too large (should use chunked prefill - Phase 3)
            return False

        # Check memory availability
        num_pages_needed = (prompt_len + self.kv_pool.page_size - 1) // self.kv_pool.page_size
        return self.kv_pool.num_free_pages >= num_pages_needed