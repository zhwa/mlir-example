"""
Chunked Prefill Manager - Scheduling for Long Contexts

Handles chunked processing of long prompts, allowing interleaving
with decode requests for better resource utilization.
"""

from typing import List, Optional, Tuple
import sys
sys.path.insert(0, '.')

from python.request import Request
from python.batch import Batch
from python.kv_pool import KVCachePool
from python.chunked_request import ChunkedRequest

class ChunkedPrefillManager:
    """
    Manages chunked prefill scheduling

    Key benefits:
    - Process prompts longer than max_prefill_tokens
    - Interleave chunk processing with decode
    - Better memory utilization (gradual allocation)
    - Fairer resource sharing between requests
    """

    def __init__(self, kv_pool: KVCachePool, max_chunk_size: int = 512,
                 max_batch_tokens: int = 2048):
        """
        Initialize chunked prefill manager

        Args:
            kv_pool: KV cache pool for page allocation
            max_chunk_size: Maximum tokens per chunk
            max_batch_tokens: Maximum tokens per batch (across all chunks)
        """
        self.kv_pool = kv_pool
        self.max_chunk_size = max_chunk_size
        self.max_batch_tokens = max_batch_tokens

        # Active chunked requests (still have chunks to process)
        self.chunked_requests: List[ChunkedRequest] = []

    def add_request(self, req: Request):
        """
        Add new request for chunked prefill

        Args:
            req: Request to process in chunks
        """
        chunked = ChunkedRequest(req, self.max_chunk_size)
        self.chunked_requests.append(chunked)

    def schedule(self) -> Optional[Batch]:
        """
        Schedule next chunk batch

        Selects chunks from multiple requests up to max_batch_tokens.
        Uses round-robin to ensure fairness.

        Returns:
            Batch of chunks, or None if no chunks available
        """
        if not self.chunked_requests:
            return None

        # Remove completed requests
        self.chunked_requests = [
            cr for cr in self.chunked_requests 
            if cr.has_more_chunks
        ]

        if not self.chunked_requests:
            return None

        selected_chunks: List[Tuple[Request, List[int]]] = []
        total_tokens = 0
        pages_allocated = 0  # Track pages allocated in this batch

        # Round-robin through requests, taking one chunk from each
        for chunked_req in self.chunked_requests:
            if not chunked_req.has_more_chunks:
                continue

            # Check if next chunk fits in batch
            chunk_size = chunked_req.peek_next_chunk_size()

            if total_tokens + chunk_size > self.max_batch_tokens:
                continue  # Skip this request, try next

            # Check if we have enough pages
            num_pages_needed = (chunk_size + self.kv_pool.page_size - 1) // self.kv_pool.page_size

            if self.kv_pool.num_free_pages - pages_allocated < num_pages_needed:
                continue  # Out of memory, skip

            # Take the chunk
            chunk_tokens = chunked_req.get_next_chunk()
            selected_chunks.append((chunked_req.request, chunk_tokens))
            total_tokens += len(chunk_tokens)
            pages_allocated += num_pages_needed

        if not selected_chunks:
            return None

        # Allocate KV cache pages for chunks
        for req, chunk_tokens in selected_chunks:
            pages = self.kv_pool.allocate(len(chunk_tokens))

            # Add pages to request (append, not replace)
            if req.kv_pages is None or len(req.kv_pages) == 0:
                req.kv_pages = pages
            else:
                req.kv_pages.extend(pages)

        return Batch.from_chunks(selected_chunks)

    def can_schedule(self) -> bool:
        """
        Check if any chunks can be scheduled

        Returns:
            True if at least one chunk can fit
        """
        for chunked_req in self.chunked_requests:
            if not chunked_req.has_more_chunks:
                continue

            chunk_size = chunked_req.peek_next_chunk_size()

            if chunk_size <= self.max_batch_tokens:
                # Check memory
                num_pages_needed = (chunk_size + self.kv_pool.page_size - 1) // self.kv_pool.page_size
                if self.kv_pool.num_free_pages >= num_pages_needed:
                    return True

        return False

    def num_active_requests(self) -> int:
        """Get number of active chunked requests"""
        return len([cr for cr in self.chunked_requests if cr.has_more_chunks])

    def __repr__(self) -> str:
        active = self.num_active_requests()
        return f"ChunkedPrefillManager(active={active}, chunk_size={self.max_chunk_size})"