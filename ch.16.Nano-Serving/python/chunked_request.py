"""
Chunked Request - Handling Long Context Prompts

Splits long prompts into chunks to:
- Prevent OOM on long contexts
- Allow interleaving with decode requests
- Better memory utilization
"""

from typing import List, Optional
import sys
sys.path.insert(0, '.')

from python.request import Request

class ChunkedRequest:
    """
    Wrapper for request with chunking state

    Tracks progress through a long prompt, processing it in chunks
    instead of all at once. This enables:
    - Processing prompts longer than max_prefill_tokens
    - Interleaving chunk processing with decode steps
    - Gradual KV cache allocation
    """

    def __init__(self, request: Request, chunk_size: int = 512):
        """
        Initialize chunked request

        Args:
            request: The underlying request to chunk
            chunk_size: Maximum tokens per chunk
        """
        self.request = request
        self.chunk_size = chunk_size
        self.current_chunk_idx = 0

        # Track how many tokens we've processed
        # Start from request.cached_len in case of partial prefill
        self.tokens_processed = request.cached_len

    @property
    def has_more_chunks(self) -> bool:
        """Check if there are more chunks to process"""
        return self.tokens_processed < len(self.request.prompt_tokens)

    @property
    def num_chunks_remaining(self) -> int:
        """Get number of chunks remaining"""
        remaining_tokens = len(self.request.prompt_tokens) - self.tokens_processed
        return (remaining_tokens + self.chunk_size - 1) // self.chunk_size

    @property
    def total_chunks(self) -> int:
        """Get total number of chunks for this request"""
        total_tokens = len(self.request.prompt_tokens)
        return (total_tokens + self.chunk_size - 1) // self.chunk_size

    @property
    def progress(self) -> float:
        """Get progress as fraction [0.0, 1.0]"""
        if len(self.request.prompt_tokens) == 0:
            return 1.0
        return self.tokens_processed / len(self.request.prompt_tokens)

    def get_next_chunk(self) -> List[int]:
        """
        Get next chunk of tokens

        Returns:
            List of token IDs for the next chunk
        """
        if not self.has_more_chunks:
            return []

        start = self.tokens_processed
        end = min(start + self.chunk_size, len(self.request.prompt_tokens))

        chunk = self.request.prompt_tokens[start:end]

        # Update state
        self.tokens_processed += len(chunk)
        self.current_chunk_idx += 1
        self.request.cached_len = self.tokens_processed

        return chunk

    def peek_next_chunk_size(self) -> int:
        """
        Peek at size of next chunk without consuming it

        Returns:
            Number of tokens in next chunk, or 0 if done
        """
        if not self.has_more_chunks:
            return 0

        start = self.tokens_processed
        end = min(start + self.chunk_size, len(self.request.prompt_tokens))
        return end - start

    def reset(self):
        """Reset chunking state (for testing)"""
        self.current_chunk_idx = 0
        self.tokens_processed = 0
        self.request.cached_len = 0

    def __repr__(self) -> str:
        return (f"ChunkedRequest(req_id={self.request.req_id}, "
                f"chunk={self.current_chunk_idx}/{self.total_chunks}, "
                f"progress={self.progress:.1%})")