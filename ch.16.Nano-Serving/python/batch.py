"""
Batch representation for processing multiple requests

Supports both prefill (parallel prompt processing) and decode (sequential generation).
"""

from dataclasses import dataclass
from typing import List, Tuple
import sys
sys.path.insert(0, '.')
import numpy as np

from python.request import Request


@dataclass
class Batch:
    """Batch of requests for processing"""
    
    requests: List[Request]
    is_prefill: bool
    input_ids: np.ndarray      # [total_tokens] for prefill or [batch_size] for decode
    positions: np.ndarray      # Position IDs for RoPE
    out_loc: np.ndarray        # Output locations in KV cache
    
    @property
    def size(self) -> int:
        """Number of requests in batch"""
        return len(self.requests)
    
    @property
    def is_decode(self) -> bool:
        """Whether this is a decode batch"""
        return not self.is_prefill
    
    @classmethod
    def from_prefill(cls, requests: List[Request]) -> "Batch":
        """
        Create prefill batch from requests
        
        Prefill processes all prompt tokens (from cached_len to end).
        Concatenates all tokens into single sequence.
        """
        if not requests:
            raise ValueError("Cannot create batch from empty request list")
        
        # Collect all tokens to process
        all_tokens = []
        all_positions = []
        out_locs = []
        
        for req in requests:
            # Get uncached portion of prompt
            start_idx = req.cached_len
            tokens = req.prompt_tokens[start_idx:]
            
            all_tokens.extend(tokens)
            
            # Position IDs start from cached_len
            positions = list(range(start_idx, start_idx + len(tokens)))
            all_positions.extend(positions)
            
            # Output locations: where to write in KV cache
            # For prefill, write to positions [cached_len, cached_len + extend_len)
            out_locs.extend(range(req.cached_len, req.cached_len + len(tokens)))
        
        return cls(
            requests=requests,
            is_prefill=True,
            input_ids=np.array(all_tokens, dtype=np.int32),
            positions=np.array(all_positions, dtype=np.int32),
            out_loc=np.array(out_locs, dtype=np.int32)
        )
    
    @classmethod
    def from_decode(cls, requests: List[Request]) -> "Batch":
        """
        Create decode batch from requests
        
        Decode processes only the last generated token.
        Each request contributes exactly 1 token.
        """
        if not requests:
            raise ValueError("Cannot create batch from empty request list")
        
        # Each request contributes its last output token
        tokens = []
        positions = []
        out_locs = []
        
        for req in requests:
            if not req.output_tokens:
                raise ValueError(f"Request {req.req_id} has no output tokens for decode")
            
            # Last generated token
            tokens.append(req.output_tokens[-1])
            
            # Position is current total length
            positions.append(req.total_len)
            
            # Write to next position in KV cache
            out_locs.append(req.total_len)
        
        return cls(
            requests=requests,
            is_prefill=False,
            input_ids=np.array(tokens, dtype=np.int32),
            positions=np.array(positions, dtype=np.int32),
            out_loc=np.array(out_locs, dtype=np.int32)
        )
    
    @classmethod
    def from_chunks(cls, chunks: List[Tuple[Request, List[int]]]) -> "Batch":
        """
        Create prefill batch from prompt chunks
        
        Used for chunked prefill where long prompts are split.
        
        Args:
            chunks: List of (request, chunk_tokens) pairs
        """
        if not chunks:
            raise ValueError("Cannot create batch from empty chunks")
        
        requests = [req for req, _ in chunks]
        all_tokens = []
        all_positions = []
        out_locs = []
        
        for req, chunk_tokens in chunks:
            all_tokens.extend(chunk_tokens)
            
            # Positions start from current cached length
            start_pos = req.cached_len
            positions = list(range(start_pos, start_pos + len(chunk_tokens)))
            all_positions.extend(positions)
            
            # Output locations
            out_locs.extend(range(req.cached_len, req.cached_len + len(chunk_tokens)))
        
        return cls(
            requests=requests,
            is_prefill=True,
            input_ids=np.array(all_tokens, dtype=np.int32),
            positions=np.array(all_positions, dtype=np.int32),
            out_loc=np.array(out_locs, dtype=np.int32)
        )
    
    def __repr__(self) -> str:
        phase = "prefill" if self.is_prefill else "decode"
        return (f"Batch({phase}, size={self.size}, "
                f"tokens={len(self.input_ids)})")
