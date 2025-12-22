"""
Decode Manager - Scheduling for Token Generation

Handles batching of requests in decode phase (autoregressive generation).
"""

from typing import List, Optional
import sys
sys.path.insert(0, '.')

from python.request import Request
from python.batch import Batch


class DecodeManager:
    """
    Manages decode scheduling
    
    Decode phase generates one token at a time for each request.
    All running requests can be batched together since they each
    process exactly one token.
    """
    
    def __init__(self, max_batch_size: int = 32):
        """
        Initialize decode manager
        
        Args:
            max_batch_size: Maximum number of requests per decode batch
        """
        self.running_requests: List[Request] = []
        self.max_batch_size = max_batch_size
        
    def add_request(self, req: Request):
        """
        Add request to running pool (after prefill)
        
        Args:
            req: Request that completed prefill
        """
        self.running_requests.append(req)
        
    def remove_finished(self):
        """Remove finished requests from running pool"""
        self.running_requests = [
            req for req in self.running_requests 
            if not req.is_finished
        ]
        
    def schedule(self) -> Optional[Batch]:
        """
        Build decode batch from running requests
        
        All running requests decode together (up to max_batch_size).
        Each request generates exactly one token.
        
        Returns:
            Batch of running requests, or None if no running requests
        """
        if not self.running_requests:
            return None
            
        # Remove any finished requests first
        self.remove_finished()
        
        if not self.running_requests:
            return None
            
        # Select up to max_batch_size requests
        selected = self.running_requests[:self.max_batch_size]
        
        return Batch.from_decode(selected)
    
    def num_running(self) -> int:
        """Get number of running requests"""
        return len(self.running_requests)
    
    def has_running(self) -> bool:
        """Check if there are running requests"""
        return len(self.running_requests) > 0
