"""
Continuous Batching Engine

Implements the main continuous batching loop:
- Dynamically add/remove requests
- Interleave prefill and decode
- Maximize throughput with radix cache
"""

from typing import List, Optional
import sys
sys.path.insert(0, '.')
import numpy as np

from python.request import Request
from python.request_pool import RequestPool
from python.executor import ModelExecutor
from python.radix_manager import RadixCacheManager
from python.chunked_prefill import ChunkedPrefillManager
from python.decode_manager import DecodeManager


def sample_token(logits: np.ndarray, temperature: float = 1.0) -> int:
    """Sample next token from logits
    
    Args:
        logits: Logits for next token [vocab_size]
        temperature: Sampling temperature (1.0 = neutral, <1 = more conservative, >1 = more random)
        
    Returns:
        Sampled token ID
    """
    if temperature == 0.0:
        # Greedy sampling
        return int(np.argmax(logits))
    
    # Temperature scaling
    logits = logits / temperature
    
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    probs = exp_logits / np.sum(exp_logits)
    
    # Sample
    return int(np.random.choice(len(probs), p=probs))


class ContinuousBatcher:
    """Implements continuous batching main loop
    
    Key Innovation: Dynamic batching
    - Requests finish at different times
    - Immediately add new requests when space available
    - Much higher throughput than static batching
    """
    
    def __init__(self,
                 executor: ModelExecutor,
                 radix_mgr: RadixCacheManager,
                 prefill_mgr: ChunkedPrefillManager,
                 decode_mgr: DecodeManager,
                 eos_token_id: int = 0):
        self.executor = executor
        self.radix_mgr = radix_mgr
        self.prefill_mgr = prefill_mgr
        self.decode_mgr = decode_mgr
        self.eos_token_id = eos_token_id
        self.request_pool = RequestPool()
        
        # Statistics
        self.total_tokens_generated = 0
        self.total_steps = 0
        
    def step(self) -> int:
        """
        Execute one batching iteration
        
        Returns:
            Number of tokens generated this step
        """
        self.total_steps += 1
        tokens_generated = 0
        
        # Step 1: Check for finished requests
        finished = []
        for req in self.request_pool.running:
            if self._is_finished(req):
                finished.append(req)
                # Free KV cache pages
                if req.kv_pages:
                    self.radix_mgr.kv_pool.free(req.kv_pages)
                    req.kv_pages = []
        
        if finished:
            self.request_pool.move_to_finished(finished)
        
        # Step 2: Try to schedule prefill (if space available)
        # Move waiting requests to prefill manager
        # TODO: Integrate radix cache into prefill allocation
        for req in list(self.request_pool.waiting):
            self.prefill_mgr.add_request(req)
            self.request_pool.waiting.remove(req)
        
        prefill_batch = self.prefill_mgr.schedule()
        if prefill_batch:
            # Execute prefill
            logits = self.executor.execute_prefill(prefill_batch)
            
            # Process output (sample first token)
            for i, req in enumerate(prefill_batch.requests):
                # Get logits for last token of this request
                # In prefill, each request may have different lengths
                next_token = sample_token(logits[i], req.temperature)
                req.output_tokens.append(next_token)
                tokens_generated += 1
                
                # Move to running (now in decode phase)
                if req not in self.request_pool.running:
                    self.request_pool.move_to_running([req])
                    # Add to decode manager
                    self.decode_mgr.add_request(req)
        
        # Step 3: Schedule decode batch
        # Remove finished requests from decode manager
        self.decode_mgr.remove_finished()
        
        decode_batch = self.decode_mgr.schedule()
        if decode_batch:
            # Execute decode
            logits = self.executor.execute_decode(decode_batch)
            
            # Process output
            for i, req in enumerate(decode_batch.requests):
                next_token = sample_token(logits[i], req.temperature)
                req.output_tokens.append(next_token)
                tokens_generated += 1
        
        self.total_tokens_generated += tokens_generated
        return tokens_generated
    
    def _is_finished(self, req: Request) -> bool:
        """Check if request should finish
        
        Finish conditions:
        1. Reached max_tokens
        2. Generated EOS token (if not ignoring EOS)
        """
        if len(req.output_tokens) >= req.max_tokens:
            return True
        
        if req.output_tokens and not req.ignore_eos:
            if req.output_tokens[-1] == self.eos_token_id:
                return True
        
        return False
    
    def run_until_complete(self, requests: List[Request], max_steps: int = 10000) -> List[Request]:
        """Run continuous batching until all requests finish
        
        Args:
            requests: Initial batch of requests
            max_steps: Maximum steps to prevent infinite loops
            
        Returns:
            List of finished requests
        """
        self.request_pool.add_requests(requests)
        
        step_count = 0
        while self.request_pool.has_pending() and step_count < max_steps:
            self.step()
            step_count += 1
        
        if step_count >= max_steps:
            print(f"Warning: Reached max_steps ({max_steps})")
        
        return self.request_pool.finished
    
    def get_stats(self) -> dict:
        """Get batching statistics"""
        return {
            'total_steps': self.total_steps,
            'total_tokens_generated': self.total_tokens_generated,
            'avg_tokens_per_step': self.total_tokens_generated / max(1, self.total_steps),
            'waiting': self.request_pool.get_waiting_count(),
            'running': self.request_pool.get_running_count(),
            'finished': self.request_pool.get_finished_count(),
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"ContinuousBatcher(steps={stats['total_steps']}, "
                f"tokens={stats['total_tokens_generated']}, "
                f"pool={self.request_pool})")
