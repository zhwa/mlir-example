"""
Request Pool for Continuous Batching

Manages the lifecycle of requests:
- waiting: Queued, not yet started
- running: Currently being processed
- finished: Completed generation
"""

from typing import List
import sys
sys.path.insert(0, '.')

from python.request import Request


class RequestPool:
    """Manages lifecycle of requests in continuous batching"""
    
    def __init__(self):
        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.finished: List[Request] = []
        
    def add_requests(self, reqs: List[Request]):
        """Add new requests to waiting queue"""
        self.waiting.extend(reqs)
        
    def move_to_running(self, reqs: List[Request]):
        """Move requests from waiting to running"""
        for req in reqs:
            if req in self.waiting:
                self.waiting.remove(req)
            if req not in self.running:
                self.running.append(req)
            
    def move_to_finished(self, reqs: List[Request]):
        """Move requests from running to finished"""
        for req in reqs:
            if req in self.running:
                self.running.remove(req)
            if req not in self.finished:
                self.finished.append(req)
                req.is_finished = True
                
    def get_waiting_count(self) -> int:
        """Number of waiting requests"""
        return len(self.waiting)
    
    def get_running_count(self) -> int:
        """Number of running requests"""
        return len(self.running)
    
    def get_finished_count(self) -> int:
        """Number of finished requests"""
        return len(self.finished)
    
    def has_pending(self) -> bool:
        """Check if there are any waiting or running requests"""
        return len(self.waiting) > 0 or len(self.running) > 0
    
    def __repr__(self) -> str:
        return (f"RequestPool(waiting={len(self.waiting)}, "
                f"running={len(self.running)}, "
                f"finished={len(self.finished)})")
