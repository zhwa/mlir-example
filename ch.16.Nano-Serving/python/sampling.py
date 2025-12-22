"""
Sampling parameters for generation
"""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Parameters controlling text generation"""
    
    temperature: float = 1.0
    top_k: int = 50
    max_tokens: int = 128
    ignore_eos: bool = False
    
    def __post_init__(self):
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {self.temperature}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
