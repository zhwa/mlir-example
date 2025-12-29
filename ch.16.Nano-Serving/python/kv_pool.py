"""
Python wrapper for C++ KV Cache Pool
"""

import sys
import os
from typing import List, Tuple
import numpy as np

# Try to import the C++ module
try:
    # Look for the built module in build directory
    build_paths = [
        '../build/x64-release/ch.16.Nano-Serving',
        '../build/x64-debug/ch.16.Nano-Serving',
        'build/x64-release/ch.16.Nano-Serving',
        'build/x64-debug/ch.16.Nano-Serving',
        '../../build/x64-release/ch.16.Nano-Serving',
        '../../build/x64-debug/ch.16.Nano-Serving',
    ]

    ch16 = None
    for path in build_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            try:
                import ch16
                break
            except ImportError:
                sys.path.pop(0)

    if ch16 is None:
        # Try direct import (if in path)
        import ch16

except ImportError as e:
    print(f"Error: Could not import ch16 module: {e}")
    print("Please build Chapter 16 first:")
    print("  cmake --build build/x64-release --target ch16")
    raise

class KVCachePool:
    """
    Python wrapper for C++ KV cache pool

    Manages page-based KV cache allocation for transformer inference.
    """

    def __init__(self, num_pages: int, page_size: int, 
                 num_layers: int, num_heads: int, head_dim: int):
        """
        Initialize KV cache pool

        Args:
            num_pages: Total number of pages in pool
            page_size: Number of tokens per page
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
        """
        self._pool = ch16.KVCachePool(num_pages, page_size, 
                                       num_layers, num_heads, head_dim)
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

    def pages_needed(self, num_tokens: int) -> int:
        """Calculate pages needed for num_tokens
        
        With page_size=16:
            15 tokens → 1 page
            16 tokens → 1 page  
            17 tokens → 2 pages
        """
        return (num_tokens + self.page_size - 1) // self.page_size

    def can_allocate(self, num_tokens: int) -> bool:
        """Check if allocation would succeed"""
        pages = self.pages_needed(num_tokens)
        return self.num_free_pages >= pages

    def allocate(self, num_tokens: int) -> List[int]:
        """
        Allocate pages for num_tokens

        Args:
            num_tokens: Number of tokens requiring cache

        Returns:
            List of allocated page indices

        Raises:
            RuntimeError: If insufficient pages available
        """
        return self._pool.allocate(num_tokens)

    def free(self, pages: List[int]):
        """
        Free pages back to pool

        Args:
            pages: List of page indices to free
        """
        self._pool.free(pages)

    def store_kv(self, k: np.ndarray, v: np.ndarray, 
                 page_indices: List[int], layer_id: int):
        """
        Store K/V tensors at specified pages

        Args:
            k: Key tensor [num_tokens, num_heads, head_dim]
            v: Value tensor [num_tokens, num_heads, head_dim]
            page_indices: Pages to store in
            layer_id: Layer index
        """
        # Ensure contiguous and correct dtype
        k = np.ascontiguousarray(k, dtype=np.float32)
        v = np.ascontiguousarray(v, dtype=np.float32)

        self._pool.store_kv(k, v, page_indices, layer_id)

    def get_layer_cache(self, layer_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get K/V cache arrays for a layer

        Args:
            layer_id: Layer index

        Returns:
            Tuple of (k_cache, v_cache) arrays
        """
        return self._pool.get_layer_cache(layer_id)

    @property
    def num_free_pages(self) -> int:
        """Get number of free pages"""
        return self._pool.get_num_free_pages()

    def __repr__(self) -> str:
        return (f"KVCachePool(pages={self.num_pages}, "
                f"page_size={self.page_size}, "
                f"free={self.num_free_pages})")