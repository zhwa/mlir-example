"""
Radix Cache Manager - High-Level Interface

Provides a high-level API for radix cache with automatic eviction,
cache hit tracking, and integration with the KV pool.
"""

from typing import List, Tuple
import sys
sys.path.insert(0, '.')

from python.radix_cache import RadixCache
from python.kv_pool import KVCachePool


class RadixCacheManager:
    """
    High-level interface for radix cache
    
    Manages prefix matching, automatic eviction, and cache statistics.
    Main entry point for using radix cache in serving engine.
    """
    
    def __init__(self, kv_pool: KVCachePool):
        """
        Initialize radix cache manager
        
        Args:
            kv_pool: KV cache pool for page management
        """
        self.cache = RadixCache(kv_pool)
        self.kv_pool = kv_pool
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_evictions = 0
    
    def get_or_allocate(self, tokens: List[int]) -> Tuple[int, List[int]]:
        """
        Get KV cache for tokens, reusing prefix if possible
        
        This is the main API for cache lookup/allocation:
        1. Find longest matching prefix in cache
        2. Allocate pages only for uncached suffix
        3. Insert full sequence into cache
        4. Evict LRU entries if out of memory
        
        Args:
            tokens: Token sequence needing KV cache
            
        Returns:
            Tuple of (cached_length, new_page_indices)
            - cached_length: Number of tokens found in cache
            - new_page_indices: Page indices allocated for uncached tokens
        """
        if not tokens:
            return 0, []
        
        # Step 1: Find matching prefix
        cached_len, last_node = self.cache.match_prefix(tokens)
        
        # Update statistics
        if cached_len > 0:
            self.cache_hits += cached_len
        
        uncached_len = len(tokens) - cached_len
        
        if uncached_len == 0:
            # Everything cached!
            return cached_len, []
        
        # Step 2: Allocate pages for uncached suffix
        self.cache_misses += uncached_len
        
        # Simplified model: 1 page per token
        # In real systems, multiple tokens share a page, but for radix cache
        # demonstration we use 1:1 mapping
        num_pages_needed = uncached_len
        
        new_pages = []
        try:
            # Allocate pages one by one to get 1 page per token
            for _ in range(uncached_len):
                page = self.kv_pool.allocate(1)  # Allocate 1 token -> gets 1 page
                new_pages.extend(page)
        except RuntimeError:
            # Out of memory - evict until we have space
            num_evicted = self.cache.evict_until_available(num_pages_needed)
            self.total_evictions += num_evicted
            
            try:
                # Try again after eviction
                for _ in range(uncached_len):
                    page = self.kv_pool.allocate(1)
                    new_pages.extend(page)
            except RuntimeError:
                # Still can't allocate - request too large
                raise MemoryError(
                    f"Cannot allocate {num_pages_needed} pages even after eviction. "
                    f"Free pages: {self.kv_pool.num_free_pages}"
                )
        
        # Step 3: Insert full sequence into cache
        # Build page list: one page per token
        # For cached tokens, we don't store duplicate page info (they're already in tree)
        # For new tokens, we assign pages
        all_pages = []
        
        # For cached prefix, get one page per token
        if cached_len > 0:
            node = self.cache.root
            for token in tokens[:cached_len]:
                child = node.get_child(token)
                if child and child.kv_pages:
                    all_pages.append(child.kv_pages[0])
                node = child
        
        # Add new pages for uncached suffix
        all_pages.extend(new_pages)
        
        # Insert complete sequence
        self.cache.insert(tokens, all_pages)
        
        return cached_len, new_pages
    
    def insert_sequence(self, tokens: List[int], kv_pages: List[int]):
        """
        Explicitly insert a sequence into cache
        
        Use this when you already have allocated pages and want to
        add them to the cache for future reuse.
        
        Args:
            tokens: Token sequence
            kv_pages: KV page indices (same length as tokens)
        """
        self.cache.insert(tokens, kv_pages)
    
    def match_prefix(self, tokens: List[int]) -> int:
        """
        Check how much of a sequence is cached
        
        Args:
            tokens: Token sequence to check
            
        Returns:
            Number of tokens that match cached prefixes
        """
        matched_len, _ = self.cache.match_prefix(tokens)
        return matched_len
    
    def evict_lru(self) -> int:
        """
        Manually evict least-recently-used entry
        
        Returns:
            Number of pages freed
        """
        freed = self.cache.evict_lru_leaf()
        if freed > 0:
            self.total_evictions += freed
        return freed
    
    @property
    def cache_hit_rate(self) -> float:
        """
        Get cache hit rate
        
        Returns:
            Fraction of tokens found in cache [0.0, 1.0]
        """
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def num_cached_sequences(self) -> int:
        """Get number of sequences in cache"""
        return self.cache.num_nodes
    
    def clear_stats(self):
        """Reset statistics counters"""
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_evictions = 0
    
    def clear_cache(self):
        """Clear entire cache"""
        self.cache.clear()
        self.clear_stats()
    
    def __repr__(self) -> str:
        return (f"RadixCacheManager("
                f"hit_rate={self.cache_hit_rate:.1%}, "
                f"sequences={self.num_cached_sequences}, "
                f"hits={self.cache_hits}, "
                f"misses={self.cache_misses})")
