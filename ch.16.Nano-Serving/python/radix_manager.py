"""
Radix Cache Manager - High-Level Interface

Provides a high-level API for radix cache with automatic eviction,
cache hit tracking, and integration with the KV pool.

Uses C++ RadixCache implementation directly from ch16 module.
"""

from typing import List, Tuple
import sys
import os

sys.path.insert(0, '.')

def import_cpp_module(module_name: str, chapter_path: str):
    """Helper to import C++ modules from build directories"""
    build_paths = [
        f'../build/x64-release/{chapter_path}',
        f'../build/x64-debug/{chapter_path}',
        f'build/x64-release/{chapter_path}',
        f'build/x64-debug/{chapter_path}',
        f'../../build/x64-release/{chapter_path}',
        f'../../build/x64-debug/{chapter_path}',
    ]
    
    for path in build_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            try:
                return __import__(module_name)
            except ImportError:
                sys.path.pop(0)
    
    # Final attempt without path (if already in sys.path)
    return __import__(module_name)

# Import C++ modules
ch16 = import_cpp_module('ch16', 'ch.16.Nano-Serving')
ch14 = import_cpp_module('ch14', 'ch.14.GPT-Optimized')
KVCachePool = ch14.KVCachePool

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
        self.cache = ch16.RadixCache()  # C++ implementation
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
            pages_freed = 0
            while pages_freed < num_pages_needed:
                try:
                    leaf, path = self.cache.find_lru_leaf()
                    freed_pages = self.cache.evict_leaf(leaf, path)
                    if freed_pages:
                        self.kv_pool.free(freed_pages)
                        pages_freed += len(freed_pages)
                        self.total_evictions += 1
                    else:
                        break
                except:
                    break

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
        # For new tokens, we assign pages
        all_pages = []

        # For cached prefix, get pages using C++ API
        if cached_len > 0:
            prefix_pages = self.cache.get_pages_for_prefix(tokens[:cached_len])
            all_pages.extend(prefix_pages)

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
        try:
            leaf, path = self.cache.find_lru_leaf()
            freed_pages = self.cache.evict_leaf(leaf, path)
            if freed_pages:
                self.kv_pool.free(freed_pages)
                self.total_evictions += 1
                return len(freed_pages)
        except:
            pass
        return 0

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
        return self.cache.get_num_nodes()

    def clear_stats(self):
        """Reset statistics counters"""
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_evictions = 0

    def __repr__(self) -> str:
        return (f"RadixCacheManager("
                f"hit_rate={self.cache_hit_rate:.1%}, "
                f"sequences={self.num_cached_sequences}, "
                f"hits={self.cache_hits}, "
                f"misses={self.cache_misses})")