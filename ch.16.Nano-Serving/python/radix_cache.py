"""
Radix Cache - Prefix Tree for KV Cache Sharing

Implements a radix tree (prefix tree) to automatically detect and reuse
shared prefixes across requests, enabling efficient KV cache sharing.
"""

from typing import List, Tuple, Optional
import time
import sys
sys.path.insert(0, '.')

from python.radix_node import RadixNode
from python.kv_pool import KVCachePool

class RadixCache:
    """
    Radix tree-based KV cache manager

    Automatically detects shared prefixes and reuses KV cache pages.
    Example:
        Request 1: [1, 2, 3, 4, 5]
        Request 2: [1, 2, 3, 6, 7]
        → [1, 2, 3] is shared, only [4, 5] and [6, 7] need new pages
    """

    def __init__(self, kv_pool: KVCachePool):
        """
        Initialize radix cache

        Args:
            kv_pool: KV cache pool for page management
        """
        self.root = RadixNode(token=None)
        self.kv_pool = kv_pool

        # Statistics
        self.num_nodes = 0
        self.total_tokens_cached = 0

    def match_prefix(self, tokens: List[int]) -> Tuple[int, RadixNode]:
        """
        Find longest matching prefix in tree

        Args:
            tokens: Token sequence to match

        Returns:
            Tuple of (match_length, last_matched_node)
        """
        node = self.root
        matched_len = 0

        for i, token in enumerate(tokens):
            child = node.get_child(token)
            if child is not None:
                node = child
                matched_len = i + 1
                node.update_access_time()
            else:
                break

        return matched_len, node

    def insert(self, tokens: List[int], kv_pages: List[int]) -> RadixNode:
        """
        Insert token sequence into tree

        Args:
            tokens: Token sequence to insert
            kv_pages: KV pages for each token (same length as tokens)

        Returns:
            Leaf node representing this sequence
        """
        if len(tokens) != len(kv_pages):
            raise ValueError(f"Tokens and pages must have same length: {len(tokens)} != {len(kv_pages)}")

        node = self.root

        for i, token in enumerate(tokens):
            child = node.get_child(token)

            if child is None:
                # Create new node
                child = node.add_child(token)
                child.kv_pages = [kv_pages[i]]
                self.num_nodes += 1
                self.total_tokens_cached += 1
            else:
                # Reuse existing node
                child.increment_ref()

            node = child

        node.update_access_time()
        return node

    def get_pages_for_prefix(self, tokens: List[int]) -> List[int]:
        """
        Get KV pages for a prefix

        Args:
            tokens: Token sequence

        Returns:
            List of KV page indices for the prefix
        """
        node = self.root
        pages = []

        for token in tokens:
            child = node.get_child(token)
            if child is None:
                break

            pages.extend(child.kv_pages)
            node = child

        return pages

    def find_lru_leaf(self) -> Tuple[Optional[RadixNode], List[RadixNode]]:
        """
        Find least-recently-used leaf node

        Returns:
            Tuple of (lru_leaf, path_to_leaf)
        """
        def _find_lru(node: RadixNode, path: List[RadixNode]) -> Tuple[Optional[RadixNode], List[RadixNode], float]:
            """Recursive helper"""
            if node.is_leaf:
                return node, path, node.last_access_time

            lru_leaf = None
            lru_path = None
            lru_time = float('inf')

            for child in node.children.values():
                leaf, leaf_path, access_time = _find_lru(child, path + [child])

                if leaf is not None and access_time < lru_time:
                    lru_time = access_time
                    lru_leaf = leaf
                    lru_path = leaf_path

            return lru_leaf, lru_path, lru_time

        leaf, path, _ = _find_lru(self.root, [])
        return leaf, path if path else []

    def evict_lru_leaf(self) -> int:
        """
        Evict least-recently-used leaf node

        Finds the LRU leaf, frees its pages (if not shared), and removes it from tree.

        Returns:
            Number of pages freed
        """
        leaf, path = self.find_lru_leaf()

        if leaf is None or not path:
            return 0  # No leaves to evict

        # Collect pages to free (only from nodes with ref_count <= 1)
        pages_to_free = []

        for node in reversed(path):  # Walk from leaf to root
            node.decrement_ref()

            if node.ref_count == 0 and node.kv_pages:
                pages_to_free.extend(node.kv_pages)
                self.total_tokens_cached -= 1

        # Free pages
        if pages_to_free:
            self.kv_pool.free(pages_to_free)

        # Remove leaf from parent
        if len(path) >= 2:
            parent = path[-2]
            parent.remove_child(leaf.token)
            self.num_nodes -= 1
        elif len(path) == 1:
            # Leaf is direct child of root
            self.root.remove_child(leaf.token)
            self.num_nodes -= 1

        return len(pages_to_free)

    def evict_until_available(self, num_pages_needed: int) -> int:
        """
        Evict LRU leaves until enough pages are available

        Args:
            num_pages_needed: Number of pages required

        Returns:
            Total number of pages freed
        """
        total_freed = 0

        while self.kv_pool.num_free_pages < num_pages_needed:
            freed = self.evict_lru_leaf()

            if freed == 0:
                # No more leaves to evict
                break

            total_freed += freed

        return total_freed

    def clear(self):
        """Clear entire cache"""
        self.root = RadixNode(token=None)
        self.num_nodes = 0
        self.total_tokens_cached = 0

    def __repr__(self) -> str:
        return (f"RadixCache(nodes={self.num_nodes}, "
                f"tokens_cached={self.total_tokens_cached}, "
                f"free_pages={self.kv_pool.num_free_pages})")