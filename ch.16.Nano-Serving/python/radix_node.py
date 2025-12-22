"""
Radix Node - Building Block for Prefix Tree

Represents a node in the radix tree for KV cache sharing.
Each node stores a token and tracks associated KV pages.
"""

from typing import Dict, List, Optional
import time

class RadixNode:
    """
    Node in radix tree for prefix sharing

    Each node represents a token in a sequence. Nodes form a tree
    where paths from root to leaves represent complete token sequences.
    Shared prefixes share the same path, enabling KV cache reuse.
    """

    def __init__(self, token: Optional[int] = None):
        """
        Initialize radix node

        Args:
            token: Token ID this node represents (None for root)
        """
        self.token = token

        # Children indexed by next token
        self.children: Dict[int, 'RadixNode'] = {}

        # KV cache pages for this token
        self.kv_pages: List[int] = []

        # Reference counting for shared nodes
        self.ref_count: int = 0

        # LRU tracking
        self.last_access_time: float = time.time()

        # Optional: track which requests use this node
        self.request_ids: List[int] = []

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)"""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """Check if this is the root node"""
        return self.token is None

    @property
    def num_descendants(self) -> int:
        """Count total number of descendant nodes"""
        if self.is_leaf:
            return 0

        count = len(self.children)
        for child in self.children.values():
            count += child.num_descendants

        return count

    def update_access_time(self):
        """Update last access time to current time"""
        self.last_access_time = time.time()

    def increment_ref(self):
        """Increment reference count"""
        self.ref_count += 1
        self.update_access_time()

    def decrement_ref(self):
        """Decrement reference count"""
        self.ref_count = max(0, self.ref_count - 1)

    def add_child(self, token: int) -> 'RadixNode':
        """
        Add child node for given token

        Args:
            token: Token ID for new child

        Returns:
            New or existing child node
        """
        if token not in self.children:
            self.children[token] = RadixNode(token)

        return self.children[token]

    def get_child(self, token: int) -> Optional['RadixNode']:
        """
        Get child node for token

        Args:
            token: Token ID to look up

        Returns:
            Child node or None if not found
        """
        return self.children.get(token)

    def remove_child(self, token: int) -> bool:
        """
        Remove child node

        Args:
            token: Token ID of child to remove

        Returns:
            True if child was removed, False if not found
        """
        if token in self.children:
            del self.children[token]
            return True
        return False

    def get_path_tokens(self, path: List['RadixNode']) -> List[int]:
        """
        Get token sequence from a path

        Args:
            path: List of nodes from root to this node

        Returns:
            List of token IDs
        """
        tokens = []
        for node in path:
            if not node.is_root:
                tokens.append(node.token)
        return tokens

    def __repr__(self) -> str:
        return (f"RadixNode(token={self.token}, "
                f"children={len(self.children)}, "
                f"pages={len(self.kv_pages)}, "
                f"refs={self.ref_count})")