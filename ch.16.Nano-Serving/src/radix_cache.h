/*
 * Radix Cache - Prefix Tree for KV Cache Sharing
 *
 * Implements a radix tree (prefix tree) to automatically detect and reuse
 * shared prefixes across requests, enabling efficient KV cache sharing.
 *
 * Uses integer node IDs instead of pointers for safer memory management.
 */

#pragma once

#include <vector>
#include <map>
#include <optional>
#include <chrono>

namespace nano_serving {

// Forward declaration
class RadixCache;

// Node ID type for type safety
using NodeID = int;
constexpr NodeID INVALID_NODE = -1;

/**
 * RadixNode - stored in arena, referenced by ID
 * 
 * No pointers, no reference counting - just plain data.
 */
class RadixNode {
public:
    explicit RadixNode(int token = -1)
        : token_(token), last_access_time_(0.0) {
        update_access_time();
    }

    // Node properties
    int get_token() const { return token_; }
    bool is_leaf() const { return children_.empty(); }
    bool is_root() const { return token_ == -1; }
    double get_last_access_time() const { return last_access_time_; }
    const std::vector<int>& get_kv_pages() const { return kv_pages_; }

    // Children access (returns node ID, not pointer)
    bool has_child(int token) const { return children_.count(token) > 0; }
    NodeID get_child(int token) const;
    void add_child(int token, NodeID child_id) { children_[token] = child_id; }
    void remove_child(int token) { children_.erase(token); }
    const std::map<int, NodeID>& get_children() const { return children_; }

    // KV pages management
    void set_kv_pages(const std::vector<int>& pages) { kv_pages_ = pages; }
    void add_kv_page(int page) { kv_pages_.push_back(page); }

    // Timing
    void update_access_time();

private:
    int token_;
    std::map<int, NodeID> children_;  // token -> child node ID
    std::vector<int> kv_pages_;
    double last_access_time_;
};

/**
 * RadixCache - Arena-based radix tree
 * 
 * Stores all nodes in a vector (arena), uses integer IDs instead of pointers.
 * No reference counting, no manual memory management - just clean data structures.
 */
class RadixCache {
public:
    RadixCache();
    ~RadixCache() = default;

    /**
     * Find longest matching prefix in tree
     *
     * @param tokens Token sequence to match
     * @return Pair of (match_length, last_matched_node_id)
     */
    std::pair<int, NodeID> match_prefix(const std::vector<int>& tokens);

    /**
     * Insert token sequence into tree
     *
     * @param tokens Token sequence to insert
     * @param kv_pages KV pages for each token (same length as tokens)
     * @return Node ID of leaf representing this sequence
     */
    NodeID insert(const std::vector<int>& tokens, const std::vector<int>& kv_pages);

    /**
     * Get KV pages for a prefix
     *
     * @param tokens Token sequence
     * @return List of KV page indices for the prefix
     */
    std::vector<int> get_pages_for_prefix(const std::vector<int>& tokens);

    /**
     * Find least-recently-used leaf node
     *
     * @return Pair of (lru_leaf_id, path_to_leaf_ids)
     */
    std::pair<NodeID, std::vector<NodeID>> find_lru_leaf();

    /**
     * Evict a leaf node (remove from tree)
     *
     * @param leaf_id Leaf node ID to evict
     * @param path Path from root to leaf (node IDs)
     * @return KV pages that were freed
     */
    std::vector<int> evict_leaf(NodeID leaf_id, const std::vector<NodeID>& path);

    /**
     * Evict LRU leaves until at least num_pages are freed
     *
     * @param num_pages Number of pages needed
     * @return Total KV pages that were freed
     */
    std::vector<int> evict_until_available(int num_pages);

    /**
     * Clear entire cache
     */
    void clear();

    // Statistics
    int get_num_nodes() const { return num_nodes_; }
    int get_total_tokens_cached() const { return total_tokens_cached_; }

    // Node access (for Python bindings)
    NodeID get_root_id() const { return root_id_; }
    const RadixNode& get_node(NodeID id) const;
    RadixNode& get_node(NodeID id);

private:
    // Arena storage - all nodes live here
    std::vector<RadixNode> nodes_;
    std::vector<bool> node_active_;  // Track which nodes are in use
    
    NodeID root_id_;
    int num_nodes_;
    int total_tokens_cached_;

    // Allocate a new node in the arena
    NodeID allocate_node(int token);
    
    // Free a node (mark as inactive)
    void free_node(NodeID id);

    // Helper for finding LRU leaf
    struct LRUResult {
        NodeID leaf_id;
        std::vector<NodeID> path;
        double time;
    };
    LRUResult find_lru_recursive(NodeID node_id, std::vector<NodeID> path);
};

} // namespace nano_serving