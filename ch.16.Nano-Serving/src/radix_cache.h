/*
 * Radix Cache - Prefix Tree for KV Cache Sharing
 *
 * Implements a radix tree (prefix tree) to automatically detect and reuse
 * shared prefixes across requests, enabling efficient KV cache sharing.
 *
 * Uses std::shared_ptr for automatic lifetime management (RAII).
 */

#pragma once

#include <vector>
#include <map>
#include <memory>
#include <chrono>

namespace nano_serving {

class RadixNode : public std::enable_shared_from_this<RadixNode> {
public:
    /**
     * Initialize radix node
     *
     * @param token Token ID this node represents (-1 for root)
     */
    explicit RadixNode(int token = -1);

    ~RadixNode() = default;

    // Prevent copying (use shared_ptr for sharing)
    RadixNode(const RadixNode&) = delete;
    RadixNode& operator=(const RadixNode&) = delete;

    // Node properties
    int get_token() const { return token_; }
    bool is_leaf() const { return children_.empty(); }
    bool is_root() const { return token_ == -1; }
    long use_count() const;  // Returns shared_ptr reference count
    double get_last_access_time() const { return last_access_time_; }
    const std::vector<int>& get_kv_pages() const { return kv_pages_; }

    // Children access (returns shared_ptr for RAII)
    std::shared_ptr<RadixNode> get_child(int token);
    std::shared_ptr<RadixNode> add_child(int token);
    bool remove_child(int token);

    // KV pages management
    void set_kv_pages(const std::vector<int>& pages) { kv_pages_ = pages; }
    void add_kv_page(int page) { kv_pages_.push_back(page); }

    // Timing
    void update_access_time();

    // Tree navigation
    int num_descendants() const;

private:
    int token_;
    std::map<int, std::shared_ptr<RadixNode>> children_;
    std::vector<int> kv_pages_;
    double last_access_time_;
};

class RadixCache {
public:
    /**
     * Initialize radix cache
     */
    RadixCache();

    ~RadixCache() = default;

    /**
     * Find longest matching prefix in tree
     *
     * @param tokens Token sequence to match
     * @return Pair of (match_length, last_matched_node)
     */
    std::pair<int, std::shared_ptr<RadixNode>> match_prefix(const std::vector<int>& tokens);

    /**
     * Insert token sequence into tree
     *
     * @param tokens Token sequence to insert
     * @param kv_pages KV pages for each token (same length as tokens)
     * @return Leaf node representing this sequence
     */
    std::shared_ptr<RadixNode> insert(const std::vector<int>& tokens, 
                                      const std::vector<int>& kv_pages);

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
     * @return Pair of (lru_leaf, path_to_leaf)
     */
    std::pair<std::shared_ptr<RadixNode>, std::vector<std::shared_ptr<RadixNode>>> find_lru_leaf();

    /**
     * Evict a leaf node (remove from tree)
     *
     * @param leaf Leaf node to evict
     * @param path Path from root to leaf
     * @return KV pages that were freed
     */
    std::vector<int> evict_leaf(std::shared_ptr<RadixNode> leaf, 
                                const std::vector<std::shared_ptr<RadixNode>>& path);

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

    // Root access
    std::shared_ptr<RadixNode> get_root() { return root_; }

private:
    std::shared_ptr<RadixNode> root_;
    int num_nodes_;
    int total_tokens_cached_;

    // Helper for finding LRU leaf
    std::tuple<std::shared_ptr<RadixNode>, std::vector<std::shared_ptr<RadixNode>>, double> 
    find_lru_recursive(std::shared_ptr<RadixNode> node, std::vector<std::shared_ptr<RadixNode>> path);
};

} // namespace nano_serving