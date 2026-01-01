// Radix Cache Implementation (pointer-free, using node IDs)
#include "radix_cache.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace nano_serving {

//===----------------------------------------------------------------------===//
// RadixNode Implementation
//===----------------------------------------------------------------------===//

NodeID RadixNode::get_child(int token) const {
    auto it = children_.find(token);
    if (it != children_.end()) {
        return it->second;
    }
    return INVALID_NODE;
}

void RadixNode::update_access_time() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    last_access_time_ = std::chrono::duration<double>(duration).count();
}

//===----------------------------------------------------------------------===//
// RadixCache Implementation
//===----------------------------------------------------------------------===//

RadixCache::RadixCache()
    : num_nodes_(0),
      total_tokens_cached_(0) {
    // Allocate root node
    root_id_ = allocate_node(-1);
}

NodeID RadixCache::allocate_node(int token) {
    // Try to find an inactive slot first (for reuse)
    for (size_t i = 0; i < node_active_.size(); ++i) {
        if (!node_active_[i]) {
            nodes_[i] = RadixNode(token);
            node_active_[i] = true;
            return static_cast<NodeID>(i);
        }
    }
    
    // No free slots, expand arena
    NodeID new_id = static_cast<NodeID>(nodes_.size());
    nodes_.emplace_back(token);
    node_active_.push_back(true);
    return new_id;
}

void RadixCache::free_node(NodeID id) {
    if (id >= 0 && id < static_cast<NodeID>(nodes_.size())) {
        node_active_[id] = false;
    }
}

const RadixNode& RadixCache::get_node(NodeID id) const {
    if (id < 0 || id >= static_cast<NodeID>(nodes_.size()) || !node_active_[id]) {
        throw std::runtime_error("Invalid node ID: " + std::to_string(id));
    }
    return nodes_[id];
}

RadixNode& RadixCache::get_node(NodeID id) {
    if (id < 0 || id >= static_cast<NodeID>(nodes_.size()) || !node_active_[id]) {
        throw std::runtime_error("Invalid node ID: " + std::to_string(id));
    }
    return nodes_[id];
}

std::pair<int, NodeID> RadixCache::match_prefix(const std::vector<int>& tokens) {
    NodeID current_id = root_id_;
    int matched_len = 0;

    for (size_t i = 0; i < tokens.size(); ++i) {
        RadixNode& node = get_node(current_id);
        NodeID child_id = node.get_child(tokens[i]);
        
        if (child_id != INVALID_NODE) {
            current_id = child_id;
            matched_len = i + 1;
            get_node(current_id).update_access_time();
        } else {
            break;
        }
    }

    return {matched_len, current_id};
}

NodeID RadixCache::insert(const std::vector<int>& tokens,
                         const std::vector<int>& kv_pages) {
    if (tokens.size() != kv_pages.size()) {
        throw std::runtime_error("Tokens and pages must have same length");
    }

    NodeID current_id = root_id_;

    for (size_t i = 0; i < tokens.size(); ++i) {
        int token = tokens[i];
        NodeID child_id = get_node(current_id).get_child(token);

        if (child_id == INVALID_NODE) {
            // Create new node
            child_id = allocate_node(token);
            // CRITICAL: Re-get node reference after allocation (vector may have reallocated)
            get_node(current_id).add_child(token, child_id);
            get_node(child_id).set_kv_pages({kv_pages[i]});
            num_nodes_++;
            total_tokens_cached_++;
        }

        current_id = child_id;
    }

    get_node(current_id).update_access_time();
    return current_id;
}

std::vector<int> RadixCache::get_pages_for_prefix(const std::vector<int>& tokens) {
    NodeID current_id = root_id_;
    std::vector<int> pages;

    for (int token : tokens) {
        RadixNode& node = get_node(current_id);
        NodeID child_id = node.get_child(token);
        
        if (child_id == INVALID_NODE) {
            break;
        }

        const auto& child_pages = get_node(child_id).get_kv_pages();
        pages.insert(pages.end(), child_pages.begin(), child_pages.end());
        current_id = child_id;
    }

    return pages;
}

RadixCache::LRUResult RadixCache::find_lru_recursive(NodeID node_id, std::vector<NodeID> path) {
    const RadixNode& node = get_node(node_id);
    
    if (node.is_leaf()) {
        return {node_id, path, node.get_last_access_time()};
    }

    NodeID lru_leaf_id = INVALID_NODE;
    std::vector<NodeID> lru_path;
    double lru_time = std::numeric_limits<double>::max();

    // Recursively find LRU leaf among all children
    for (const auto& [token, child_id] : node.get_children()) {
        std::vector<NodeID> child_path = path;
        child_path.push_back(child_id);
        
        auto result = find_lru_recursive(child_id, child_path);
        
        if (result.leaf_id != INVALID_NODE && result.time < lru_time) {
            lru_leaf_id = result.leaf_id;
            lru_path = result.path;
            lru_time = result.time;
        }
    }

    return {lru_leaf_id, lru_path, lru_time};
}

std::pair<NodeID, std::vector<NodeID>> RadixCache::find_lru_leaf() {
    auto result = find_lru_recursive(root_id_, {});
    return {result.leaf_id, result.path};
}

std::vector<int> RadixCache::evict_leaf(NodeID leaf_id, const std::vector<NodeID>& path) {
    if (leaf_id == INVALID_NODE || path.empty()) {
        return {};
    }

    std::vector<int> freed_pages;

    // Collect pages from leaf to root (in reverse order of path)
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        NodeID node_id = *it;
        RadixNode& node = get_node(node_id);

        // If this node is a leaf, evict it
        if (node.is_leaf()) {
            const auto& pages = node.get_kv_pages();
            freed_pages.insert(freed_pages.end(), pages.begin(), pages.end());

            // Remove from parent
            if (it + 1 != path.rend()) {
                NodeID parent_id = *(it + 1);
                RadixNode& parent = get_node(parent_id);
                parent.remove_child(node.get_token());
                free_node(node_id);
                num_nodes_--;
                total_tokens_cached_--;
            }
        } else {
            // Stop if node has children
            break;
        }
    }

    return freed_pages;
}

std::vector<int> RadixCache::evict_until_available(int num_pages) {
    std::vector<int> all_freed_pages;

    while (all_freed_pages.size() < static_cast<size_t>(num_pages) && num_nodes_ > 0) {
        auto [leaf_id, path] = find_lru_leaf();
        if (leaf_id == INVALID_NODE) {
            break;  // No more leaves to evict
        }

        std::vector<int> freed_pages = evict_leaf(leaf_id, path);
        all_freed_pages.insert(all_freed_pages.end(), freed_pages.begin(), freed_pages.end());
    }

    return all_freed_pages;
}

void RadixCache::clear() {
    nodes_.clear();
    node_active_.clear();
    root_id_ = allocate_node(-1);
    num_nodes_ = 0;
    total_tokens_cached_ = 0;
}

} // namespace nano_serving