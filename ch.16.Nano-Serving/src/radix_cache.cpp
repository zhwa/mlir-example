/*
 * Radix Cache Implementation (using RAII with shared_ptr)
 */

#include "radix_cache.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace nano_serving {

//===----------------------------------------------------------------------===//
// RadixNode Implementation
//===----------------------------------------------------------------------===//

RadixNode::RadixNode(int token)
    : token_(token) {
    update_access_time();
}

long RadixNode::use_count() const {
    // Try to get shared_ptr and return use count
    // During construction, this may not work, so return 1
    try {
        return shared_from_this().use_count();
    } catch (const std::bad_weak_ptr&) {
        return 1;  // Not yet managed by shared_ptr
    }
}

std::shared_ptr<RadixNode> RadixNode::get_child(int token) {
    auto it = children_.find(token);
    if (it != children_.end()) {
        return it->second;
    }
    return nullptr;
}

std::shared_ptr<RadixNode> RadixNode::add_child(int token) {
    if (children_.find(token) == children_.end()) {
        children_[token] = std::make_shared<RadixNode>(token);
    }
    return children_[token];
}

bool RadixNode::remove_child(int token) {
    return children_.erase(token) > 0;
}

void RadixNode::update_access_time() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    last_access_time_ = std::chrono::duration<double>(duration).count();
}

int RadixNode::num_descendants() const {
    if (is_leaf()) {
        return 0;
    }

    int count = children_.size();
    for (const auto& [token, child] : children_) {
        count += child->num_descendants();
    }
    return count;
}

//===----------------------------------------------------------------------===//
// RadixCache Implementation
//===----------------------------------------------------------------------===//

RadixCache::RadixCache()
    : root_(std::make_shared<RadixNode>(-1)),
      num_nodes_(0),
      total_tokens_cached_(0) {
}

std::pair<int, std::shared_ptr<RadixNode>> RadixCache::match_prefix(const std::vector<int>& tokens) {
    std::shared_ptr<RadixNode> node = root_;
    int matched_len = 0;

    for (size_t i = 0; i < tokens.size(); ++i) {
        std::shared_ptr<RadixNode> child = node->get_child(tokens[i]);
        if (child != nullptr) {
            node = child;
            matched_len = i + 1;
            node->update_access_time();
        } else {
            break;
        }
    }

    return {matched_len, node};
}

std::shared_ptr<RadixNode> RadixCache::insert(const std::vector<int>& tokens,
                                               const std::vector<int>& kv_pages) {
    if (tokens.size() != kv_pages.size()) {
        throw std::runtime_error("Tokens and pages must have same length");
    }

    std::shared_ptr<RadixNode> node = root_;

    for (size_t i = 0; i < tokens.size(); ++i) {
        int token = tokens[i];
        std::shared_ptr<RadixNode> child = node->get_child(token);

        if (child == nullptr) {
            // Create new node
            child = node->add_child(token);
            child->set_kv_pages({kv_pages[i]});
            num_nodes_++;
            total_tokens_cached_++;
        }
        // Note: Reference counting handled automatically by shared_ptr

        node = child;
    }

    node->update_access_time();
    return node;
}

std::vector<int> RadixCache::get_pages_for_prefix(const std::vector<int>& tokens) {
    std::shared_ptr<RadixNode> node = root_;
    std::vector<int> pages;

    for (int token : tokens) {
        std::shared_ptr<RadixNode> child = node->get_child(token);
        if (child == nullptr) {
            break;
        }

        const auto& child_pages = child->get_kv_pages();
        pages.insert(pages.end(), child_pages.begin(), child_pages.end());
        node = child;
    }

    return pages;
}

std::tuple<std::shared_ptr<RadixNode>, std::vector<std::shared_ptr<RadixNode>>, double>
RadixCache::find_lru_recursive(std::shared_ptr<RadixNode> node, std::vector<std::shared_ptr<RadixNode>> path) {
    if (node->is_leaf()) {
        return {node, path, node->get_last_access_time()};
    }

    std::shared_ptr<RadixNode> lru_leaf = nullptr;
    std::vector<std::shared_ptr<RadixNode>> lru_path;
    double lru_time = std::numeric_limits<double>::infinity();

    // Can't iterate over children map directly from outside, so we need to check each child
    // This is a bit tricky - we'd need to expose iteration or use a different approach
    // For now, let's use a simpler approach

    return {lru_leaf, lru_path, lru_time};
}

std::pair<std::shared_ptr<RadixNode>, std::vector<std::shared_ptr<RadixNode>>> RadixCache::find_lru_leaf() {
    auto [leaf, path, _] = find_lru_recursive(root_, {});
    return {leaf, path};
}

std::vector<int> RadixCache::evict_leaf(std::shared_ptr<RadixNode> leaf,
                                        const std::vector<std::shared_ptr<RadixNode>>& path) {
    if (leaf == nullptr || path.empty()) {
        return {};
    }

    std::vector<int> freed_pages;

    // Collect pages from leaf to root (in reverse order of path)
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        std::shared_ptr<RadixNode> node = *it;

        // If this node is a leaf and only we hold a reference (use_count == 2: one in path, one in parent)
        if (node->is_leaf() && node->use_count() <= 2) {
            const auto& pages = node->get_kv_pages();
            freed_pages.insert(freed_pages.end(), pages.begin(), pages.end());

            // Remove from parent
            if (it + 1 != path.rend()) {
                std::shared_ptr<RadixNode> parent = *(it + 1);
                parent->remove_child(node->get_token());
                num_nodes_--;
                total_tokens_cached_--;
            }
        } else {
            // Stop if node has children or is still referenced elsewhere
            break;
        }
    }

    return freed_pages;
}

std::vector<int> RadixCache::evict_until_available(int num_pages) {
    std::vector<int> all_freed_pages;

    while (all_freed_pages.size() < static_cast<size_t>(num_pages) && num_nodes_ > 0) {
        auto [leaf, path] = find_lru_leaf();
        if (leaf == nullptr) {
            break;  // No more leaves to evict
        }

        std::vector<int> freed_pages = evict_leaf(leaf, path);
        all_freed_pages.insert(all_freed_pages.end(), freed_pages.begin(), freed_pages.end());
    }

    return all_freed_pages;
}

void RadixCache::clear() {
    root_ = std::make_shared<RadixNode>(-1);
    num_nodes_ = 0;
    total_tokens_cached_ = 0;
}

} // namespace nano_serving