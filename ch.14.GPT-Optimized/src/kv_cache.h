// KV Cache Pool - Page-based memory management for GPT inference
// Manages physical KV cache pages for transformer inference.
// Supports page allocation, deallocation, and K/V storage.
#pragma once

#include <vector>
#include <set>
#include <memory>
#include <stdexcept>

namespace gpt_optimized {

class KVCachePool {
public:
    KVCachePool(int num_pages, int page_size, int num_layers, int num_heads, int head_dim);
    ~KVCachePool() = default;

    // Allocate pages for tokens
    std::vector<int> allocate(int num_tokens);

    /**
     * Free pages back to pool
     * 
     * @param pages Vector of page indices to free
     */
    void free(const std::vector<int>& pages);

    /**
     * Store K/V tensors at specified pages
     * 
     * @param k Key tensor [num_tokens, num_heads, head_dim]
     * @param v Value tensor [num_tokens, num_heads, head_dim]
     * @param page_indices Pages to store in
     * @param layer_id Layer index
     * @param num_tokens Number of tokens to store
     */
    void storeKV(const float* k, const float* v, 
                 const std::vector<int>& page_indices,
                 int layer_id, int num_tokens);

    /**
     * Get pointers to K/V cache for a layer
     * 
     * @param layer_id Layer index
     * @return Pair of (k_cache_ptr, v_cache_ptr)
     */
    std::pair<float*, float*> getLayerCache(int layer_id);

    /**
     * Calculate pages needed for num_tokens
     */
    int pagesNeeded(int num_tokens) const {
        return (num_tokens + page_size_ - 1) / page_size_;
    }

    /**
     * Check if allocation would succeed
     */
    bool canAllocate(int num_tokens) const {
        return static_cast<int>(free_pages_.size()) >= pagesNeeded(num_tokens);
    }

    /**
     * Get number of free pages
     */
    int getNumFreePages() const { return free_pages_.size(); }

    /**
     * Get total number of pages
     */
    int getNumPages() const { return num_pages_; }

    /**
     * Get page size
     */
    int getPageSize() const { return page_size_; }

private:
    int num_pages_;
    int page_size_;
    int num_layers_;
    int num_heads_;
    int head_dim_;

    // K/V cache storage
    // Shape: [num_layers][num_pages * page_size * num_heads * head_dim]
    std::vector<std::vector<float>> k_cache_;
    std::vector<std::vector<float>> v_cache_;

    // Free page tracking
    std::set<int> free_pages_;

    // Helper: get linear index for cache access
    int getLinearIndex(int page_idx, int token_offset_in_page, int head_idx) const;
};

} // namespace gpt_optimized