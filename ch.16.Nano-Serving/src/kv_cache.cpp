/*
 * KV Cache Pool Implementation
 */

#include "kv_cache.h"
#include <algorithm>
#include <cstring>
#include <sstream>

namespace nano_serving {

KVCachePool::KVCachePool(int num_pages, int page_size, 
                         int num_layers, int num_heads, int head_dim)
    : num_pages_(num_pages),
      page_size_(page_size),
      num_layers_(num_layers),
      num_heads_(num_heads),
      head_dim_(head_dim) {

    // Initialize free pages
    for (int i = 0; i < num_pages; ++i) {
        free_pages_.insert(i);
    }

    // Allocate cache storage
    int cache_size_per_layer = num_pages * page_size * num_heads * head_dim;

    k_cache_.resize(num_layers);
    v_cache_.resize(num_layers);

    for (int layer = 0; layer < num_layers; ++layer) {
        k_cache_[layer].resize(cache_size_per_layer, 0.0f);
        v_cache_[layer].resize(cache_size_per_layer, 0.0f);
    }
}

std::vector<int> KVCachePool::allocate(int num_tokens) {
    // Calculate number of pages needed
    int num_pages_needed = (num_tokens + page_size_ - 1) / page_size_;

    if (static_cast<int>(free_pages_.size()) < num_pages_needed) {
        std::ostringstream oss;
        oss << "KV cache pool exhausted: need " << num_pages_needed 
            << " pages, only " << free_pages_.size() << " available";
        throw std::runtime_error(oss.str());
    }

    // Allocate pages
    std::vector<int> allocated;
    allocated.reserve(num_pages_needed);

    auto it = free_pages_.begin();
    for (int i = 0; i < num_pages_needed; ++i) {
        allocated.push_back(*it);
        it = free_pages_.erase(it);
    }

    return allocated;
}

void KVCachePool::free(const std::vector<int>& pages) {
    for (int page : pages) {
        if (page < 0 || page >= num_pages_) {
            throw std::runtime_error("Invalid page index: " + std::to_string(page));
        }
        free_pages_.insert(page);
    }
}

int KVCachePool::getLinearIndex(int page_idx, int token_offset_in_page, 
                                 int head_idx) const {
    // Cache layout: [num_pages][page_size][num_heads][head_dim]
    // Linear index = page_idx * (page_size * num_heads * head_dim)
    //              + token_offset * (num_heads * head_dim)
    //              + head_idx * head_dim
    int tokens_per_page = page_size_ * num_heads_ * head_dim_;
    int tokens_in_head = num_heads_ * head_dim_;

    return page_idx * tokens_per_page 
         + token_offset_in_page * tokens_in_head
         + head_idx * head_dim_;
}

void KVCachePool::storeKV(const float* k, const float* v, 
                          const std::vector<int>& page_indices,
                          int layer_id, int num_tokens) {
    if (layer_id < 0 || layer_id >= num_layers_) {
        throw std::runtime_error("Invalid layer_id: " + std::to_string(layer_id));
    }

    auto& k_layer = k_cache_[layer_id];
    auto& v_layer = v_cache_[layer_id];

    int token_idx = 0;

    for (size_t page_num = 0; page_num < page_indices.size(); ++page_num) {
        int page_idx = page_indices[page_num];

        // How many tokens to store in this page
        int tokens_in_page = std::min(page_size_, num_tokens - token_idx);

        for (int token_offset = 0; token_offset < tokens_in_page; ++token_offset) {
            for (int head = 0; head < num_heads_; ++head) {
                int cache_idx = getLinearIndex(page_idx, token_offset, head);

                // Source index in input tensor: [token_idx][head][:]
                int src_idx = (token_idx * num_heads_ + head) * head_dim_;

                // Copy head_dim elements
                std::memcpy(&k_layer[cache_idx], &k[src_idx], head_dim_ * sizeof(float));
                std::memcpy(&v_layer[cache_idx], &v[src_idx], head_dim_ * sizeof(float));
            }
            token_idx++;
        }
    }
}

std::pair<float*, float*> KVCachePool::getLayerCache(int layer_id) {
    if (layer_id < 0 || layer_id >= num_layers_) {
        throw std::runtime_error("Invalid layer_id: " + std::to_string(layer_id));
    }

    return {k_cache_[layer_id].data(), v_cache_[layer_id].data()};
}

} // namespace nano_serving