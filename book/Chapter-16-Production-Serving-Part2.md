# Chapter 16 Part 2: Nano-Serving Implementation - Core Components

Part 1 introduced production LLM serving concepts—paged KV cache, continuous batching, radix cache, chunked prefill. These algorithmic techniques enable 100-1000× speedups in real systems like vLLM and SGLang. Part 2 demonstrates **our specific implementation** combining MLIR compiler acceleration with Python inference orchestration.

This chapter builds nano-serving in six progressive phases. Part 2 covers **Phases 0-2** (core components and integration), Part 3 covers **Phases 3-6** (advanced features and complete system). By the end, you'll understand how Python scheduling coordinates with C++ model execution through clean interfaces.

**Implementation Philosophy**: Production systems prioritize throughput—CUDA kernels, GPU-optimized attention, multi-GPU tensor parallelism. Our implementation prioritizes **clarity**—readable Python, clean C++ bindings, educational structure. The algorithms are identical; the optimization level differs. Understanding our implementation prepares you to read vLLM/SGLang source code.

**Architecture Overview**:

```
Python Layer (Scheduling & Control)
├── Request/Batch abstractions
├── KV pool wrapper
├── Prefill/Decode managers
├── Radix cache (prefix tree)
└── Continuous batching loop
         │
         ▼ (Python/C++ boundary via pybind11)
         │
C++/MLIR Layer (Model Execution)
├── KV cache pool (paged memory)
├── GPT model (MLIR JIT from Ch.14)
└── Forward pass optimization
```

**Why This Split?** Python excels at high-level logic (tree operations, scheduling decisions, request management). C++ excels at performance-critical code (matrix operations, memory management, tight loops). Combining both leverages each language's strengths.

**Development Approach**: Build incrementally with tests at each phase. Each component has clear inputs/outputs and can be tested independently before integration. This matches real-world development—validate components before assembling the system.

## 16.10 Phase 0: Request and Batch Abstractions

Before implementing schedulers or memory management, we need data structures representing user tasks and batched execution.

**Request Class**: Tracks a single generation task through its lifecycle.

```python
# python/request.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class Request:
    """Represents a single generation request"""
    
    req_id: int                      # Unique identifier
    prompt_tokens: List[int]         # Input token IDs [1, 2, 3, ...]
    max_tokens: int                  # Generate up to N tokens
    temperature: float = 1.0         # Sampling temperature
    ignore_eos: bool = False         # Continue past EOS token?
    
    # State tracking
    cached_len: int = 0              # Tokens with KV cache computed
    output_tokens: List[int] = field(default_factory=list)
    is_finished: bool = False
    
    # KV cache management
    kv_pages: List[int] = field(default_factory=list)  # Physical pages
    
    @property
    def total_len(self) -> int:
        """Total tokens processed (prompt + output)"""
        return len(self.prompt_tokens) + len(self.output_tokens)
    
    @property
    def extend_len(self) -> int:
        """Tokens to process in next forward pass"""
        if len(self.output_tokens) == 0:
            # Prefill phase: process from cached_len to end of prompt
            return len(self.prompt_tokens) - self.cached_len
        else:
            # Decode phase: only process last generated token
            return 1
    
    @property
    def device_len(self) -> int:
        """Total tokens in device memory (cached + extend)"""
        return self.cached_len + self.extend_len
```

**Key Properties**:

- `cached_len`: How many tokens have KV cache computed (increases during execution)
- `extend_len`: How many new tokens need processing this iteration (decreases from prompt_len to 1)
- `device_len`: Total tokens in memory (cached + new)

**Request Lifecycle Example**:

```python
# User submits "What is MLIR?" → tokens [1, 2, 3, 4, 5]
req = Request(
    req_id=0,
    prompt_tokens=[1, 2, 3, 4, 5],
    max_tokens=10,
    temperature=0.7
)

# Initial state
req.cached_len == 0         # No KV cache yet
req.extend_len == 5         # Need to process 5 prompt tokens
req.total_len == 5          # 5 tokens total

# After prefill (first forward pass)
req.cached_len = 5          # All prompt tokens cached
req.output_tokens = [42]    # Generated first token
req.extend_len == 1         # Only process new token
req.total_len == 6          # 5 prompt + 1 output

# After decode iteration 2
req.cached_len = 6          # Previous token now cached
req.output_tokens = [42, 73]
req.extend_len == 1         # Still just one new token
req.total_len == 7
```

**Batch Class**: Groups requests for parallel execution.

```python
# python/batch.py
from dataclasses import dataclass
from typing import List
import numpy as np
from python.request import Request

@dataclass
class Batch:
    """Group of requests processed together"""
    
    requests: List[Request]
    input_ids: np.ndarray       # Token IDs to process
    positions: np.ndarray       # Position indices for each token
    kv_indices: List[int]       # KV cache locations
    
    @staticmethod
    def from_prefill(requests: List[Request]) -> 'Batch':
        """Create batch for prefill phase (process all prompt tokens)"""
        all_tokens = []
        all_positions = []
        kv_indices = []
        
        for req in requests:
            # Get tokens needing processing
            start = req.cached_len
            end = len(req.prompt_tokens)
            tokens = req.prompt_tokens[start:end]
            
            # Position indices (relative to this request)
            positions = list(range(start, end))
            
            all_tokens.extend(tokens)
            all_positions.extend(positions)
            
            # KV cache write locations
            for i in range(len(tokens)):
                kv_indices.append(req.cached_len + i)
        
        return Batch(
            requests=requests,
            input_ids=np.array(all_tokens, dtype=np.int32),
            positions=np.array(all_positions, dtype=np.int32),
            kv_indices=kv_indices
        )
    
    @staticmethod
    def from_decode(requests: List[Request]) -> 'Batch':
        """Create batch for decode phase (one token per request)"""
        tokens = []
        positions = []
        kv_indices = []
        
        for req in requests:
            # Last generated token (or last prompt token if first iteration)
            if req.output_tokens:
                token = req.output_tokens[-1]
            else:
                token = req.prompt_tokens[-1]
            
            tokens.append(token)
            positions.append(req.device_len - 1)  # Position of last token
            kv_indices.append(req.device_len - 1)  # Where to write KV
        
        return Batch(
            requests=requests,
            input_ids=np.array(tokens, dtype=np.int32),
            positions=np.array(positions, dtype=np.int32),
            kv_indices=kv_indices
        )
```

**Prefill vs Decode Batches**:

```python
# Example: Two requests in prefill
req1 = Request(req_id=0, prompt_tokens=[1, 2, 3, 4], max_tokens=10)
req2 = Request(req_id=1, prompt_tokens=[5, 6, 7], max_tokens=10)

prefill_batch = Batch.from_prefill([req1, req2])
# input_ids: [1, 2, 3, 4, 5, 6, 7]  (all prompt tokens)
# positions: [0, 1, 2, 3, 0, 1, 2]   (per-request positions)

# After prefill, both in decode
req1.output_tokens = [10]
req2.output_tokens = [20]

decode_batch = Batch.from_decode([req1, req2])
# input_ids: [10, 20]        (one token per request)
# positions: [4, 3]          (next position for each)
```

**Phase 0 Tests** (9 tests validating request/batch logic):

```python
# test_phase0_request_batch.py
def test_request_properties():
    """Validate request state tracking"""
    req = Request(req_id=0, prompt_tokens=[1, 2, 3], max_tokens=5)
    
    assert req.total_len == 3
    assert req.extend_len == 3  # Need prefill
    assert req.cached_len == 0
    
    # Simulate prefill
    req.cached_len = 3
    req.output_tokens = [10]
    
    assert req.total_len == 4   # 3 prompt + 1 output
    assert req.extend_len == 1  # Only new token

def test_prefill_batch():
    """Test prefill batch construction"""
    reqs = [
        Request(req_id=0, prompt_tokens=[1, 2, 3], max_tokens=5),
        Request(req_id=1, prompt_tokens=[4, 5], max_tokens=5)
    ]
    
    batch = Batch.from_prefill(reqs)
    
    # Should concatenate all tokens
    assert list(batch.input_ids) == [1, 2, 3, 4, 5]
    assert list(batch.positions) == [0, 1, 2, 0, 1]
    assert len(batch.requests) == 2

def test_decode_batch():
    """Test decode batch construction"""
    reqs = [
        Request(req_id=0, prompt_tokens=[1, 2, 3], max_tokens=5),
        Request(req_id=1, prompt_tokens=[4, 5], max_tokens=5)
    ]
    
    # Simulate completed prefill
    reqs[0].cached_len = 3
    reqs[0].output_tokens = [10, 11]
    reqs[1].cached_len = 2
    reqs[1].output_tokens = [20]
    
    batch = Batch.from_decode(reqs)
    
    # Should have one token per request (last generated)
    assert list(batch.input_ids) == [11, 20]
    assert len(batch.positions) == 2
    assert len(batch.requests) == 2
```

Phase 0 establishes the data model for requests and batches—the foundation for all subsequent phases.

## 16.11 Phase 1: KV Cache Pool (C++ Implementation)

KV cache stores attention keys and values for all previous tokens. Naive allocation (max_seq_len per request) wastes memory. **Paged allocation** divides cache into fixed-size pages, allocating on-demand.

**C++ Implementation** (performance-critical memory management):

```cpp
// src/kv_cache.h
#pragma once
#include <vector>
#include <set>
#include <stdexcept>

namespace nano_serving {

class KVCachePool {
public:
    /**
     * Initialize KV cache pool
     * 
     * @param num_pages Total pages in pool
     * @param page_size Tokens per page (typically 16)
     * @param num_layers Transformer layers
     * @param num_heads Attention heads
     * @param head_dim Dimension per head
     */
    KVCachePool(int num_pages, int page_size, 
                int num_layers, int num_heads, int head_dim);
    
    /**
     * Allocate pages for tokens
     * 
     * @param num_tokens Number of tokens requiring cache
     * @return Vector of allocated page indices
     */
    std::vector<int> allocate(int num_tokens);
    
    /**
     * Free pages back to pool
     * 
     * @param pages Page indices to free
     */
    void free(const std::vector<int>& pages);
    
    /**
     * Store K/V tensors at pages
     * 
     * @param k Key tensor [num_tokens, num_heads, head_dim]
     * @param v Value tensor [num_tokens, num_heads, head_dim]
     * @param page_indices Physical pages to write
     * @param layer_id Layer index
     * @param num_tokens Number of tokens to store
     */
    void storeKV(const float* k, const float* v,
                 const std::vector<int>& page_indices,
                 int layer_id, int num_tokens);
    
    /**
     * Get cache pointers for layer
     * 
     * @param layer_id Layer index
     * @return Pair of (k_cache_ptr, v_cache_ptr)
     */
    std::pair<float*, float*> getLayerCache(int layer_id);
    
    int getNumFreePages() const { return free_pages_.size(); }
    int getPageSize() const { return page_size_; }

private:
    int num_pages_;
    int page_size_;
    int num_layers_;
    int num_heads_;
    int head_dim_;
    
    // K/V cache storage [num_layers][num_pages * page_size * num_heads * head_dim]
    std::vector<std::vector<float>> k_cache_;
    std::vector<std::vector<float>> v_cache_;
    
    // Free page tracking
    std::set<int> free_pages_;
};

} // namespace nano_serving
```

**Implementation** (src/kv_cache.cpp):

```cpp
#include "kv_cache.h"
#include <algorithm>

namespace nano_serving {

KVCachePool::KVCachePool(int num_pages, int page_size,
                         int num_layers, int num_heads, int head_dim)
    : num_pages_(num_pages), page_size_(page_size),
      num_layers_(num_layers), num_heads_(num_heads), head_dim_(head_dim) {
    
    // Allocate cache storage
    int tokens_per_layer = num_pages * page_size;
    int elements_per_token = num_heads * head_dim;
    int layer_size = tokens_per_layer * elements_per_token;
    
    k_cache_.resize(num_layers);
    v_cache_.resize(num_layers);
    
    for (int i = 0; i < num_layers; i++) {
        k_cache_[i].resize(layer_size, 0.0f);
        v_cache_[i].resize(layer_size, 0.0f);
    }
    
    // Initialize free pages
    for (int i = 0; i < num_pages; i++) {
        free_pages_.insert(i);
    }
}

std::vector<int> KVCachePool::allocate(int num_tokens) {
    int pages_needed = (num_tokens + page_size_ - 1) / page_size_;
    
    if (free_pages_.size() < static_cast<size_t>(pages_needed)) {
        throw std::runtime_error("Out of KV cache memory");
    }
    
    std::vector<int> allocated;
    auto it = free_pages_.begin();
    
    for (int i = 0; i < pages_needed; i++) {
        allocated.push_back(*it);
        it = free_pages_.erase(it);
    }
    
    return allocated;
}

void KVCachePool::free(const std::vector<int>& pages) {
    for (int page : pages) {
        free_pages_.insert(page);
    }
}

void KVCachePool::storeKV(const float* k, const float* v,
                          const std::vector<int>& page_indices,
                          int layer_id, int num_tokens) {
    int token_stride = num_heads_ * head_dim_;
    
    for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
        int page_idx = page_indices[token_idx / page_size_];
        int page_offset = token_idx % page_size_;
        int linear_idx = page_idx * page_size_ + page_offset;
        
        // Copy K and V
        for (int i = 0; i < token_stride; i++) {
            k_cache_[layer_id][linear_idx * token_stride + i] = 
                k[token_idx * token_stride + i];
            v_cache_[layer_id][linear_idx * token_stride + i] = 
                v[token_idx * token_stride + i];
        }
    }
}

std::pair<float*, float*> KVCachePool::getLayerCache(int layer_id) {
    return {k_cache_[layer_id].data(), v_cache_[layer_id].data()};
}

} // namespace nano_serving
```

**Python Bindings** (pybind11 exposes C++ to Python):

```cpp
// src/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "kv_cache.h"

namespace py = pybind11;
using namespace nano_serving;

PYBIND11_MODULE(_nano_serving, m) {
    py::class_<KVCachePool>(m, "KVCachePool")
        .def(py::init<int, int, int, int, int>(),
             py::arg("num_pages"),
             py::arg("page_size"),
             py::arg("num_layers"),
             py::arg("num_heads"),
             py::arg("head_dim"))
        .def("allocate", &KVCachePool::allocate)
        .def("free", &KVCachePool::free)
        .def("get_num_free_pages", &KVCachePool::getNumFreePages)
        .def("get_page_size", &KVCachePool::getPageSize);
}
```

**Python Wrapper** (clean interface for Python code):

```python
# python/kv_pool.py
import sys
sys.path.insert(0, 'build/x64-release/ch.16.Nano-Serving')
import _nano_serving

class KVCachePool:
    """Python wrapper for C++ KV cache pool"""
    
    def __init__(self, num_pages: int, page_size: int,
                 num_layers: int, num_heads: int, head_dim: int):
        self._pool = _nano_serving.KVCachePool(
            num_pages, page_size, num_layers, num_heads, head_dim
        )
        self.page_size = page_size
    
    def allocate(self, num_tokens: int):
        """Allocate pages for tokens"""
        return self._pool.allocate(num_tokens)
    
    def free(self, pages):
        """Free pages back to pool"""
        self._pool.free(pages)
    
    def get_num_free_pages(self) -> int:
        """Get number of free pages"""
        return self._pool.get_num_free_pages()
    
    def pages_needed(self, num_tokens: int) -> int:
        """Calculate pages needed for tokens"""
        return (num_tokens + self.page_size - 1) // self.page_size
```

**Phase 1 Tests** (8 tests validating memory management):

```python
# test_phase1_kv_pool.py
def test_kv_pool_init():
    """Test KV pool initialization"""
    pool = KVCachePool(
        num_pages=64,
        page_size=16,
        num_layers=4,
        num_heads=4,
        head_dim=32
    )
    
    assert pool.get_num_free_pages() == 64
    assert pool.page_size == 16

def test_allocate_free():
    """Test page allocation and deallocation"""
    pool = KVCachePool(64, 16, 4, 4, 32)
    
    # Allocate 20 tokens = 2 pages
    pages = pool.allocate(20)
    assert len(pages) == 2
    assert pool.get_num_free_pages() == 62
    
    # Free pages
    pool.free(pages)
    assert pool.get_num_free_pages() == 64

def test_out_of_memory():
    """Test OOM handling"""
    pool = KVCachePool(4, 16, 4, 4, 32)  # Only 4 pages
    
    # Allocate all pages
    pages = pool.allocate(64)  # 64 tokens = 4 pages
    assert pool.get_num_free_pages() == 0
    
    # Try to allocate more (should raise)
    try:
        pool.allocate(1)
        assert False, "Should raise OOM"
    except:
        pass  # Expected

def test_multiple_allocations():
    """Test multiple concurrent allocations"""
    pool = KVCachePool(64, 16, 4, 4, 32)
    
    pages1 = pool.allocate(32)  # 2 pages
    pages2 = pool.allocate(48)  # 3 pages
    pages3 = pool.allocate(16)  # 1 page
    
    assert pool.get_num_free_pages() == 58
    
    # Free in different order
    pool.free(pages2)
    assert pool.get_num_free_pages() == 61
    
    pool.free(pages1)
    pool.free(pages3)
    assert pool.get_num_free_pages() == 64
```

**Memory Efficiency**:

```python
# Example: 256 concurrent requests
page_size = 16
num_pages = 1024

# Contiguous allocation (max_seq_len=2048)
contiguous_memory = 256 * 2048 * 36_000  # bytes per token
print(f"Contiguous: {contiguous_memory / 1e9:.1f} GB")  # 18.9 GB

# Paged allocation (avg 300 tokens used)
paged_memory = 256 * 300 * 36_000
print(f"Paged: {paged_memory / 1e9:.1f} GB")  # 2.8 GB (6.8× better!)
```

Phase 1 provides efficient memory management—the foundation for scaling to hundreds of concurrent requests.

## 16.12 Phase 2: Prefill and Decode Managers

Prefill (process prompt) and decode (generate tokens) have different characteristics. Separate managers optimize each phase independently.

**Prefill Manager** (FCFS with token budget):

```python
# python/prefill_manager.py
from typing import List, Optional
from python.request import Request
from python.batch import Batch

class PrefillManager:
    """Manages prefill phase scheduling
    
    Strategy: First-Come-First-Served with token budget
    - Process requests in arrival order
    - Limit total tokens per batch (memory constraint)
    """
    
    def __init__(self, token_budget: int = 512):
        """
        Args:
            token_budget: Maximum tokens per prefill batch
        """
        self.queue = []
        self.token_budget = token_budget
    
    def add_request(self, req: Request):
        """Add request needing prefill"""
        self.queue.append(req)
    
    def schedule(self) -> Optional[Batch]:
        """Select requests for next prefill batch
        
        Returns:
            Batch of requests to prefill, or None if queue empty
        """
        if not self.queue:
            return None
        
        selected = []
        total_tokens = 0
        
        # FCFS with budget constraint
        while self.queue:
            req = self.queue[0]
            req_tokens = req.extend_len
            
            if total_tokens + req_tokens <= self.token_budget:
                # Fits in budget
                selected.append(self.queue.pop(0))
                total_tokens += req_tokens
            else:
                # Would exceed budget
                break
        
        if selected:
            return Batch.from_prefill(selected)
        
        return None
    
    def is_empty(self) -> bool:
        return len(self.queue) == 0
```

**Decode Manager** (batch all running requests):

```python
# python/decode_manager.py
from typing import List, Optional
from python.request import Request
from python.batch import Batch

class DecodeManager:
    """Manages decode phase scheduling
    
    Strategy: Batch all running requests
    - Each request generates one token
    - High parallelism (batch dimension)
    """
    
    def __init__(self, max_batch_size: int = 256):
        """
        Args:
            max_batch_size: Maximum requests in decode batch
        """
        self.running = []
        self.max_batch_size = max_batch_size
    
    def add_request(self, req: Request):
        """Add request that finished prefill"""
        self.running.append(req)
    
    def schedule(self) -> Optional[Batch]:
        """Create decode batch from all running requests
        
        Returns:
            Batch with all running requests, or None if none running
        """
        if not self.running:
            return None
        
        # Batch up to max_batch_size requests
        batch_reqs = self.running[:self.max_batch_size]
        return Batch.from_decode(batch_reqs)
    
    def remove_finished(self):
        """Remove completed requests"""
        self.running = [r for r in self.running if not r.is_finished]
    
    def is_empty(self) -> bool:
        return len(self.running) == 0
```

**Integration Example**:

```python
# Serving loop with separate managers
prefill_mgr = PrefillManager(token_budget=512)
decode_mgr = DecodeManager(max_batch_size=32)

while True:
    # 1. Schedule prefill (new requests)
    prefill_batch = prefill_mgr.schedule()
    if prefill_batch:
        logits = model.forward(prefill_batch)
        
        # Sample first token
        for req, logit in zip(prefill_batch.requests, logits):
            token = sample(logit)
            req.output_tokens.append(token)
            req.cached_len = len(req.prompt_tokens)
            
            # Move to decode
            decode_mgr.add_request(req)
    
    # 2. Schedule decode (running requests)
    decode_batch = decode_mgr.schedule()
    if decode_batch:
        logits = model.forward(decode_batch)
        
        # Sample next tokens
        for req, logit in zip(decode_batch.requests, logits):
            token = sample(logit)
            req.output_tokens.append(token)
            req.cached_len += 1
        
        # Remove finished
        decode_mgr.remove_finished()
```

**Phase 2 Tests** (11 tests validating scheduling):

```python
# test_phase2_prefill_decode.py
def test_prefill_manager_fcfs():
    """Test FCFS scheduling with token budget"""
    mgr = PrefillManager(token_budget=100)
    
    # Add requests
    mgr.add_request(Request(0, [1]*50, 10))   # 50 tokens
    mgr.add_request(Request(1, [2]*30, 10))   # 30 tokens
    mgr.add_request(Request(2, [3]*40, 10))   # 40 tokens
    
    # First batch: req0 (50) + req1 (30) = 80 tokens
    batch = mgr.schedule()
    assert len(batch.requests) == 2
    assert batch.requests[0].req_id == 0
    assert batch.requests[1].req_id == 1
    
    # Second batch: req2 (40 tokens)
    batch = mgr.schedule()
    assert len(batch.requests) == 1
    assert batch.requests[0].req_id == 2

def test_decode_manager_batch_all():
    """Test decode batches all running requests"""
    mgr = DecodeManager(max_batch_size=32)
    
    # Add running requests
    for i in range(10):
        req = Request(i, [1, 2, 3], 5)
        req.cached_len = 3
        req.output_tokens = [10]
        mgr.add_request(req)
    
    # Should batch all 10 requests
    batch = mgr.schedule()
    assert len(batch.requests) == 10
    assert len(batch.input_ids) == 10  # One token per request

def test_prefill_then_decode():
    """Test transition from prefill to decode"""
    prefill_mgr = PrefillManager(token_budget=100)
    decode_mgr = DecodeManager()
    
    # Add requests
    req1 = Request(0, [1, 2, 3], 5)
    req2 = Request(1, [4, 5], 5)
    
    prefill_mgr.add_request(req1)
    prefill_mgr.add_request(req2)
    
    # Prefill batch
    batch = prefill_mgr.schedule()
    assert len(batch.requests) == 2
    
    # Simulate prefill completion
    for req in batch.requests:
        req.cached_len = len(req.prompt_tokens)
        req.output_tokens = [10]
        decode_mgr.add_request(req)
    
    # Decode batch
    batch = decode_mgr.schedule()
    assert len(batch.requests) == 2
    assert all(req.cached_len > 0 for req in batch.requests)
```

**Performance Characteristics**:

```
Prefill:
  - Small batch size (2-8 requests)
  - Many tokens per request (100-1000)
  - Compute-bound (matrix multiplication)
  - Optimization: Minimize memory writes

Decode:
  - Large batch size (32-256 requests)
  - One token per request
  - Memory-bound (KV cache loads)
  - Optimization: Maximize bandwidth utilization
```

Phase 2 establishes specialized scheduling for each execution phase—the key to efficient serving.

## 16.13 Model Executor: MLIR Integration

The executor wraps the GPT model from Chapter 14, providing a clean interface for prefill and decode execution.

**Model Configuration**:

```python
# python/executor.py
from dataclasses import dataclass
import numpy as np

@dataclass
class ModelConfig:
    """GPT model configuration"""
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    
    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head
```

**Executor Implementation**:

```python
class ModelExecutor:
    """Executes model forward passes
    
    Integrates with MLIR-compiled GPT model from Chapter 14.
    Handles both prefill and decode batches.
    """
    
    def __init__(self, config: ModelConfig, weights: dict, kv_pool):
        """
        Args:
            config: Model configuration
            weights: Model weights dictionary
            kv_pool: KV cache pool instance
        """
        self.config = config
        self.weights = weights
        self.kv_pool = kv_pool
        
        # Initialize model (MLIR JIT compilation from Ch.14)
        self.model = self._build_model()
    
    def _build_model(self):
        """Build and JIT-compile GPT model"""
        # This would use Chapter 14's optimized GPT implementation
        # For Phase 2, we use a simplified version for testing
        return SimplifiedGPT(self.config, self.weights)
    
    def execute_prefill(self, batch) -> np.ndarray:
        """Execute prefill forward pass
        
        Args:
            batch: Prefill batch with multiple tokens per request
            
        Returns:
            Logits for last position of each request [batch_size, vocab_size]
        """
        # Forward pass through model
        logits = self.model.forward(
            input_ids=batch.input_ids,
            positions=batch.positions,
            kv_cache=self.kv_pool
        )
        
        # Extract logits for last token of each request
        request_logits = []
        offset = 0
        
        for req in batch.requests:
            req_len = req.extend_len
            # Get logits for last token
            last_logits = logits[offset + req_len - 1]
            request_logits.append(last_logits)
            offset += req_len
        
        return np.array(request_logits)
    
    def execute_decode(self, batch) -> np.ndarray:
        """Execute decode forward pass
        
        Args:
            batch: Decode batch with one token per request
            
        Returns:
            Logits for each request [batch_size, vocab_size]
        """
        # Forward pass (one token per request)
        logits = self.model.forward(
            input_ids=batch.input_ids,
            positions=batch.positions,
            kv_cache=self.kv_pool
        )
        
        return logits
```

**Simplified Model** (for testing without full MLIR stack):

```python
class SimplifiedGPT:
    """Simplified GPT for testing executor integration
    
    In production, this would be the MLIR-JIT compiled model
    from Chapter 14. For Phase 2 testing, we use random logits.
    """
    
    def __init__(self, config: ModelConfig, weights: dict):
        self.config = config
        self.weights = weights
    
    def forward(self, input_ids: np.ndarray, positions: np.ndarray, 
                kv_cache) -> np.ndarray:
        """
        Args:
            input_ids: Token IDs [num_tokens]
            positions: Position indices [num_tokens]
            kv_cache: KV cache pool
            
        Returns:
            Logits [num_tokens, vocab_size]
        """
        num_tokens = len(input_ids)
        vocab_size = self.config.vocab_size
        
        # For testing: return random logits
        # Real implementation uses MLIR-compiled transformer
        np.random.seed(input_ids[0])  # Deterministic for testing
        logits = np.random.randn(num_tokens, vocab_size).astype(np.float32)
        
        return logits
```

**Usage Example**:

```python
# Initialize
config = ModelConfig(vocab_size=256, n_layer=4, n_head=4, n_embd=128)
weights = {}  # Load from checkpoint
kv_pool = KVCachePool(num_pages=64, page_size=16, 
                     num_layers=4, num_heads=4, head_dim=32)

executor = ModelExecutor(config, weights, kv_pool)

# Prefill batch
req = Request(0, prompt_tokens=[1, 2, 3, 4, 5], max_tokens=10)
batch = Batch.from_prefill([req])

logits = executor.execute_prefill(batch)
print(f"Prefill output: {logits.shape}")  # [1, 256]

# Decode batch
req.cached_len = 5
req.output_tokens = [10]
batch = Batch.from_decode([req])

logits = executor.execute_decode(batch)
print(f"Decode output: {logits.shape}")  # [1, 256]
```

## 16.14 Integration Testing and Data Flow

Phase 2 components work together through well-defined interfaces. Integration tests validate the complete data flow.

**Data Flow Diagram**:

```
User Request
     │
     ▼
Request Object (Python)
     │
     ▼
PrefillManager.add_request()
     │
     ▼
PrefillManager.schedule() → Batch
     │
     ▼
ModelExecutor.execute_prefill()
     │ (crosses Python/C++ boundary)
     ▼
MLIR Model Forward Pass (C++)
     │
     ▼
KV Cache Write (C++)
     │
     ▼
Return Logits to Python
     │
     ▼
Sample Next Token
     │
     ▼
DecodeManager.add_request()
     │
     ▼
DecodeManager.schedule() → Batch
     │
     ▼
ModelExecutor.execute_decode()
     │
     ▼
... (repeat decode until finished)
```

**Integration Test**:

```python
# test_phase2_integration.py
def test_end_to_end_prefill_decode():
    """Test complete prefill → decode flow"""
    
    # Setup
    config = ModelConfig(vocab_size=256, n_layer=2, n_head=4, n_embd=64)
    kv_pool = KVCachePool(64, 16, 2, 4, 16)
    executor = ModelExecutor(config, {}, kv_pool)
    
    prefill_mgr = PrefillManager(token_budget=100)
    decode_mgr = DecodeManager(max_batch_size=32)
    
    # Add requests
    req1 = Request(0, [1, 2, 3, 4], max_tokens=3)
    req2 = Request(1, [5, 6, 7], max_tokens=3)
    
    prefill_mgr.add_request(req1)
    prefill_mgr.add_request(req2)
    
    # Prefill phase
    prefill_batch = prefill_mgr.schedule()
    assert prefill_batch is not None
    
    logits = executor.execute_prefill(prefill_batch)
    assert logits.shape == (2, 256)  # 2 requests
    
    # Sample first token
    for i, req in enumerate(prefill_batch.requests):
        token = np.argmax(logits[i])
        req.output_tokens.append(token)
        req.cached_len = len(req.prompt_tokens)
        decode_mgr.add_request(req)
    
    # Decode phase (generate remaining tokens)
    for step in range(2):  # 2 more tokens to reach max_tokens=3
        decode_batch = decode_mgr.schedule()
        assert decode_batch is not None
        assert len(decode_batch.requests) == 2
        
        logits = executor.execute_decode(decode_batch)
        assert logits.shape == (2, 256)
        
        for i, req in enumerate(decode_batch.requests):
            token = np.argmax(logits[i])
            req.output_tokens.append(token)
            req.cached_len += 1
            
            if len(req.output_tokens) >= req.max_tokens:
                req.is_finished = True
        
        decode_mgr.remove_finished()
    
    # Verify completion
    assert len(req1.output_tokens) == 3
    assert len(req2.output_tokens) == 3
    assert req1.is_finished
    assert req2.is_finished

def test_memory_management():
    """Test KV cache allocation/deallocation across lifecycle"""
    config = ModelConfig(vocab_size=256, n_layer=2, n_head=4, n_embd=64)
    kv_pool = KVCachePool(16, 16, 2, 4, 16)  # Limited memory
    
    initial_free = kv_pool.get_num_free_pages()
    
    # Allocate pages for request
    req = Request(0, [1, 2, 3, 4, 5], max_tokens=5)
    pages_needed = kv_pool.pages_needed(len(req.prompt_tokens))
    req.kv_pages = kv_pool.allocate(pages_needed)
    
    assert kv_pool.get_num_free_pages() == initial_free - pages_needed
    
    # Free pages when request finishes
    kv_pool.free(req.kv_pages)
    
    assert kv_pool.get_num_free_pages() == initial_free
```

**Performance Characteristics** (Phase 2):

```python
# Benchmark
def benchmark_phase2():
    config = ModelConfig(vocab_size=256, n_layer=4, n_head=4, n_embd=128)
    kv_pool = KVCachePool(256, 16, 4, 4, 32)
    executor = ModelExecutor(config, {}, kv_pool)
    
    # Create 32 requests (avg 50 tokens prompt, 20 tokens generation)
    requests = [
        Request(i, list(range(50)), max_tokens=20)
        for i in range(32)
    ]
    
    prefill_mgr = PrefillManager(token_budget=512)
    decode_mgr = DecodeManager(max_batch_size=32)
    
    for req in requests:
        prefill_mgr.add_request(req)
    
    total_tokens = 0
    steps = 0
    
    # Serving loop
    while not prefill_mgr.is_empty() or not decode_mgr.is_empty():
        # Prefill
        prefill_batch = prefill_mgr.schedule()
        if prefill_batch:
            executor.execute_prefill(prefill_batch)
            for req in prefill_batch.requests:
                req.cached_len = len(req.prompt_tokens)
                req.output_tokens.append(0)
                decode_mgr.add_request(req)
            total_tokens += sum(r.extend_len for r in prefill_batch.requests)
        
        # Decode
        decode_batch = decode_mgr.schedule()
        if decode_batch:
            executor.execute_decode(decode_batch)
            for req in decode_batch.requests:
                req.output_tokens.append(0)
                req.cached_len += 1
                if len(req.output_tokens) >= req.max_tokens:
                    req.is_finished = True
            decode_mgr.remove_finished()
            total_tokens += len(decode_batch.requests)
        
        steps += 1
    
    print(f"Total tokens: {total_tokens}")
    print(f"Steps: {steps}")
    print(f"Tokens/step: {total_tokens/steps:.1f}")
```

## 16.15 Summary and Looking Ahead

Chapter 16 Part 2 built the core infrastructure components that enable sophisticated LLM serving, demonstrating how production systems separate concerns between high-level orchestration and performance-critical execution.

**Foundation Components**. We established the data model with Request and Batch abstractions that track user tasks through their lifecycle, distinguishing between prefill batches (computing initial KV cache from prompts) and decode batches (generating tokens autoregressively). These abstractions hide the complexity of multi-request coordination behind clean interfaces, making the scheduler implementation straightforward.

**Memory Management**. The C++ KV cache pool implements paged allocation, allowing non-contiguous memory blocks to serve each request's cache needs. This eliminates the fragmentation inherent in contiguous allocation schemes, where memory must be pre-allocated for maximum possible sequence length. Pybind11 bindings expose the C++ memory pool to Python, enabling high-level code to allocate and manage pages without sacrificing performance. The page table indirection adds minimal overhead compared to the memory savings from reduced fragmentation.

**Scheduling Infrastructure**. The PrefillManager implements FCFS scheduling with token budget constraints, preventing long prefill operations from monopolizing compute resources. The DecodeManager batches all running requests together for each generation step, maximizing hardware utilization during the decode phase. The ModelExecutor wraps the MLIR-compiled GPT model, providing a unified forward() interface that handles both prefill and decode execution paths. This separation of concerns allows each component to optimize for its specific workload pattern.

**Architectural Patterns**. The Python-C++ language split reflects a fundamental tradeoff in ML systems engineering. Python provides flexibility for complex scheduling logic, request lifecycle management, and algorithm experimentation. C++ delivers performance for memory-intensive operations and numerical computation. The clean interface boundaries—Batch abstractions, executor APIs, memory pool interfaces—make this split maintainable. Each component can be tested independently before integration, following standard software engineering practices.

**Looking to Part 3**. The infrastructure established here enables advanced serving techniques. Chunked prefill will interleave long prefill operations with decode steps, ensuring fairness when requests have vastly different prompt lengths. Radix cache will automatically detect shared prefixes across requests, eliminating redundant computation when multiple users submit prompts with common beginnings (like system prompts in chat applications). Continuous batching will allow dynamic request addition and removal, maintaining high throughput even as workload changes. The NanoServingEngine will integrate all components into a cohesive API suitable for demonstration and teaching.

This progression from basic infrastructure to advanced optimizations mirrors real ML systems development. Build solid foundations—data models, memory management, basic scheduling—before adding sophisticated features. Test components independently before integration. Separate high-level logic from performance-critical code. These patterns apply beyond LLM serving to any ML system requiring complex orchestration: reinforcement learning environments, distributed training coordinators, or multi-task inference servers. MLIR's JIT compilation and clean C++ interoperability enable this architectural approach, providing both Python's expressiveness and C++'s performance where each matters most.