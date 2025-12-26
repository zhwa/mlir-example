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

Each request maintains state about prompt processing and generation progress. The key insight is distinguishing between **cached tokens** (already have KV cache computed) and **extend tokens** (need processing this iteration).

**Core State Properties** (from [`request.py`](../ch.16.Nano-Serving/python/request.py)):

```python
class Request:
    """Represents a single generation request"""
    req_id: int
    prompt_tokens: List[int]
    max_tokens: int
    cached_len: int = 0              # Tokens with KV cache
    output_tokens: List[int] = []
    kv_pages: List[int] = []         # Physical pages allocated
    
    @property
    def extend_len(self) -> int:
        """Tokens to process in next forward pass"""
        if not self.output_tokens:
            # Prefill: process remaining prompt
            return len(self.prompt_tokens) - self.cached_len
        else:
            # Decode: only process last token
            return 1
```

The `extend_len` property encapsulates the prefill→decode transition: starts at full prompt length, drops to 1 after first token generated.

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

Batching concatenates multiple requests into single tensors for parallel processing. The key difference between prefill and decode: prefill processes **many tokens per request**, decode processes **one token per request**.

**Batch Construction Pattern** (from [`batch.py`](../ch.16.Nano-Serving/python/batch.py)):

```python
class Batch:
    """Group of requests processed together"""
    requests: List[Request]
    input_ids: np.ndarray       # Token IDs to process
    positions: np.ndarray       # Position indices
    
    @staticmethod
    def from_prefill(requests: List[Request]) -> 'Batch':
        """Prefill: concatenate all prompt tokens"""
        all_tokens = []
        all_positions = []
        
        for req in requests:
            tokens = req.prompt_tokens[req.cached_len:]  # Uncached portion
            positions = range(req.cached_len, len(req.prompt_tokens))
            all_tokens.extend(tokens)
            all_positions.extend(positions)
        
        return Batch(requests, np.array(all_tokens), np.array(all_positions))
    
    @staticmethod
    def from_decode(requests: List[Request]) -> 'Batch':
        """Decode: one token per request"""
        tokens = [req.output_tokens[-1] for req in requests]
        positions = [req.device_len - 1 for req in requests]
        
        return Batch(requests, np.array(tokens), np.array(positions))
```

Prefill batches grow linearly with prompt length; decode batches stay constant size regardless of generation length.

**Prefill vs Decode Batches**:

```python
# Example: Two requests
req1 = Request(req_id=0, prompt_tokens=[1, 2, 3, 4], max_tokens=10)
req2 = Request(req_id=1, prompt_tokens=[5, 6, 7], max_tokens=10)

prefill_batch = Batch.from_prefill([req1, req2])
# input_ids: [1, 2, 3, 4, 5, 6, 7]  (all prompt tokens concatenated)
# positions: [0, 1, 2, 3, 0, 1, 2]   (per-request positions)

# After prefill completes
req1.cached_len = 4; req1.output_tokens = [10]
req2.cached_len = 3; req2.output_tokens = [20]

decode_batch = Batch.from_decode([req1, req2])
# input_ids: [10, 20]        (one token per request)
# positions: [4, 3]          (next position for each)
```

Phase 0 establishes the data model for requests and batches—the foundation for all subsequent phases.

## 16.11 Phase 1: KV Cache Pool (C++ Implementation)

KV cache stores attention keys and values for all previous tokens. Naive allocation (max_seq_len per request) wastes memory. **Paged allocation** divides cache into fixed-size pages, allocating on-demand.

**C++ Implementation** (performance-critical memory management):

The KV cache pool implements paged memory allocation in C++ for performance. The key operations are allocation (find free pages) and deallocation (return pages to free pool).

**Core Paging Algorithm** (from [`kv_cache.cpp`](../ch.16.Nano-Serving/src/kv_cache.cpp)):

```cpp
class KVCachePool {
private:
    int num_pages_, page_size_;
    std::set<int> free_pages_;  // Free page tracking
    std::vector<std::vector<float>> k_cache_, v_cache_;  // Per-layer storage

public:
    std::vector<int> allocate(int num_tokens) {
        // Calculate pages needed (ceiling division)
        int pages_needed = (num_tokens + page_size_ - 1) / page_size_;
        
        if (free_pages_.size() < pages_needed) {
            throw std::runtime_error("Out of KV cache memory");
        }
        
        // Allocate from free pool
        std::vector<int> allocated;
        auto it = free_pages_.begin();
        for (int i = 0; i < pages_needed; i++) {
            allocated.push_back(*it);
            it = free_pages_.erase(it);
        }
        return allocated;
    }
    
    void free(const std::vector<int>& pages) {
        for (int page : pages)
            free_pages_.insert(page);
    }
};
```

The `std::set<int>` provides $O(\log n)$ allocation/deallocation. Production systems optimize with free lists for $O(1)$ operations.

**Python Bindings** via pybind11 (from [`bindings.cpp`](../ch.16.Nano-Serving/src/bindings.cpp)):

```cpp
PYBIND11_MODULE(_nano_serving, m) {
    py::class_<KVCachePool>(m, "KVCachePool")
        .def(py::init<int, int, int, int, int>())
        .def("allocate", &KVCachePool::allocate)
        .def("free", &KVCachePool::free)
        .def("get_num_free_pages", &KVCachePool::getNumFreePages);
}
```

This exposes C++ classes directly to Python with automatic type conversion for vectors and basic types.

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

Prefill scheduling must balance fairness (serve requests in order) with memory constraints (limit batch size). The token budget prevents out-of-memory errors from processing too many long prompts simultaneously.

**FCFS Scheduling Logic** (from [`prefill_manager.py`](../ch.16.Nano-Serving/python/prefill_manager.py)):

```python
class PrefillManager:
    """Schedules prefill with token budget constraint"""
    def __init__(self, token_budget: int = 512):
        self.queue = []  # FIFO queue
        self.token_budget = token_budget
    
    def schedule(self) -> Optional[Batch]:
        """Pack requests into batch up to token budget"""
        selected = []
        total_tokens = 0
        
        # Greedy FCFS packing
        while self.queue:
            req = self.queue[0]
            req_tokens = req.extend_len
            
            if total_tokens + req_tokens <= self.token_budget:
                selected.append(self.queue.pop(0))
                total_tokens += req_tokens
            else:
                break  # Would exceed budget
        
        return Batch.from_prefill(selected) if selected else None
```

The greedy algorithm is optimal for FCFS: serve as many waiting requests as fit in budget, maintaining arrival order.

**Decode Manager** (batch all running requests):

Decode scheduling is simpler than prefill: batch **all running requests** together since each generates exactly one token. The batch size is bounded only by hardware capacity, not token count.

**Decode Batching Logic** (from [`decode_manager.py`](../ch.16.Nano-Serving/python/decode_manager.py)):

```python
class DecodeManager:
    """Batches all running requests for decode"""
    def __init__(self, max_batch_size: int = 256):
        self.running = []
        self.max_batch_size = max_batch_size
    
    def schedule(self) -> Optional[Batch]:
        """Batch all running requests (up to max)"""
        if not self.running:
            return None
        
        batch_reqs = self.running[:self.max_batch_size]
        return Batch.from_decode(batch_reqs)
    
    def remove_finished(self):
        """Remove completed requests"""
        self.running = [r for r in self.running if not r.is_finished]
```

Decode throughput scales linearly with batch size: 256 concurrent requests generate 256 tokens per iteration.

**Integration Pattern**:

```python
# Serving loop coordinates prefill and decode
prefill_mgr = PrefillManager(token_budget=512)
decode_mgr = DecodeManager(max_batch_size=32)

while has_work():
    # Phase 1: Prefill new requests
    if prefill_batch := prefill_mgr.schedule():
        logits = model.forward(prefill_batch)
        for req, logit in zip(prefill_batch.requests, logits):
            req.output_tokens.append(sample(logit))
            req.cached_len = len(req.prompt_tokens)
            decode_mgr.add_request(req)  # Transition to decode
    
    # Phase 2: Decode running requests
    if decode_batch := decode_mgr.schedule():
        logits = model.forward(decode_batch)
        for req, logit in zip(decode_batch.requests, logits):
            req.output_tokens.append(sample(logit))
            req.cached_len += 1
            if len(req.output_tokens) >= req.max_tokens:
                req.is_finished = True
        
        decode_mgr.remove_finished()
```

The loop alternates: process new prompts (prefill) → generate tokens (decode) → repeat. Requests flow from prefill manager to decode manager.

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

**Model Executor**: Wraps the MLIR-compiled GPT model from Chapter 14, providing a unified interface for both prefill and decode execution.

The executor handles the complexity of extracting per-request logits from batched output. Prefill returns many tokens per request but we only need the **last token's logits** for sampling. Decode returns one token per request, so all logits are used.

**Executor Interface** (from [`executor.py`](../ch.16.Nano-Serving/python/executor.py)):

```python
class ModelExecutor:
    """Executes model forward passes"""
    def __init__(self, config, weights, kv_pool):
        self.model = build_mlir_model(config, weights)  # Ch.14 GPT
        self.kv_pool = kv_pool
    
    def execute_prefill(self, batch) -> np.ndarray:
        """Prefill: extract last token logits per request"""
        logits = self.model.forward(batch.input_ids, batch.positions, self.kv_pool)
        
        # Extract last token logits for each request
        result = []
        offset = 0
        for req in batch.requests:
            last_logits = logits[offset + req.extend_len - 1]
            result.append(last_logits)
            offset += req.extend_len
        
        return np.array(result)  # [num_requests, vocab_size]
    
    def execute_decode(self, batch) -> np.ndarray:
        """Decode: return all logits (one per request)"""
        return self.model.forward(batch.input_ids, batch.positions, self.kv_pool)
```

The asymmetry (extracting vs returning all) reflects the prefill/decode token count difference.

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

The boundary between Python and C++ is crossed only during model execution—scheduling, batching, and lifecycle management remain in Python for flexibility.

**Integration Validation**: The test suite verifies end-to-end flow from request creation through prefill→decode transitions to completion. Key invariants: `cached_len` increases monotonically, `extend_len` transitions from prompt length to 1, KV pages are allocated before prefill and freed after completion.

## 16.15 Summary and Part 3 Preview

Chapter 16 Part 2 established the core infrastructure for LLM serving by separating high-level orchestration (Python) from performance-critical execution (C++).

**Foundation Built**: Request and Batch abstractions model the serving lifecycle, distinguishing prefill (computing KV cache from prompts) from decode (autoregressive generation). These clean interfaces hide complexity from schedulers. The C++ KV cache pool implements paged allocation via free lists, eliminating memory fragmentation. Pybind11 bindings expose C++ performance to Python flexibility. PrefillManager implements FCFS scheduling with token budget constraints, while DecodeManager batches all running requests for maximum parallelism. ModelExecutor wraps MLIR-compiled models, providing unified forward() interfaces for both phases.

**Architectural Insights**: The Python-C++ split reflects fundamental engineering tradeoffs. Python handles complex scheduling logic, request state machines, and algorithm experimentation. C++ delivers performance for memory operations and numerical computation. Clean boundaries—batch abstractions, executor APIs, memory pool interfaces—make this maintainable. Each component tests independently before integration, following standard software practices. The separation of concerns allows optimization at each layer: high-level algorithms improve in Python; low-level kernels optimize in C++.

**Looking to Part 3**: Advanced serving techniques build on this foundation. **Chunked prefill** will interleave long prompt processing with decode steps, ensuring fairness when request sizes vary dramatically. **Radix cache** will automatically detect shared prefixes across requests (like system prompts in chatbots), eliminating redundant computation through tree-based lookup. **Continuous batching** will enable dynamic request admission/removal, maintaining high throughput as workload changes. **NanoServingEngine** will integrate everything into a cohesive API demonstrating production-quality serving patterns.

This progression mirrors real ML systems development: build solid foundations before sophisticated features. The patterns established here—data model design, memory management, phase-specific scheduling, cross-language boundaries—apply broadly to any ML system requiring complex orchestration: RL environments, distributed training, multi-task inference servers. MLIR's JIT compilation and clean C++ interoperability enable this architecture, providing Python's expressiveness where logic matters and C++'s performance where speed matters.

**Connection to ch.14**: Our [`generation.py`](../ch.14.GPT-Optimized/generation.py) provides the transformer model that ModelExecutor wraps. Part 2 added serving infrastructure around the model; Part 3 adds advanced algorithmic optimizations for production workloads.