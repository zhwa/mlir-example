# Chapter 16 Part 2: Nano-Serving Implementation

Part 1 introduced production LLM serving concepts—paged KV cache, continuous batching, radix cache, chunked prefill. These algorithmic techniques enable 100-1000× speedups in real systems like vLLM and SGLang. Part 2 demonstrates **our specific implementation** combining MLIR compiler acceleration with Python inference orchestration.

This chapter builds nano-serving in six progressive phases, covering both **core components** (Phases 0-2) and **advanced features** (Phases 3-6). By the end, you'll understand how Python scheduling coordinates with C++ model execution through clean interfaces, and how production systems achieve performance through algorithmic innovation.

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
    k_caches: List[np.ndarray] = None  # Actual K tensors per layer
    v_caches: List[np.ndarray] = None  # Actual V tensors per layer
    
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
        positions = [req.total_len for req in requests]
        
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
PYBIND11_MODULE(ch16, m) {
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

## 16.15 Phases 0-2 Summary

The first three phases established the core infrastructure for LLM serving by separating high-level orchestration (Python) from performance-critical execution (C++).

**Foundation Built**: Request and Batch abstractions model the serving lifecycle, distinguishing prefill (computing KV cache from prompts) from decode (autoregressive generation). These clean interfaces hide complexity from schedulers. The C++ KV cache pool implements paged allocation via free lists, eliminating memory fragmentation. Pybind11 bindings expose C++ performance to Python flexibility. PrefillManager implements FCFS scheduling with token budget constraints, while DecodeManager batches all running requests for maximum parallelism. ModelExecutor wraps MLIR-compiled models, providing unified forward() interfaces for both phases.

**Architectural Insights**: The Python-C++ split reflects fundamental engineering tradeoffs. Python handles complex scheduling logic, request state machines, and algorithm experimentation. C++ delivers performance for memory operations and numerical computation. Clean boundaries—batch abstractions, executor APIs, memory pool interfaces—make this maintainable. Each component tests independently before integration, following standard software practices. The separation of concerns allows optimization at each layer: high-level algorithms improve in Python; low-level kernels optimize in C++.

**Next: Advanced Optimizations**: Phases 3-6 build on this foundation with sophisticated serving techniques. Chunked prefill will interleave long prompt processing with decode steps, ensuring fairness when request sizes vary dramatically. Radix cache will automatically detect shared prefixes across requests (like system prompts in chatbots), eliminating redundant computation through tree-based lookup. Continuous batching will enable dynamic request admission/removal, maintaining high throughput as workload changes. NanoServingEngine will integrate everything into a cohesive API demonstrating production-quality serving patterns.

## 16.16 Phase 3: Chunked Prefill for Fair Scheduling

### 16.16.1 The Long Prompt Starvation Problem

Interactive LLM serving faces a fundamental scheduling challenge: **long prompts monopolize GPU resources**, causing short prompts to wait unacceptably long. This creates poor user experience where small queries (chatbot messages, quick questions) experience high latency due to large context processing (document analysis, multi-shot examples) ahead in the queue.

**The Problem Visualized**:

```
Time →
┌────────────────────────────────────────┐
│  Req 0: 2000 tokens (200ms)            │ ← GPU 100% busy
└────────────────────────────────────────┘
                                         │
                        Req 1: 50 tokens (5ms) waits 200ms!  ← 40× latency penalty
                        Req 2: 100 tokens (10ms) waits 205ms! ← 20× latency penalty
```

**Naive FCFS (First-Come-First-Served) scheduling** processes requests sequentially. A single 2000-token prompt (200ms prefill on modern GPUs) forces all subsequent requests to wait, regardless of their size. This violates the principle of **fairness**: small requests should not be penalized by large requests ahead in the queue.

**Key Insight**: Prefill computation is **divisible**—we can split it into smaller chunks and interleave execution. Unlike indivisible operations (single matrix multiplication), attention computation over 2000 tokens can be broken into 4× 500-token chunks without sacrificing correctness. The KV cache for tokens 0-499 remains valid when computing tokens 500-999.

### 16.16.2 Chunked Prefill: Progressive Computation

**Chunked prefill** transforms long prompts into multiple small scheduling units:

**Algorithm**: Break prompt into fixed-size chunks (typically 256-512 tokens)

```
Original request: [2000 tokens] → [200ms prefill]

Chunked execution:
  Chunk 0: [256 tokens] → [26ms]
  Chunk 1: [256 tokens] → [26ms]
  Chunk 2: [256 tokens] → [26ms]
  ...
  Chunk 7: [232 tokens] → [24ms]
```

**Scheduling Strategy**: Round-robin across all active requests

```
Time →
┌──────┐
│ Req 0│ Chunk 0 (256 tokens, 26ms)
└──────┘
       ┌─┐
       │1│ Complete prefill (50 tokens, 5ms)
       └─┘
          ┌──────┐
          │ Req 0│ Chunk 1 (256 tokens, 26ms)
          └──────┘
                 ┌──┐
                 │ 2│ Complete prefill (100 tokens, 10ms)
                 └──┘
                    ┌──────┐
                    │ Req 0│ Chunk 2 (256 tokens, 26ms)
                    └──────┘
```

The improvement is substantial. Request 1 now starts processing after just 26ms instead of waiting the full 200ms for request 0 to complete. This represents meaningful latency reduction for interactive workloads. More importantly, every request makes progress proportional to its actual work requirement rather than being blocked by whatever happens to be ahead in the queue. Short requests complete quickly even when large context processing requests are present, maintaining system responsiveness across mixed workloads.

### 16.16.3 Fairness vs Throughput Trade-off

Chunking introduces overhead that must be balanced against its fairness benefits. Context switching between chunks requires GPU state saves and restores, though modern GPUs handle this efficiently. Each chunk must recompute attention over all previously cached tokens, adding computational cost. Smaller chunks also reduce effective batch size at any given moment, which can lower GPU utilization compared to processing large contiguous sequences.

The choice of chunk size involves fundamental tradeoffs. Smaller chunks (256-512 tokens) prioritize fairness and reduce tail latency, making them suitable for interactive workloads like chatbots where users expect quick responses. Larger chunks (1024+ tokens) minimize overhead and maximize throughput, fitting batch-oriented workloads like document processing where total completion time matters more than individual request latency. Production systems often adjust chunk size dynamically based on current queue depth and mix of request sizes, using smaller chunks when many short requests are waiting and larger chunks when processing homogeneous workloads.

### 16.16.4 Implementation Considerations

Production implementations must handle several practical concerns. The KV cache grows incrementally as chunks process—after the first chunk completes, the cache contains entries for those tokens; after the second chunk, entries for both chunks exist. Each chunk's attention computation uses all previously cached entries plus the new chunk's tokens. Request state tracking becomes more complex since the system must know how many tokens each request has already cached and which chunk to process next. The scheduler queries this state to determine appropriate work units for the current iteration.

**Core Chunking State** (from [`chunked_request.py`](../ch.16.Nano-Serving/python/chunked_request.py)):

```python
class ChunkedRequest:
    """Tracks progress through a long prompt"""
    def __init__(self, request: Request, chunk_size: int = 512):
        self.request = request
        self.chunk_size = chunk_size
        self.tokens_processed = request.cached_len  # Start from cached
    
    @property
    def has_more_chunks(self) -> bool:
        """Check if there are more chunks to process"""
        return self.tokens_processed < len(self.request.prompt_tokens)
    
    def get_next_chunk(self) -> List[int]:
        """Get next chunk of tokens to process"""
        start = self.tokens_processed
        end = min(start + self.chunk_size, len(self.request.prompt_tokens))
        return self.request.prompt_tokens[start:end]
```

The key insight: each request maintains `tokens_processed` counter, enabling resumption from any point. The scheduler simply iterates through chunked requests, processing one chunk per iteration until `has_more_chunks` returns false.

Round-robin scheduling is simple but not optimal for all scenarios. Production systems often use priority queues where each request receives a score based on estimated completion time divided by chunk size, balancing fairness for small requests with throughput for large ones. This prevents both starvation of large requests by many small requests and blocking of small requests by large contexts. Prefill and decode phases can coexist in the same serving loop—while chunking processes a new request's prompt, the system simultaneously generates output tokens for requests already in decode phase, as we'll see in Section 16.18.

**Connection to ch.14 Implementation**: Our [`generation.py`](../ch.14.GPT-Optimized/generation.py#L45-L120) uses `generate_cached()` with full prefill, suitable for demonstration. Production systems extend this with chunk iteration and scheduling policies shown above.

## 16.17 Phase 4: Radix Cache for Automatic Prefix Sharing

### 16.17.1 The Redundant Computation Problem

Real-world LLM workloads exhibit substantial prefix overlap. Chatbot applications prepend the same system prompt to every user query, consuming dozens to hundreds of tokens that remain constant across requests. Few-shot learning scenarios include identical example demonstrations before each prompt, often spanning hundreds of tokens. Document question-answering systems attach the same large document context to multiple different questions, sometimes consuming thousands of tokens that differ only in the final query portion.

Naive KV caching treats each request independently, computing attention for all tokens from scratch. When one hundred requests share a five-hundred-token system prompt, this approach computes the same five hundred KV vectors one hundred times, wasting enormous GPU cycles on redundant computation. The key insight is that token sequences naturally form a prefix tree structure. If we organize the KV cache as a radix tree, shared prefixes need only be computed once. All requests following the same initial token path can reuse those cached entries, with branching occurring only where requests diverge.

### 16.17.2 Radix Tree Data Structure

**Tree Structure**: Each node represents a token position, edges represent token values

```
Example requests:
  Request A: [1, 2, 3, 4, 5] (system prompt + query 1)
  Request B: [1, 2, 3, 6, 7] (system prompt + query 2)
  Request C: [1, 2, 8, 9]    (system prompt prefix + different continuation)

Radix tree:
                    [root]
                      |
                    token=1
                      |
                    token=2
                    /   \
                token=3  token=8
                /   \       \
            token=4 token=6  token=9
               |       |
            token=5 token=7

Shared path [1→2]: KV cache computed once, used by all requests
Branching at token=2: Different continuations store separate KV cache
```

Each radix tree node represents a single token position in the sequence and stores the corresponding KV cache pages in physical memory. The node maintains a map from token IDs to child nodes, enabling fast traversal when looking up prefixes. Reference counting tracks how many active requests currently use this node—when a request begins processing, it increments the reference count for all nodes along its path; when the request completes, it decrements them. The last access timestamp supports LRU eviction by identifying which branches of the tree have gone unused for the longest time.

**Core Node Structure** (from [`radix_node.py`](../ch.16.Nano-Serving/python/radix_node.py)):

```python
class RadixNode:
    """Node in radix tree for prefix sharing"""
    def __init__(self, token: Optional[int] = None):
        self.token = token
        self.children: Dict[int, 'RadixNode'] = {}  # token_id -> child
        self.kv_pages: List[int] = []               # Physical pages
        self.ref_count: int = 0                     # Active requests
        self.last_access_time: float = time.time()  # For LRU
    
    def get_child(self, token: int) -> Optional['RadixNode']:
        return self.children.get(token)
    
    def add_child(self, token: int) -> 'RadixNode':
        if token not in self.children:
            self.children[token] = RadixNode(token)
        return self.children[token]
```

The simplicity is deliberate: nodes are lightweight containers connecting tree structure (via `children` dict) to physical memory (via `kv_pages` list). The entire radix cache algorithm builds on these basic operations.

### 16.17.3 Cache Lookup Algorithm

When a new request arrives with tokens `[t1, t2, t3, ...]`, walk the tree to find the longest prefix match:

**Prefix Matching Logic** (from [`radix_cache.py`](../ch.16.Nano-Serving/python/radix_cache.py)):

```python
def match_prefix(self, tokens: List[int]) -> Tuple[int, RadixNode]:
    """Find longest matching prefix in tree
    
    Returns:
        (matched_length, last_matched_node)
    """
    node = self.root
    matched_len = 0
    
    for i, token in enumerate(tokens):
        child = node.get_child(token)
        if child is not None:
            node = child              # Advance in tree
            matched_len = i + 1       # Count matched tokens
            node.update_access_time() # Update LRU
        else:
            break  # No match, stop here
    
    return matched_len, node
```

**Example execution**:
- Tree contains path [1, 2, 3]
- New request: [1, 2, 3, 6, 7]
- Lookup walks: root → [1] → [2] → [3] → (no child for 6)
- Returns: `matched_length=3`, reuse KV cache for tokens [1,2,3]
- Need to compute: tokens [6, 7] starting from node representing [1,2,3]

The algorithm is optimal: $O(n)$ where $n$ is the shorter of (token sequence length, tree depth). No backtracking, no hash lookups—pure tree traversal.

### 16.17.4 Memory Management and Eviction

**Physical Memory Layout**: KV cache stored in **pages** (e.g., 256 tokens per page). Each radix node holds references to physical pages in the KV pool.

**Problem**: Tree grows unbounded as requests add new paths. GPU memory is limited (40GB on A100, stores ~100K tokens with Llama-70B).

**Solution**: **LRU eviction** removes unused branches when memory pressure occurs:

1. **Eviction candidates**: Nodes with `ref_count == 0` (no active requests)
2. **LRU scoring**: Sort candidates by `last_access_time`
3. **Eviction process**: 
   - Remove leaf nodes first (preserve shared prefixes)
   - Free KV pages back to pool
   - Update parent node's children map

The eviction policy preserves hot paths while removing cold ones. Frequently accessed sequences like system prompts remain in the cache because their access timestamps constantly update. One-time queries with unique continuations become eviction candidates once their reference counts drop to zero. The leaf-first removal strategy ensures shared prefixes near the tree root survive longer than unique continuations near the leaves, maintaining the most valuable cached state.

The memory efficiency gain can be dramatic. Consider one hundred requests sharing a five-hundred-token prefix. Without radix caching, the system stores fifty thousand token entries redundantly—each request maintains its own copy of the identical prefix. With radix caching, only five hundred tokens need storage, shared across all requests. This represents a ninety-nine percent reduction in memory consumption for the shared portion.

### 16.17.5 Integration with Request Lifecycle

During prefill, the system first looks up the request's token sequence in the radix tree to find the longest matching prefix. It reuses the existing KV cache entries for all matched tokens and computes new entries only for the unmatched portion. As computation proceeds, the system inserts new nodes into the tree representing the unique continuation of this request and increments reference counts for all nodes along the path, indicating an active request depends on them.

The decode phase extends the cached sequence with each generated token. As the model produces new tokens, the system creates corresponding child nodes in the radix tree and stores their KV vectors. If multiple requests share the same generation path (rare but possible in beam search or when temperature is zero), they share these decode-phase nodes as well.

When a request completes, cleanup involves decrementing reference counts for all nodes in the request's path. Nodes reaching zero reference count become eligible for eviction—no active request depends on them anymore. The LRU eviction process runs periodically, examining evictable nodes and freeing the least recently accessed ones when memory pressure requires it.

### 16.17.6 Implementation Notes

Since the radix cache is shared across all concurrent requests, thread safety becomes critical. Read operations like prefix lookups can occur simultaneously without coordination—multiple threads traverse the tree to find their matching prefixes in parallel. Write operations like inserting new nodes or performing eviction require exclusive access to prevent corruption. This read-write locking pattern allows high concurrency for the common case (lookups) while ensuring consistency for modifications.

Page allocation performance matters since every token needs KV cache storage. Free list data structures enable fast allocation by maintaining a pool of available pages. Pre-allocating the entire page pool at system startup eliminates malloc calls during serving, preventing latency spikes from memory allocation. Some implementations cache hash values for token sequences at each node, enabling quick rejection of non-matching paths through hash comparison before committing to deep tree traversal.

**Connection to ch.14**: Our [KV caching implementation](../ch.14.GPT-Optimized/generation.py#L45-L75) uses simple array-based cache per request. Production radix cache extends this by sharing cache across requests with tree-based lookup shown above.

## 16.18 Phase 5: Continuous Batching

### 16.18.1 The Static Batching Bottleneck

Traditional batch serving waits for **all requests to complete** before accepting new work. This creates GPU idle time when early-finishing requests complete but the batch continues running for slower requests.

**Static Batch Problem**:

```
Batch of 8 requests:
┌────┐ Finished at 20ms ─┐
├────┤ Finished at 25ms  │
├────┤ Finished at 30ms  │
├────┤ Finished at 35ms  │  GPU still processing
├────┤ Finished at 40ms  │  remaining requests...
├─────┤ Finished at 55ms  │
├──────┤ Finished at 70ms │
└─────────────┘ Finished at 150ms ← Entire batch waits!

New requests arrive at 25ms, 30ms, 40ms...
All wait 150ms for batch to complete!
```

**Key Insight**: LLM decode generates **one token per request per iteration**. When a request finishes (reaches max_tokens or EOS), we can immediately replace it with a waiting request **without breaking the batch**. Each iteration is independent—the batch can change composition dynamically.

### 16.18.2 Continuous Batching Lifecycle

**Request States**: Waiting → Running → Finished

**Dynamic Batch Formation**: Each iteration reconstructs the batch:

```
Iteration t:
┌───────────────────────────────┐
│ Running: [Req 0, 1, 2, 3, 4]  │ 5 requests
└───────────────────────────────┘
         ↓ Generate tokens
Iteration t+1:
- Req 2 finished (hit EOS)
- Req 5 and Req 6 added from waiting queue
┌─────────────────────────────────────┐
│ Running: [Req 0, 1, 3, 4, 5, 6]     │ 6 requests
└─────────────────────────────────────┘
         ↓ Generate tokens
Iteration t+2:
- Req 0, 3 finished
- Req 7 added (only 1 waiting)
┌─────────────────────────────────────┐
│ Running: [Req 1, 4, 5, 6, 7]        │ 5 requests
└─────────────────────────────────────┘
```

This dynamic composition delivers several advantages. The GPU batch size stays near its maximum configured value since finished requests immediately give way to waiting ones. New requests begin processing as soon as capacity exists rather than waiting for an entire batch cycle to complete. Most importantly, the GPU never idles while requests remain in the queue—there are no wasted cycles waiting for the slowest request in a static batch to finish.

### 16.18.3 Scheduling Policies

Production scheduling requires multiple cooperating policies. Admission control limits the running batch size to prevent out-of-memory errors—the system must refuse new requests when adding them would exceed available GPU memory. Priority scheduling allows differentiation between request classes, perhaps giving paid users faster service than free tier users. Preemption extends this further by allowing high-priority requests to evict currently running low-priority ones, with the evicted requests returning to the waiting queue while their partial KV cache remains intact in the radix cache, preserving their progress. Fairness policies track how much service each request has received, ensuring all requests eventually make progress and preventing pathological cases where many small requests starve a few large ones or vice versa.

### 16.18.4 Memory Management Integration

Tight memory control integration is essential for continuous batching. Before admitting a new request, the system estimates required memory by calculating tokens needed (prompt length plus maximum generation length minus any cached prefix) and converting to page count. It then checks whether the KV pool has sufficient free pages. If not, the system attempts LRU eviction from the radix cache to free space. Should eviction prove insufficient, the system either rejects the request with backpressure or preempts lower-priority running requests. All allocation must be atomic—either the request receives all needed memory or none, preventing partial state that could cause inconsistencies.

**Memory Estimation Math** (from [`kv_pool.py`](../ch.16.Nano-Serving/python/kv_pool.py) wrapper):

```python
def pages_needed(self, num_tokens: int) -> int:
    """Calculate pages needed for num_tokens
    
    With page_size=16:
        15 tokens → 1 page
        16 tokens → 1 page  
        17 tokens → 2 pages
    """
    return (num_tokens + self.page_size - 1) // self.page_size

def can_allocate(self, num_tokens: int) -> bool:
    """Check if allocation would succeed"""
    pages = self.pages_needed(num_tokens)
    return self.get_num_free_pages() >= pages
```

Before admitting a request:
1. Calculate: `new_tokens = len(prompt) + max_gen - cached_len`
2. Convert: `pages = (new_tokens + page_size - 1) // page_size` (ceiling division)
3. Check: `free_pages >= pages` (atomic check + allocate)
4. If insufficient: try eviction, then reject/preempt

The ceiling division ensures we never under-allocate. Example: 17 tokens with 16-token pages needs 2 pages, not 1.06.

Batch size adapts dynamically to memory pressure. When abundant free memory exists (above fifty percent free), the system maximizes batch size to fully utilize the GPU. As memory pressure increases to moderate levels (twenty to fifty percent free), the system reduces batch size somewhat while increasing eviction frequency. Under high memory pressure (below twenty percent free), the system becomes conservative with small batches and aggressive eviction, prioritizing stability over throughput.

### 16.18.5 Integration with Prefill and Decode

The complete serving loop integrates all three phases. Each iteration begins by scheduling prefill chunks for new or partially-processed requests. If any chunks are ready, the system executes them and transitions completed prefills to decode phase. Next comes decode scheduling for all running requests—the system generates one token per request, removes those that have finished, and admits waiting requests to fill available batch capacity. Finally, memory management runs, checking pressure levels and triggering radix cache eviction or considering preemption when necessary.

**Continuous Batching Main Loop** (simplified from [`continuous_batcher.py`](../ch.16.Nano-Serving/python/continuous_batcher.py)):

```python
class ContinuousBatcher:
    def step(self) -> int:
        """Execute one batching iteration"""
        tokens_generated = 0
        
        # Phase 1: Schedule prefill chunks
        prefill_batch = self.prefill_mgr.get_next_batch()
        if prefill_batch:
            self.executor.prefill(prefill_batch)
            # Transition completed to decode
            for req in prefill_batch.finished:
                self.request_pool.move_to_running(req)
        
        # Phase 2: Decode running requests
        decode_batch = self.decode_mgr.get_batch(
            self.request_pool.running)
        if decode_batch:
            logits = self.executor.decode(decode_batch)
            for i, req in enumerate(decode_batch.requests):
                token = sample_token(logits[i])  # Sample next token
                req.append_token(token)
                
                if req.is_finished():  # Check EOS or max_tokens
                    self.request_pool.mark_finished(req)
                    tokens_generated += 1
        
        # Phase 3: Admit waiting requests (fill capacity)
        while self.can_admit_more():
            waiting = self.request_pool.get_next_waiting()
            if not waiting:
                break
            self.radix_mgr.allocate_for_request(waiting)
            self.request_pool.move_to_running(waiting)
        
        return tokens_generated
```

The key: **no global synchronization**. Each iteration rebuilds the batch from current state. Finished requests leave, new requests enter, all without blocking.

Prefill and decode interleaving requires balancing latency against overhead. Small prefill chunks around 256 tokens take roughly the same time as several decode iterations, allowing natural interleaving. Large prefill chunks exceeding 1024 tokens typically run exclusively without decode to minimize context switching overhead. Adaptive systems choose chunk size dynamically—when many requests wait in queue, smaller chunks maintain responsiveness; when few requests wait, larger chunks maximize throughput by reducing overhead.

### 16.18.6 Performance Characteristics

Continuous batching substantially outperforms static batching on realistic workloads with heterogeneous request lengths. Static batching must wait for the slowest request to complete before accepting new work, leaving the GPU underutilized as early-finishing requests sit idle. Continuous batching maintains high GPU utilization by immediately backfilling capacity as requests complete. The improvement is most dramatic when request generation lengths vary significantly—static batching wastes cycles proportional to the variance in completion times, while continuous batching absorbs this variance by constantly refreshing the active batch.

**Connection to ch.14**: Our [`generation.py`](../ch.14.GPT-Optimized/generation.py#L80-L120) processes one request at a time (batch size 1). Production systems extend this with request queues, dynamic batching, and the scheduling policies described above.

## 16.19 Phase 6: Complete Integration

### 16.19.1 System Architecture

Production serving engines integrate **four core subsystems**:

```
┌────────────────────────────────────────────┐
│              NanoServingEngine             │
├────────────────────────────────────────────┤
│                                            │
│  ┌───────────────┐  ┌────────────────────┐ │
│  │ Request Queue │──│ Scheduling Policy  │ │
│  │(FIFO/Priority)│  │ - Admission control│ │
│  └───────────────┘  │ - Preemption       │ │
│                     │ - Fairness         │ │
│         │           └────────────────────┘ │
│         ↓                    │             │
│  ┌──────────────┐            ↓             │
│  │ Prefill Mgr  │  ┌────────────────┐      │
│  │ - Chunked    │  │ Decode Manager │      │
│  │ - Round-robin│  │ - Batch decode │      │
│  └──────────────┘  │ - Remove done  │      │
│         │          └────────────────┘      │
│         │                    │             │
│         ↓                    ↓             │
│  ┌─────────────────────────────────────┐   │
│  │         Model Executor              │   │
│  │  - Run transformer forward pass     │   │
│  │  - Coordinate KV cache access       │   │
│  └─────────────────────────────────────┘   │
│         │                    │             │
│         ↓                    ↓             │
│  ┌───────────────┐  ┌────────────────┐     │
│  │ Radix Cache   │  │   KV Pool      │     │
│  │ - Prefix tree │  │ - Paged memory │     │
│  │ - LRU eviction│  │ - Allocation   │     │
│  └───────────────┘  └────────────────┘     │
│                                            │
└────────────────────────────────────────────┘
```

### 16.19.2 Request Dataflow

**Lifecycle from submission to completion**:

```
1. [Client] Submit request (prompt_tokens, max_tokens, temperature)
             ↓
2. [Admission] Check memory availability
   - Estimate tokens_needed = prompt + max_tokens
   - Check if KV pool has capacity
   - Try radix cache prefix match to reduce allocation
             ↓
3. [Prefill Phase]
   Chunk 0 → compute KV cache for tokens 0-255
   Chunk 1 → compute KV cache for tokens 256-511
   ...
   Last chunk → compute remaining tokens
   Generate first output token
             ↓
4. [Decode Phase]
   Iteration 1 → generate token 1, extend KV cache
   Iteration 2 → generate token 2, extend KV cache
   ...
   Iteration N → generate token N (EOS or max_tokens)
             ↓
5. [Completion] Return generated tokens
   - Decrement radix tree ref_counts
   - Mark KV pages as evictable (ref_count == 0)
   - Move to finished queue
```

### 16.19.3 Critical Performance Parameters

Production serving engines expose several critical parameters that control performance characteristics. The maximum batch size fundamentally trades throughput against latency—larger batches increase GPU utilization but may delay individual requests. Chunk size determines fairness granularity, with smaller chunks improving responsiveness at the cost of context switching overhead. Page size affects memory fragmentation, with smaller pages reducing waste but increasing management overhead. Total KV pool capacity limits how many concurrent requests the system can handle. Eviction threshold determines when the system begins freeing cached entries, balancing memory availability against cache hit rates.

These parameters must be tuned for specific workloads. Interactive chatbot applications prioritize low latency, so they typically use moderate batch sizes around 32 requests, small chunk sizes around 256 tokens for responsiveness, and relatively aggressive eviction around twenty percent free memory to maintain cache effectiveness. Batch document processing workloads instead optimize for throughput, using larger batch sizes up to 64 requests, large chunks around 1024 tokens to minimize overhead, and conservative eviction around ten percent free to maximize batch efficiency. Most production deployments fall somewhere between these extremes, requiring careful profiling to find appropriate balance points.

### 16.19.4 Monitoring and Observability

Effective monitoring tracks metrics across three categories. System health metrics reveal GPU utilization percentage, available KV pool pages, radix cache hit rate, and average batch size—these indicate whether the system is using resources effectively. Request metrics measure throughput in tokens per second, latency percentiles across the distribution, queue depth showing waiting requests, and completion rate indicating how quickly the system processes work. Resource usage metrics track GPU memory consumption, page allocation rate showing memory churn, eviction frequency indicating cache pressure, and context switch overhead from chunked prefill.

Alerting should trigger on conditions indicating inefficiency or degradation. Low GPU utilization below seventy percent suggests underutilization that could be addressed by increasing batch size. Low free memory below ten percent indicates pressure requiring either reduced batch size or more aggressive eviction. High tail latency beyond acceptable thresholds suggests chunk size needs adjustment. Low cache hit rates indicate the workload lacks prefix overlap, meaning radix caching provides little benefit.

### 16.19.5 Failure Modes and Mitigation

Several failure modes require specific mitigations. Out-of-memory errors occur when the system cannot allocate KV pages for new requests. The response progression starts with aggressive LRU eviction from the radix cache to free space, then preempts low-priority running requests if necessary, next rejects new admissions with backpressure signals to upstream systems, and finally reduces maximum batch size dynamically to prevent recurrence.

Request starvation happens when some requests wait indefinitely while others process. Mitigation involves priority aging where waiting time gradually increases a request's priority, ensuring even low-priority requests eventually reach the front. Service quantum enforcement guarantees each request receives some minimum processing per scheduling round. The system may also preempt runaway requests that have generated excessive output, preventing them from monopolizing resources.

GPU underutilization where batch size remains below maximum despite waiting requests indicates inefficiency. This often stems from overly conservative admission policies that reject requests unnecessarily. Increasing chunk size reduces the number of small scheduling units competing for slots. Reducing memory reservation overhead by tightening estimates can also help admit more concurrent requests.

### 16.19.6 Deployment Patterns

Multi-GPU deployment follows several patterns depending on model size and throughput requirements. Tensor parallelism splits model layers across multiple GPUs, with each GPU computing a portion of each layer for every request. Pipeline parallelism assigns different transformer layers to different GPUs, with requests flowing through the pipeline. Replication runs independent serving engines on each GPU with external load balancing, maximizing throughput for smaller models that fit on single GPUs.

Request routing strategies affect cache efficiency and load distribution. Sticky routing directs requests with the same system prompt to the same GPU, maximizing radix cache hit rates by concentrating prefix sharing. Load balancing routes each request to the least-loaded engine, optimizing utilization at the cost of cache efficiency. Affinity routing collocates related requests, such as sending all questions about the same document to one engine to maximize cache sharing for that context.

**Connection to ch.14**: Our implementation demonstrates core concepts with educational simplicity. Production systems add HTTP server, request routing, metrics collection, and multi-GPU coordination on top of the serving loop shown in [generation.py](../ch.14.GPT-Optimized/generation.py).

## 16.20 Performance Analysis

Nano-serving demonstrates how each algorithmic technique contributes to serving efficiency.

- **Parallel batching**: Processes multiple requests simultaneously rather than sequentially
- **Paged KV cache**: Eliminates memory fragmentation that wastes GPU capacity
- **Continuous batching**: Dynamically adds/removes requests to maintain high utilization
- **Radix cache**: Reuses computation for shared prefixes when workloads exhibit overlap

The effectiveness of each optimization depends on workload characteristics—batch sizes, prompt lengths, prefix overlap, and generation lengths all influence the relative impact.

**Optimization Breakdown**: Each algorithmic component contributes multiplicatively to overall throughput. Consider 32 requests with 100-token prompts and 20-token generation:

- **Baseline (sequential)**: 32 × 120 = 3,840 computation steps, processing requests one by one
- **+ Parallel batching**: Reduces to ~480 steps through 8-way parallelism, providing the foundation
- **+ KV cache**: Further reduces to ~672 steps (32 prefills + 32×20 decode iterations), eliminating quadratic recomputation
- **+ Continuous batching**: Halves effective steps to ~336 through doubled utilization by filling idle cycles
- **+ Radix cache**: With 60% prefix sharing, drops to ~134 steps, as most prefills reuse cached computation

The cumulative effect yields roughly 30× speedup from combining these techniques. Each optimization addresses a different inefficiency, making them complementary rather than redundant.

**Token Sampling** (from [`continuous_batcher.py`](../ch.16.Nano-Serving/python/continuous_batcher.py)):

```python
def sample_token(logits: np.ndarray, temperature: float = 1.0) -> int:
    """Sample next token from logits
    
    Args:
        logits: Logits for next token [vocab_size]
        temperature: Sampling temperature
            1.0 = neutral, <1 = conservative, >1 = random
    """
    if temperature == 0.0:
        return int(np.argmax(logits))  # Greedy
    
    # Temperature scaling
    logits = logits / temperature
    
    # Softmax: exp(logits) / sum(exp(logits))
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    probs = exp_logits / np.sum(exp_logits)
    
    # Sample from distribution
    return int(np.random.choice(len(probs), p=probs))
```

The temperature parameter controls randomness: $\text{probs}_i = \frac{\exp(\text{logits}_i / T)}{\sum_j \exp(\text{logits}_j / T)}$. Lower $T$ concentrates probability on top tokens (deterministic), higher $T$ flattens distribution (creative).

**Memory Efficiency**: Paged KV cache allocation eliminates the fragmentation inherent in contiguous schemes. For 32 concurrent requests with a maximum sequence length of 2,048 tokens, contiguous allocation pre-reserves memory for worst-case usage, consuming approximately 786 MB regardless of actual token counts. With paged allocation tracking actual usage of 120 tokens per request on average, memory consumption drops to roughly 46 MB—a 17× reduction. This efficiency enables higher concurrency, as GPU memory hosts more concurrent requests without artificial padding.

**Cache Hit Rate Analysis**: Radix cache effectiveness varies dramatically across workload types. Chatbot scenarios with 1000-token system prompts typically achieve 91% hit rates, as nearly all requests share the system prompt. RAG workloads with 500-token document contexts achieve moderate 73% hit rates when multiple queries target the same documents. Code completion with short 100-token context achieves only 45% hit rates due to higher uniqueness in code paths. Understanding these patterns helps predict radix cache benefits for specific deployment scenarios.

Parallel batching processes multiple requests simultaneously rather than sequentially, providing the foundation for all subsequent optimizations. Paged KV cache eliminates memory fragmentation that would otherwise waste GPU capacity, enabling higher concurrency. Continuous batching dynamically adds and removes requests to maintain high utilization regardless of varying completion times. Radix cache reuses computation for shared prefixes when workloads exhibit overlap, with effectiveness depending strongly on actual prefix patterns in the request stream.

The effectiveness of each optimization depends on workload characteristics—batch sizes, prompt lengths, prefix overlap, and generation lengths all influence the relative impact. Workloads with high prefix overlap benefit dramatically from radix caching, while those with uniform request lengths see less benefit from continuous batching. Understanding these dependencies helps tune production systems for specific deployment scenarios.

## 16.21 From Education to Production: Real Inference Systems

Our nano-serving implementation serves purely educational purposes—demonstrating how paged attention, continuous batching, and radix caching work at the algorithmic level. Production LLM serving systems build on these same foundations but add industrial-strength optimizations for real-world deployment.

**vLLM: Production Paged Attention**. Berkeley's vLLM pioneered the application of paged memory management to LLM serving, introducing the PagedAttention algorithm that eliminates KV cache fragmentation. The system uses highly optimized CUDA kernels for attention computation, achieving throughput of 100,000+ tokens per second on modern GPUs. vLLM implements continuous batching in Python for orchestration flexibility while delegating compute-intensive operations to compiled CUDA code. The architecture supports tensor parallelism for distributing large models across multiple GPUs, making it suitable for serving models with 70B+ parameters. vLLM's contribution lies in proving that operating system memory management techniques can dramatically improve LLM serving efficiency.

**SGLang: Structured Generation with Radix Caching**. SGLang extends vLLM's paged attention with sophisticated prefix caching through radix tree data structures. When multiple requests share common prefixes—such as system prompts in chat applications or shared context in RAG systems—SGLang's radix cache automatically detects and reuses the computed KV cache entries. This yields 2-10× speedups on workloads with high prefix overlap. SGLang also introduces structured generation primitives, enabling constrained decoding for JSON output, regex patterns, and grammar-based generation. The system uses FlashInfer, an optimized attention backend that delivers 150,000+ tokens per second throughput. SGLang's innovation demonstrates that caching strategies from traditional systems apply effectively to transformer inference.

**TensorRT-LLM: NVIDIA's Optimized Stack**. NVIDIA's TensorRT-LLM represents the hardware vendor approach, providing deeply integrated optimizations for NVIDIA GPUs. The system implements paged attention and continuous batching entirely in compiled C++/CUDA, eliminating Python overhead in the critical path. TensorRT-LLM leverages tensor cores, custom fused kernels, and NVIDIA's decades of GPU optimization expertise to achieve 300,000+ tokens per second on high-end hardware like H100. The framework supports advanced quantization (INT8, INT4, FP8) for memory bandwidth optimization, in-flight batching for minimal latency overhead, and both tensor and pipeline parallelism for scaling to massive models. TensorRT-LLM excels in deployment scenarios requiring maximum hardware utilization.

**Common Architectural Patterns**. Despite implementation differences, all production systems follow similar architectural principles. They separate orchestration logic (written in high-level languages like Python) from performance-critical execution (implemented in CUDA or compiled C++). All systems implement some form of paged memory management to reduce KV cache fragmentation, though page sizes and management policies vary. Continuous batching appears universally as the scheduling foundation, enabling dynamic workload adaptation. Multi-GPU support through tensor parallelism or pipeline parallelism is standard for large model serving. The systems differ primarily in optimization depth, feature breadth, and integration with specific hardware platforms.

**The Algorithmic Foundation**. Modern LLM serving achieves 100-1000× speedups over naive implementations primarily through algorithmic innovation rather than hardware alone. Paged memory management eliminates the fragmentation that wastes 20-50% of GPU memory in contiguous KV cache schemes. Continuous batching converts latency-bound single-request serving into throughput-optimized multi-request processing. Prefix caching exploits workload patterns to avoid redundant computation. These techniques generalize beyond LLMs to any sequential generation task—reinforcement learning, autoregressive models, beam search, and more. The algorithmic principles remain constant even as hardware evolves.

**Understanding nano-serving prepares you for production systems**. The implementation in this chapter demonstrates each algorithm's core logic without the complexity of GPU programming or distributed systems. Reading vLLM or SGLang source code after understanding our implementation reveals that the fundamental structures—page tables, request schedulers, cache lookup mechanisms—match closely. Production systems add CUDA kernels, multi-GPU coordination, and fault tolerance, but the algorithmic essence remains recognizable. This progression from educational prototype to production system mirrors typical ML systems development: prototype in high-level code to validate algorithms, then optimize critical paths with compiled implementations.

## 16.22 Conclusion: From MLIR Fundamentals to Production Systems

This book traced a complete journey through ML systems engineering, from compiler internals to production deployment. Chapter 1 introduced MLIR's multi-level IR philosophy and execution engine, establishing the foundation for everything that followed. Chapters 2-4 covered dynamic shapes, compilation infrastructure, and bufferization—the essential transformations that bridge functional tensor operations to imperative memory operations. Chapters 5-9 built progressively sophisticated dialects, culminating in production-quality TableGen definitions for custom operations.

The transformer implementation in Chapters 10-14 demonstrated how these compiler techniques enable real neural architectures. We implemented attention mechanisms, optimized GPT inference, and explored the memory management patterns that make modern language models practical. Chapter 15 introduced GPU concepts, preparing the ground for understanding how production systems achieve extreme throughput. Chapter 16 completed the picture by showing how serving systems orchestrate compiled models with advanced scheduling, memory management, and caching strategies.

The architectural patterns recurring throughout this book—separating high-level orchestration from performance-critical execution, using declarative specifications to generate imperative implementations, applying operating system techniques to ML problems—represent the current state of ML systems engineering. These patterns appear in PyTorch, TensorFlow, JAX, and every major ML framework. Understanding them equips you to read production ML systems code, contribute to open-source frameworks, and design your own ML infrastructure.

**Learning Path Forward**. Explore vLLM and SGLang repositories to see production implementations of the algorithms covered here. The concepts are identical; the code differs primarily in optimization depth and hardware targeting. Experiment with MLIR's GPU dialects if hardware acceleration interests you—the patterns from CPU-based transformations apply directly to GPU code generation. Consider contributing to MLIR's ecosystem; the community actively develops new dialects for emerging ML workloads. Read recent research on LLM serving optimizations like speculative decoding, continuous batching variants, and attention algorithm improvements. The field evolves rapidly, but the foundational principles remain stable.

Modern ML systems achieve remarkable performance through the combination of compiler techniques, algorithmic innovation, and systems engineering discipline. MLIR provides the infrastructure for expressing these optimizations systematically, from high-level tensor operations to low-level hardware instructions. The serving systems demonstrate how these compiled models integrate into production environments with sophisticated scheduling and resource management. Together, compiler technology and systems architecture enable the ML applications transforming industries today.

Congratulations on completing this comprehensive exploration of MLIR for machine learning. You've gained the knowledge to understand, extend, and build ML systems at every level of the stack.