# Chapter 16: Production LLM Serving

Chapters 1-15 built a complete nano GPT model with MLIR compiler acceleration. The implementation demonstrates transformer architecture, KV caching, and parallel execution patterns. However, building a production LLM serving system requires more than a correct model—it demands efficient request scheduling, memory management, and multi-request batching. This chapter teaches production serving techniques used in real-world systems like vLLM, SGLang, and TensorRT-LLM.

**Educational Approach**: This chapter follows textbook style—each concept is immediately followed by its implementation. Theory and practice interleave naturally, building understanding progressively rather than separating abstractions from concrete code. The algorithms presented here are used in production systems worldwide: vLLM pioneered paged attention and continuous batching, SGLang added radix caching for prefix sharing, and TensorRT-LLM provides NVIDIA's optimized stack.

**Why Production Serving is Different**. Research implementations process one request at a time: encode prompt, generate tokens sequentially, return result. Production systems serve thousands of concurrent users with diverse requirements—some need low latency (chat), others need high throughput (batch document summarization). The naive approach fails catastrophically.

Naive implementations process one request at a time, regenerating attention for the entire sequence at each step. This approach suffers from three fundamental inefficiencies. First, the quadratic redundancy problem: each generation step recomputes attention scores for all previous tokens, creating $O(N^2)$ wasted computation where N is the sequence length. Second, the sequential bottleneck: requests queue up waiting for the current request to complete token-by-token generation, leaving the GPU idle between requests. Third, memory over-allocation: the system reserves maximum sequence length memory for every request regardless of actual usage, preventing concurrent processing of multiple requests.

```python
# Each iteration recomputes attention for ALL previous tokens
for _ in range(max_tokens):
    logits = model.forward(all_tokens)  # O(N²) attention!
    next_token = sample(logits[-1])
    all_tokens.append(next_token)
```

Production systems solve these problems with **KV caching** (O(N²) → O(N)), **continuous batching** (parallel multi-request execution), and **paged memory management** (dramatically increased capacity). The combination enables serving many concurrent users interactively. The practical impact becomes clear when comparing user experience: naive implementations process one request at a time with sequential token generation, creating deep queues where users wait unacceptably long. Production systems like vLLM and SGLang batch many requests together, generating tokens in parallel across the batch, keeping queue depth minimal and response times short. This chapter explains these techniques through concept-then-implementation pairs.

## 16.1 Request Lifecycle: Concept and Implementation

### 16.1.1 Request Abstraction

Production serving organizes computation around **requests**—user tasks with specific prompts and generation parameters. Each request tracks input specifications and dynamic state throughout execution.

Each request maintains several state properties: `prompt_tokens` holds the original input token sequence, `cached_len` tracks how many tokens have their KV cache computed, `output_tokens` accumulates generated tokens, and `extend_len` indicates how many tokens need processing this iteration—computed as `len(prompt) - cached_len` during prefill or simply 1 during decode.

**Example Request Lifecycle**:

```python
# Initial state: User submits "What is MLIR?" → tokens [1, 2, 3, 4, 5]
req = Request(
    prompt=[1, 2, 3, 4, 5],
    max_tokens=10,
    cached_len=0,     # No KV cache yet
    device_len=5      # 5 prompt tokens
)

# After prefill (first forward pass)
req.cached_len = 5    # All prompt tokens cached
req.device_len = 6    # Added first generated token
req.output_tokens = [42]  # Generated token 42

# After decode iteration 2
req.cached_len = 6    # Previous token now cached
req.device_len = 7    # Added second generated token
req.output_tokens = [42, 73]

# ... continues until device_len == max_device_len or EOS token
```

### 16.1.2 Implementation: Request Class

The `Request` class encapsulates a single user request and tracks its state throughout the lifecycle:

```python
class Request:
    def __init__(self, req_id, prompt_tokens, max_tokens=100):
        self.req_id = req_id
        self.prompt_tokens = prompt_tokens  # Input sequence
        self.max_tokens = max_tokens        # Generation limit
        self.cached_len = 0                 # Tokens with KV cache computed
        self.output_tokens = []             # Generated tokens
        self.kv_pages = []                  # Memory pages allocated

    @property
    def extend_len(self):
        """How many tokens to process this iteration"""
        if self.cached_len < len(self.prompt_tokens):
            return len(self.prompt_tokens) - self.cached_len  # Prefill
        else:
            return 1  # Decode: one new token

    @property
    def is_finished(self):
        return len(self.output_tokens) >= self.max_tokens
```

The `extend_len` property dynamically determines workload—many tokens during prefill, exactly one during decode.

### 16.1.3 Batch Abstraction

Multiple requests execute together in batches. The key difference: prefill processes **many tokens per request**, decode processes **one token per request**.

```python
class Batch:
    def __init__(self, requests, input_ids, positions):
        self.requests = requests
        self.input_ids = input_ids    # Flattened token sequence
        self.positions = positions    # Position indices for each token

    @staticmethod
    def from_prefill(requests):
        """Concatenate all uncached prompt tokens"""
        all_tokens, all_positions = [], []
        for req in requests:
            tokens = req.prompt_tokens[req.cached_len:]
            positions = range(req.cached_len, len(req.prompt_tokens))
            all_tokens.extend(tokens)
            all_positions.extend(positions)
        return Batch(requests, all_tokens, all_positions)

    @staticmethod
    def from_decode(requests):
        """One token per request (last generated)"""
        tokens = [req.output_tokens[-1] for req in requests]
        positions = [len(req.prompt_tokens) + len(req.output_tokens) - 1 
                     for req in requests]
        return Batch(requests, tokens, positions)
```

Batch construction varies by phase—concatenated sequences for prefill, single tokens for decode.

## 16.2 Paged KV Cache: Concept and Implementation

### 16.2.1 The Memory Fragmentation Problem

The naive approach allocates max_seq_len memory for each request—wasteful when actual prompt lengths vary widely (50-2000 tokens). Production systems use **paged memory management**: divide KV cache into fixed-size pages, allocate on-demand.

```
Contiguous allocation (naive):
Request 1 (50 tokens):   ██░░░░░░░░░░░░░░░░ (massive waste)
Request 2 (100 tokens):  ████░░░░░░░░░░░░░░ (substantial waste)
Request 3 (1500 tokens): ███████████████░░░ (modest waste)
```

**Paged solution**: Divide KV cache into fixed-size pages (e.g., 16 tokens per page), allocate on-demand.

```
Physical memory (pages):
[0] [1] [2] [3] [4] [5] [6] [7] [8] [9] ... [N]

Request 1 (50 tokens):  Uses pages [0, 1, 2]      (3 pages, minimal waste)
Request 2 (100 tokens): Uses pages [5, 6, 7, 8, 9] (6 pages, minimal waste)
Request 3 (1500 tokens): Uses pages [10-103]       (94 pages, minimal waste)
```

### 16.2.2 Implementation: KV Cache Pool

The pool manages a free list of pages and allocates on-demand:

```cpp
class KVCachePool {
    std::set<int> free_pages_;
    int page_size_, num_pages_;

public:
    std::vector<int> allocate(int num_tokens) {
        int pages_needed = (num_tokens + page_size_ - 1) / page_size_;  // Ceiling division

        if (free_pages_.size() < pages_needed)
            throw std::runtime_error("Out of memory");

        std::vector<int> allocated;
        auto it = free_pages_.begin();
        for (int i = 0; i < pages_needed; i++) {
            allocated.push_back(*it);
            free_pages_.erase(it++);
        }
        return allocated;
    }

    void deallocate(const std::vector<int>& pages) {
        for (int page : pages)
            free_pages_.insert(page);
    }
};
```

The allocate method computes pages needed, pops them from the free set, and returns page IDs. The deallocate method returns pages to the free set.

**Page Table Translation**. The paging system maintains a mapping from logical token positions to physical memory locations. When accessing the KV cache for a specific token, the system first calculates which logical page contains that token through integer division by page size. The page table then maps this logical page number to a physical page in GPU memory. Finally, the offset within that page locates the exact cache entry. This indirection enables non-contiguous allocation while maintaining $O(1)$ access time—the same performance as contiguous memory but with dramatically better utilization.

For example, accessing token 42 in a request with 16-token pages involves three simple calculations: `logical_page = 42 // 16 = 2` identifies the third page, `offset = 42 % 16 = 10` finds the position within that page, and `physical_page = page_table[request_id][2]` retrieves the actual memory location. The final address combines the physical page start with the offset.

Paging allocates based on actual usage rather than worst-case, enabling 5-10× more concurrent requests.

## 16.3 Continuous Batching: Concept and Implementation

### 16.3.1 The Static Batching Problem

Traditional batching waits for **all requests** to complete before accepting new work. This creates severe underutilization when requests finish at different times.

```
Static batching:
Req 1: ████████████████████████████████████████ (1000 tokens)
Req 2: ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (100 tokens, then idles!)
Req 3: ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ (120 tokens, then idles!)
Req 4: ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (100 tokens, then idles!)
GPU Utilization: 4/4 → 1/4 (25%) for 880 iterations
```

LLM decode generates exactly one token per request per iteration. When a request finishes, its slot becomes immediately available.

### 16.3.2 Implementation: Continuous Batching Loop

The serving loop implements dynamic batch management:

```python
def continuous_batching_loop(request_pool, model, max_batch_size=32):
    while True:
        # Step 1: Admit new requests up to capacity
        while len(request_pool.running) < max_batch_size and request_pool.waiting:
            req = request_pool.waiting.pop(0)
            request_pool.running.append(req)

        if not request_pool.running:
            continue

        # Step 2: Form batch from current running requests
        batch = Batch.from_decode(request_pool.running)

        # Step 3: Execute forward pass
        logits = model.forward(batch.input_ids, batch.positions)

        # Step 4: Update each request with generated token
        for req, token in zip(batch.requests, sample(logits)):
            req.output_tokens.append(token)
            req.cached_len += 1

        # Step 5: Remove finished requests (frees slots immediately!)
        request_pool.running = [r for r in request_pool.running if not r.is_finished]
```

Step 5 immediately frees slots for new requests, maintaining high GPU utilization throughout variable-length generations. This produces substantial utilization improvement:

```
Continuous batching:
Req 1: ████████████████████████████████████████ (1000 tokens)
Req 2: ██████████ (finishes at 100)
Req 3: ████████████ (finishes at 120)
Req 4: ██████████ (finishes at 100)
Req 5:           ████████████████ (joins after Req 2 finishes)
Req 6:                     ████████████████████ (joins after Req 4)
GPU Utilization: Stays near 4/4 consistently!
```

## 16.4 Radix Cache: Concept and Implementation

### 16.4.1 The Prefix Reuse Problem

Many requests share common prefixes: system prompts, few-shot examples, document context. Computing KV cache independently for each request wastes computation.

Consider chatbot requests with shared system prompt (10 tokens) plus unique queries (2-4 tokens). Naive approach computes system prompt KV cache independently for each request—identical computation wasted!

**Solution**: Organize KV cache as a **radix tree** where shared prefixes are computed once and reused.

```
Radix tree:
                [root]
                  |
               [1..10] ← Computed once, shared by all requests
              /   |   \
          [20,21] [30,31] [40..43] ← Only unique portions computed
```

### 16.4.2 Radix Tree Properties

A radix tree is a **compressed trie** where each node stores one token and its corresponding KV cache pages, each path from root to node represents a token sequence, shared prefixes are stored once as internal nodes, and unique suffixes branch off as leaf nodes.

**Key Properties**. The tree has several important properties enabling efficient prefix sharing. All requests starting with the same prefix share the same nodes—for example, requests beginning with [1, 2, 3] share those three nodes. Each complete path from root to leaf is unique, ensuring unambiguous request identification. Finding a cached prefix takes linear time in sequence length, making lookup $O(m)$ where m is the query sequence length.

**Comparison to Alternative Caching Strategies**:

| Strategy | Structure | Lookup | Memory | Best For |
|----------|-----------|--------|--------|----------|
| No cache | None | N/A | Minimal | Unique queries |
| Hash table | token[] → cache | O(N) | High | Exact matches only |
| Radix tree | Tree nodes | O(N) | Moderate | Prefix sharing |

### 16.4.3 Implementation Design: Arena Allocation

The implementation uses arena-based memory management, storing all nodes in a contiguous `std::vector` and referencing them by integer IDs rather than pointers. This provides several benefits: no manual memory management, cache-friendly access, and clear ownership semantics. Nodes never move in memory after allocation, so integer IDs remain valid.

### 16.4.4 Node Structure

Each node stores token value, KV pages, children map, and access timestamp:

```cpp
using NodeID = int;
constexpr NodeID INVALID_NODE = -1;

class RadixNode {
    int token_;                        // Token at this position
    std::vector<int> kv_pages_;        // Physical pages for this token's KV cache
    std::map<int, NodeID> children_;   // token → child node ID
    double last_access_time_;          // For LRU eviction

public:
    NodeID get_child(int token) const {
        auto it = children_.find(token);
        return (it != children_.end()) ? it->second : INVALID_NODE;
    }

    void add_child(int token, NodeID child_id) {
        children_[token] = child_id;
    }

    bool is_leaf() const { return children_.empty(); }
};
```

Using `std::map` for children handles sparse vocabularies (50k tokens) efficiently—only storing edges that actually exist.

### 16.4.5 Radix Cache Operations

Three core operations manage the cache:

```cpp
class RadixCache {
    std::vector<RadixNode> nodes_;  // Arena storage
    NodeID root_id_;

public:
    // Find longest matching prefix
    std::pair<int, NodeID> match_prefix(const std::vector<int>& tokens) {
        NodeID current = root_id_;
        int matched = 0;

        for (int token : tokens) {
            NodeID child = nodes_[current].get_child(token);
            if (child == INVALID_NODE) break;
            current = child;
            matched++;
            nodes_[current].update_access_time();
        }
        return {matched, current};
    }

    // Insert new sequence
    void insert(const std::vector<int>& tokens, const std::vector<int>& pages) {
        auto [matched, parent] = match_prefix(tokens);

        for (int i = matched; i < tokens.size(); i++) {
            NodeID new_node = allocate_node();
            nodes_[new_node].set_token(tokens[i]);
            nodes_[new_node].set_kv_page(pages[i]);
            nodes_[parent].add_child(tokens[i], new_node);
            parent = new_node;
        }
    }

    // Evict LRU leaf
    std::vector<int> evict_lru_leaf() {
        NodeID leaf = find_lru_leaf();  // Scan for oldest leaf
        if (leaf == INVALID_NODE) return {};

        auto pages = nodes_[leaf].kv_pages();
        remove_from_parent(leaf);
        return pages;
    }
};
```

The `match_prefix()` method walks the tree linearly, `insert()` extends the tree only for new suffixes, and `evict_lru_leaf()` removes only leaves to preserve shared prefixes.

### 16.4.6 Usage Pattern

On request admission:
1. **Query cache**: `match_prefix()` returns how many tokens already cached
2. **Allocate pages**: Only for uncached portion of prompt
3. **After prefill**: Insert completed sequence into cache for future reuse

Subsequent requests with matching prefixes skip redundant computation.

### 16.4.7 LRU Eviction Strategy

When memory is full, evict leaf nodes (unique suffixes) while preserving internal nodes (shared prefixes). Internal nodes serve multiple requests with high utility, while leaf nodes serve only single requests with low utility after completion. Evicting internal nodes would break the cache for all descendants. The implementation scans active nodes to find the leaf with the oldest timestamp, removes it from the parent's children map, and returns pages to the pool.

**Eviction example**:

```
Before eviction:
                [root]
                  |
               [1..10] ← Internal (shared by 3 requests)
              /   |   \
    [20,21,22] [30,31] [40..43] ← Leaves (unique suffixes)
    (LRU)      (recent) (recent)

After evicting [20,21,22]:
                [root]
                  |
               [1..10] ← Preserved (still useful)
              /   \
           [30,31] [40..43] ← Kept (recent)

Pages freed: 3 pages returned to pool
Cache preserved: Common prefix [1..10] still cached
```

## 16.5 Chunked Prefill: Concept and Implementation

### 16.5.1 The Long Prompt Starvation Problem

Interactive LLM serving faces a fundamental scheduling challenge: long prompts monopolize GPU resources, causing short prompts to wait unacceptably long. This creates poor user experience where small queries (chatbot messages, quick questions) experience high latency due to large context processing (document analysis, multi-shot examples) ahead in the queue.

Naive FCFS (First-Come-First-Served) scheduling processes requests sequentially. A single 2000-token prompt (200ms prefill on modern GPUs) forces all subsequent requests to wait, regardless of their size. This violates the principle of fairness: small requests should not be penalized by large requests ahead in the queue.

```
Naive FCFS:
┌────────────────────────────────────────┐
│  Req 0: 2000 tokens (200ms)            │ ← GPU 100% busy
└────────────────────────────────────────┘
                        Req 1: 50 tokens waits 200ms! (40× penalty)
```

**Key Insight**: Prefill computation is **divisible**—we can split it into smaller chunks and interleave execution. Unlike indivisible operations (single matrix multiplication), attention computation over 2000 tokens can be broken into 4× 500-token chunks without sacrificing correctness. The KV cache for tokens 0-499 remains valid when computing tokens 500-999.

### 16.5.2 Chunking Strategy

The solution is to divide each prompt into fixed-size chunks (e.g., 256 tokens) and process them round-robin across all waiting requests. This provides bounded wait time—no request waits longer than one chunk duration. All requests advance at similar rates with proportional progress, though overhead slightly increases due to multiple kernel launches.

Tradeoffs:

| Chunk Size | Fairness | Throughput | Best For |
|------------|----------|------------|----------|
| 128 tokens | Excellent | Lower | Interactive chat |
| 512 tokens | Good | Higher | Mixed workload |
| 2048 tokens | Poor | Highest | Batch processing |

### 16.5.3 Implementation: Chunked Scheduling

The scheduler implements round-robin chunk distribution:

```python
class ChunkedPrefillManager:
    def __init__(self, token_budget=512, chunk_size=256):
        self.token_budget = token_budget
        self.chunk_size = chunk_size
        self.queue = []

    def schedule(self):
        selected = []
        total_tokens = 0
        requeue = []

        while self.queue and total_tokens < self.token_budget:
            req = self.queue.pop(0)

            # Extract next chunk
            start = req.cached_len
            end = min(start + self.chunk_size, len(req.prompt_tokens))
            chunk = req.prompt_tokens[start:end]

            if total_tokens + len(chunk) <= self.token_budget:
                selected.append((req, chunk, start))
                total_tokens += len(chunk)
                req.cached_len = end

                # Requeue if more chunks remain
                if req.cached_len < len(req.prompt_tokens):
                    requeue.append(req)
            else:
                self.queue.insert(0, req)  # Put back, budget exhausted
                break

        self.queue.extend(requeue)
        return Batch.from_chunks(selected) if selected else None
```

Round-robin ensures fair progress—each request gets one chunk per round, preventing long prompts from starving short ones.

### 16.5.4 Execution Timeline

**Without chunking (FCFS)**:

```
Time: 0ms      50ms     100ms    150ms    200ms
Req 0: [========================================] (2000 tokens)
Req 1:                                          [==] (50 tokens, waited 200ms)
Req 2:                                              [===] (70 tokens, waited 205ms)
```

**With chunking (256-token chunks)**:

```
Time: 0ms   13ms   26ms   39ms   52ms   65ms
Req 0: [Ch0][Ch1][Ch2][Ch3][Ch4][Ch5]... (2000 tokens, 8 chunks total)
Req 1:      [complete]                   (50 tokens, waited 13ms - 15× better!)
Req 2:           [complete]              (70 tokens, waited 26ms - 8× better!)
Decode:          [run][run][run][run]... (Interleaved with prefill)
```

Short requests complete quickly instead of waiting for long requests to finish entirely.

### 16.5.5 Integration with Decode Phase

Chunked prefill enables **mixed-phase execution**: prefill chunks and decode iterations run in the same time slice. Each iteration processes prefill chunks (token budget limited), then decode iterations (batch size limited), ensuring both make progress.

## 16.6 Prefill-Decode Separation: Concept and Implementation

### 16.6.1 Phase Characteristics

Prefill and decode phases have fundamentally different performance characteristics requiring specialized optimization strategies.

**Prefill Phase**. Prefill processes many tokens per request (100-1000) with quadratic attention complexity, making it compute-bound where the bottleneck is matrix multiplication throughput. The optimization goal is maximizing floating-point operations while minimizing memory writes, typically using smaller batches of requests (2-8) with longer sequences. Memory access patterns follow sequential KV cache writes as the cache populates from left to right.

**Decode Phase**. Decode generates one token per request with linear attention complexity over cached keys and values, making it memory bandwidth-bound where the bottleneck is loading KV cache entries. The optimization goal is maximizing memory bandwidth utilization through large batch parallelism (32-256 requests), processing many requests simultaneously where each contributes just a single token. Memory access patterns involve random KV cache reads across many pages as attention attends to all previous tokens.

Different phases need different batch sizes—small for compute-bound prefill to avoid memory overflow, large for bandwidth-bound decode to saturate memory bandwidth.

### 16.6.2 Scheduling Fundamentals

The different phase characteristics necessitate different scheduling strategies. Prefill uses FCFS (First-Come-First-Served) with token budget constraints to maintain fairness while preventing long prompts from monopolizing resources. The token budget caps the total tokens processed in a single iteration, ensuring predictable memory usage and latency bounds. Decode uses greedy batch formation up to max batch size, maximizing throughput by saturating memory bandwidth with as many concurrent requests as possible. Since each request generates exactly one token per iteration, the batch size directly determines parallelism.

### 16.6.3 Implementation: Two-Phase Managers

Separate managers handle each phase's distinct scheduling requirements:

```python
class PrefillManager:
    def __init__(self, token_budget=512):
        self.token_budget = token_budget
        self.queue = []

    def schedule(self):
        """FCFS with token budget constraint"""
        selected = []
        total_tokens = 0

        for req in self.queue:
            tokens_needed = len(req.prompt_tokens) - req.cached_len
            if total_tokens + tokens_needed <= self.token_budget:
                selected.append(req)
                total_tokens += tokens_needed
            else:
                break  # Budget exhausted

        self.queue = [r for r in self.queue if r not in selected]
        return Batch.from_prefill(selected) if selected else None

class DecodeManager:
    def __init__(self, max_batch_size=128):
        self.max_batch_size = max_batch_size
        self.running = []

    def schedule(self):
        """Greedy packing up to max batch size"""
        if not self.running:
            return None
        batch = self.running[:self.max_batch_size]
        return Batch.from_decode(batch)

    def remove_finished(self):
        finished = [r for r in self.running if r.is_finished]
        self.running = [r for r in self.running if not r.is_finished]
        return finished
```

Prefill uses FCFS with token budget for fairness and small batches, while decode uses greedy packing for throughput with large batches.

### 16.6.4 Integration Pattern

Coordinate both managers in a unified step function:

```python
class TwoPhaseScheduler:
    def __init__(self):
        self.prefill_mgr = PrefillManager(token_budget=512)
        self.decode_mgr = DecodeManager(max_batch_size=128)

    def step(self, model):
        # Phase 1: Prefill with token budget
        if prefill_batch := self.prefill_mgr.schedule():
            logits = model.forward(prefill_batch.input_ids, prefill_batch.positions)

            for req in prefill_batch.requests:
                if req.cached_len == len(req.prompt_tokens):
                    # Prefill complete, generate first token
                    token = sample(logits[req.req_id])
                    req.output_tokens.append(token)
                    self.decode_mgr.add_request(req)  # Transition to decode

        # Phase 2: Decode with large batch
        if decode_batch := self.decode_mgr.schedule():
            logits = model.forward(decode_batch.input_ids, decode_batch.positions)

            for req, logit in zip(decode_batch.requests, logits):
                req.output_tokens.append(sample(logit))
                req.cached_len += 1

            return self.decode_mgr.remove_finished()

        return []
```

Requests transition seamlessly from prefill to decode—completed prefills immediately join decode batch in next iteration.

## 16.7 Complete System Integration

Previous sections introduced individual components—paged KV cache, continuous batching, radix cache, chunked prefill, and two-phase scheduling. This section shows how these pieces fit together into a coherent serving system. The integration is non-trivial: components must cooperate to manage shared resources (GPU memory, compute capacity), coordinate request state transitions (waiting → prefill → decode → finished), and balance competing objectives (throughput vs latency, fairness vs efficiency).

The architecture follows a layered design where each component has well-defined responsibilities and clean interfaces. Higher layers handle scheduling and orchestration using Python for expressiveness and rapid development. Lower layers implement performance-critical operations in C++/MLIR for computational efficiency. This separation of concerns makes the system maintainable—scheduling logic can evolve independently of execution kernels, and execution optimizations don't require modifying scheduling algorithms.

### 16.7.1 NanoServing Architecture

The serving engine organizes computation as a pipeline where requests flow through multiple stages, each handling a specific aspect of the serving process. Understanding this dataflow is essential to comprehending how production LLM serving works.

```
┌─────────────────────────────────────────────────┐
│              NanoServingEngine                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  Request Queue (FIFO)                           │
│         ↓                                       │
│  Radix Cache Lookup (prefix match)              │
│         ↓                                       │
│  KV Pool Allocation (paged memory)              │
│         ↓                                       │
│  Chunked Prefill Manager (round-robin)          │
│         ↓                                       │
│  Decode Manager (continuous batching)           │
│         ↓                                       │
│  Model Executor (MLIR-compiled forward pass)    │
│         ↓                                       │
│  Radix Cache Insertion (for future reuse)       │
│         ↓                                       │
│  Response Queue (completed requests)            │
│                                                 │
└─────────────────────────────────────────────────┘
```

The pipeline architecture reveals how serving systems compose simple components into sophisticated behavior. Each stage performs a focused task: the request queue provides a buffer absorbing bursts of incoming requests; radix cache lookup identifies reusable computation from previous requests; KV pool allocation reserves GPU memory for the new request's state; chunked prefill manager ensures fair scheduling across mixed request sizes; decode manager batches running requests for parallel token generation; the model executor runs the actual transformer forward passes; radix cache insertion preserves completed work for future reuse; and the response queue accumulates finished requests for client retrieval.

The vertical flow represents request lifecycle progression. New requests enter at the top and flow downward through processing stages, accumulating computed state (KV cache entries, generated tokens) as they proceed. Requests spending time in prefill and decode stages consume the majority of execution time—these are the hot paths that determine overall system throughput. The radix cache provides a cross-cutting concern, allowing requests to skip computation by reusing cached state from structurally similar predecessors.

Resource management threads through the entire pipeline. The KV pool tracks memory allocation, the radix cache manages sharing and eviction, and the schedulers enforce capacity limits. These components continuously communicate—the prefill manager queries the KV pool before admitting requests, the radix cache may trigger eviction to free memory, and the decode manager adjusts batch size based on available capacity. This coordination ensures the system never exceeds physical limits while maximizing utilization of available resources.

### 16.7.2 NanoServingEngine Implementation

The `NanoServingEngine` class implements this architecture, coordinating all components through three primary methods: initialization establishes component instances and configures their parameters, request admission handles incoming work by checking cache and allocating memory, and the main serving loop executes iterative processing to make incremental progress on all active requests.

```python
class NanoServingEngine:
    def __init__(self, model_path, gpu_memory_gb=24):
        self.model = ModelExecutor(model_path)
        self.kv_pool = KVCachePool(page_size=16, num_pages=compute_pages(gpu_memory_gb))
        self.radix_cache = RadixCache(self.kv_pool)
        self.prefill_mgr = ChunkedPrefillManager(token_budget=512, chunk_size=256)
        self.decode_mgr = DecodeManager(max_batch_size=128)

    def add_request(self, prompt_tokens, max_tokens=100):
        req = Request(self.next_req_id, prompt_tokens, max_tokens)
        self.next_req_id += 1

        # Query radix cache for prefix match
        matched_len, _ = self.radix_cache.match_prefix(prompt_tokens)
        req.cached_len = matched_len

        # Allocate KV pages (with LRU eviction if OOM)
        uncached = len(prompt_tokens) - matched_len
        req.kv_pages = self.allocate_with_eviction(uncached + max_tokens)

        self.prefill_mgr.add_request(req)
        return req.req_id

    def step(self):
        # Phase 1: Chunked prefill
        if prefill_batch := self.prefill_mgr.schedule():
            logits = self.model.forward(prefill_batch.input_ids, prefill_batch.positions)

            for req in prefill_batch.requests:
                if req.cached_len == len(req.prompt_tokens):
                    req.output_tokens.append(sample(logits[req.req_id]))
                    self.decode_mgr.add_request(req)
                    self.radix_cache.insert(req.prompt_tokens, req.kv_pages)

        # Phase 2: Decode
        if decode_batch := self.decode_mgr.schedule():
            logits = self.model.forward(decode_batch.input_ids, decode_batch.positions)

            for req, logit in zip(decode_batch.requests, logits):
                req.output_tokens.append(sample(logit))
                req.cached_len += 1

            for req in self.decode_mgr.remove_finished():
                self.kv_pool.deallocate(req.kv_pages)

    def serve(self):
        while True:
            self.step()
```

**Initialization and Configuration**. The constructor establishes the serving infrastructure by instantiating and configuring all subsystems. The model executor wraps the MLIR-compiled GPT from Chapters 1-15, providing a forward pass interface that accepts input token IDs and position indices, returning logits for next token prediction. The KV pool allocates paged memory on the GPU—page size determines granularity (16 tokens is typical), and total pages derive from available GPU memory. The radix cache wraps the KV pool, managing the prefix tree structure and LRU eviction policy. The chunked prefill manager enforces a token budget (typically 512 tokens) to bound per-iteration memory usage and prevent long prefills from monopolizing the GPU. The decode manager specifies maximum batch size (often 128-256 concurrent requests) to saturate memory bandwidth during token generation.

These configuration parameters fundamentally shape system behavior. Smaller token budgets increase scheduling fairness at the cost of slightly reduced throughput, while larger budgets maximize compute efficiency but may create head-of-line blocking. Smaller chunk sizes provide finer-grained interleaving between prefill and decode, improving responsiveness for interactive workloads. Larger batch sizes increase decode throughput by amortizing memory access costs across more requests, but may increase latency for individual requests. Production systems often expose these as tunable parameters, allowing operators to optimize for their specific workload characteristics.

**Request Admission**. The `add_request()` method handles new request arrival through a three-step process. First, radix cache lookup finds the longest matching prefix in the cache tree—this may be zero (cache miss), partial (some prefix cached), or complete (full prompt cached, rare but possible in exact duplicates). The matched length determines how many tokens can skip computation. Second, memory allocation reserves KV pages for uncached tokens plus expected generation length. The allocation may trigger LRU eviction if memory pressure is high—the system preferentially evicts cold prefixes (rarely accessed leaf nodes) while preserving hot prefixes (frequently accessed internal nodes). Third, queue insertion adds the request to the prefill manager, making it eligible for scheduling in subsequent iterations.

The admission logic demonstrates how radix caching provides transparent optimization. From the client's perspective, requests simply get queued for processing. Internally, the system examines structural similarities with previous requests, reusing computation when possible. Requests sharing a 1000-token system prompt might skip 98% of prefill work, but the client sees identical behavior whether the cache hits or misses. This transparency is valuable—serving logic remains simple while opportunistic optimization happens automatically.

**Main Serving Loop**. The `step()` method performs one iteration of the continuous batching cycle, coordinating both prefill and decode phases. Phase 1 processes prefill chunks: the chunked prefill manager selects requests and chunk sizes respecting the token budget, the model executor runs the forward pass computing KV cache for those tokens, and completion detection identifies requests finishing their prefill. Completed requests undergo three actions: sample the first output token from prefill logits, insert the completed prompt into the radix cache for future reuse, and transition to the decode manager for subsequent token generation.

Phase 2 handles decode generation: the decode manager forms a batch from all running requests (up to max batch size), the model executor computes logits using cached KV entries and new tokens, sampling produces next tokens which extend each request's output, and completion detection identifies finished requests. Cleanup deallocates KV pages for finished requests, returning them to the pool for reuse. The critical insight is that both phases execute within a single iteration—the serving loop interleaves prefill chunks with decode batches, ensuring all requests make progress regardless of their current phase.

The `serve()` method implements the outermost loop, continuously calling `step()` to process all queued and running requests. This simple infinite loop represents the core of production serving: iterate forever, processing whatever work exists, maintaining high GPU utilization through dynamic batching and fair scheduling. Real systems add error handling, graceful shutdown, metrics collection, and health checks around this core loop, but the fundamental pattern remains—continuous iteration making incremental progress on the active request set.

### 16.7.3 Request Dataflow

Understanding a concrete example clarifies how abstract components interact during actual request processing. Consider a request with a partially cached prompt.

```
1. [Client] Submit: prompt_tokens=[1..10, 20, 21, 22], max_tokens=50

2. [Admission]
   radix_cache.match_prefix([1..10, 20, 21, 22]) → matched=10 (cache hit!)
   kv_pool.allocate(3 uncached + 50 generation = 53 tokens) → 4 pages
   prefill_mgr.add_request(req)

3. [Chunked Prefill - Round 1]
   schedule() → extract chunk [20, 21, 22]
   model.forward() → compute KV cache for these 3 tokens
   req.cached_len = 13 (prefill complete!)
   sample() → first output token = 42
   radix_cache.insert([1..10, 20, 21, 22]) → future requests reuse
   decode_mgr.add_request(req)

4. [Decode - Iterations 1-50]
   schedule() → batch with req and others
   model.forward([42]) → logits
   sample() → token 73, append to req.output_tokens
   ... (repeat 49 more times)

5. [Completion]
   req.is_finished == True
   kv_pool.deallocate(4 pages)
   Return: output_tokens = [42, 73, 108, ..., 99]
   Note: Cached prefix [1..10, 20, 21, 22] remains in tree
```

This trace reveals several key system behaviors. The radix cache provides automatic optimization—step 2 discovers that 10 of 13 prompt tokens are already cached, reducing prefill work by 77%. This optimization is transparent to the client and opportunistic based on previous request patterns. Memory allocation uses page-level granularity—53 tokens require 4 pages (assuming 16-token pages), with the last page partially filled. This demonstrates how paging trades perfect memory efficiency for allocation flexibility.

The prefill-to-decode transition happens automatically based on state. When `cached_len` reaches `len(prompt_tokens)`, the request has completed prefill and immediately joins the decode batch. No explicit state machine or transition logic—the condition `cached_len == len(prompt_tokens)` serves as the transition predicate. This elegant design avoids complex state tracking; request progress naturally determines phase membership.

Decode iterations show the batching pattern. Step 4 processes 50 iterations generating one token each, but the request shares the batch with other concurrent requests. If 32 requests run simultaneously, each iteration generates 32 tokens total—high GPU utilization through parallel processing. The trace focuses on one request for clarity, but production serving typically processes dozens to hundreds of requests per iteration, amortizing overhead across large batches.

Completion and cleanup demonstrate resource lifecycle management. The request finishes when it reaches max tokens or generates an end-of-sequence marker, triggering deallocation of its KV pages (4 pages returned to the pool). However, the radix cache entry persists—the path [1..10, 20, 21, 22] remains in the tree for future requests to reuse. This persistence is valuable: if 100 requests share this prefix, the first request pays the computation cost while subsequent 99 requests benefit from the cached state.

### 16.7.4 Performance Summary

Each algorithmic component addresses a specific performance bottleneck, and their effects compose multiplicatively rather than additively. Understanding the individual contributions clarifies why production serving achieves such dramatic speedups over naive implementations.

**KV Cache**: Eliminates O(N²) redundant attention computation → O(N) incremental updates (5-6× reduction per request).

**Continuous Batching**: Dynamic admission/eviction maintains high GPU utilization when requests complete at different times (2× improvement over static batching).

**Radix Cache**: Shared prefixes computed once and reused. High-sharing workloads (chatbots with system prompts) see 2-3× prefill reduction. Low-sharing workloads (creative writing) see minimal benefit.

**Chunked Prefill**: Trades ~10% throughput for 8× better tail latency by preventing long prompts from starving short ones.

**Paged Memory**: Avoids worst-case allocation, allocating based on actual usage rather than theoretical maximum. Typical capacity increase: 5-10× more concurrent users on same hardware.

**Combined Impact**. These optimizations compose multiplicatively. Consider a workload with 32 concurrent requests, 100-token prompts with 60% prefix sharing, and 20-token generation lengths. Naive sequential processing would perform 32 × 120 = 3,840 total forward passes (one per token per request). KV caching reduces this to 32 × (100 + 20) since we avoid recomputing attention for cached tokens. Parallel batching with continuous scheduling reduces the iteration count further by processing multiple requests simultaneously—now we need approximately 120 iterations (100 for longest prefill + 20 for decode). Radix caching with 60% sharing reduces prefill cost by nearly 60%, saving around 1,920 forward passes. The cumulative effect yields roughly 30× speedup from combining these techniques.

The multiplicative composition is crucial. Applying only KV caching gives moderate improvement. Adding only continuous batching provides another modest gain. But combining all techniques—paged memory enabling high concurrency, continuous batching maintaining GPU saturation, radix caching eliminating redundant computation, and chunked prefill ensuring fairness—creates the dramatic performance profile users experience in production LLM serving systems. Each optimization unlocks additional capacity that subsequent optimizations can exploit.

**Workload Sensitivity**. The relative impact of each technique depends heavily on workload characteristics. Radix caching shines on workloads with substantial prefix sharing (chatbots, few-shot prompting, document Q&A) but provides little benefit for entirely unique requests (creative writing, diverse queries). Chunked prefill dramatically improves tail latency when request sizes vary widely but adds overhead when all requests are similar length. Continuous batching provides maximum benefit when generation lengths differ substantially, allowing short requests to complete and free capacity for new work. Paged memory helps most when actual usage varies widely from theoretical maximum, reducing waste from over-allocation.

Production serving systems must tune these components for their specific workload patterns. Interactive chat applications prioritize low latency, so they use smaller batch sizes, aggressive chunking, and careful radix cache tuning. Batch document processing prioritizes throughput, favoring larger batches, minimal chunking overhead, and less frequent eviction. Most deployments fall somewhere between these extremes, requiring measurement and profiling to find optimal parameters.

## 16.8 Conclusion

This chapter demonstrated production LLM serving through six integrated concepts, each addressing a specific bottleneck in naive implementations:

1. **Request lifecycle**: The state tracking abstraction (Request and Batch classes) provides clean interfaces for managing user tasks through their lifecycle, encapsulating the complexity of phase transitions and progress tracking.

2. **Paged KV cache**: Virtual memory techniques applied to attention cache eliminate fragmentation and enable flexible allocation. The page table indirection costs essentially nothing while dramatically improving memory utilization.

3. **Continuous batching**: Dynamic admission and eviction maintains consistently high GPU utilization regardless of varying generation lengths. The key insight—one token per request per iteration—enables seamless batch composition changes.

4. **Radix cache**: Tree-based prefix sharing automatically detects and reuses computation from structurally similar requests. The arena-based implementation provides efficient memory management without complex reference counting.

5. **Chunked prefill**: Breaking long prompts into schedulable chunks prevents head-of-line blocking. The round-robin strategy ensures fairness with minimal overhead.

6. **System integration**: The NanoServingEngine coordinates all components through a clean pipeline architecture, demonstrating how individual optimizations compose into a complete serving system.

The progression from individual techniques to complete integration reveals an important lesson: production systems achieve their performance through composition of orthogonal optimizations. Each technique addresses a different bottleneck, enabling the next optimization to have greater effect. Paged memory enables higher concurrency, continuous batching keeps the GPU saturated with that concurrency, radix caching reduces the actual computation needed, and chunked prefill ensures fairness across heterogeneous workloads.

This chapter completes the journey from compiler foundations to production deployment. Chapters 1-15 built the MLIR infrastructure for efficient model execution—dynamic shapes, bufferization, custom dialects, and GPU concepts. Chapter 16 showed how compiled models integrate into serving systems that orchestrate multiple concurrent requests. The compiler optimizes individual operations; the serving system optimizes resource allocation across requests. Together, they enable the interactive AI applications users experience today.

Production systems like vLLM, SGLang, and TensorRT-LLM extend these foundations with hardware-specific optimizations: custom CUDA kernels for paged attention, multi-GPU tensor parallelism for large models, and sophisticated scheduling policies for complex workloads. Understanding the algorithmic core presented here—the fundamental data structures, scheduling policies, and resource management strategies—provides the foundation for exploring those production implementations with confidence. The techniques are universal; the implementations vary by hardware and workload requirements.