# Chapter 16: Production LLM Serving

Chapters 1-15 built a complete nano GPT model with MLIR compiler acceleration. The implementation demonstrates transformer architecture, KV caching, and parallel execution patterns. However, building a production LLM serving system requires more than a correct model—it demands efficient request scheduling, memory management, and multi-request batching. This chapter teaches production serving techniques used in real-world systems like vLLM, SGLang, and TensorRT-LLM.

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

## 16.1 Request Lifecycle

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

### Request Class

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

### Batch Abstraction

Multiple requests execute together in batches to achieve parallelism. The scheduler and manager components (introduced in sections 16.3 and 16.6) use the Batch abstraction to group requests for efficient GPU execution. The key challenge is that prefill and decode phases have fundamentally different batching requirements: prefill processes many tokens per request (the entire uncached prompt), while decode processes exactly one token per request (the newly generated token).

The Batch class provides static factory methods that handle these different cases. During prefill, `from_prefill()` concatenates all uncached prompt tokens from multiple requests into a single flattened sequence, enabling parallel processing across both the batch dimension and sequence dimension. During decode, `from_decode()` collects just the last generated token from each request, since decode iterations process one new token at a time. Both methods also track position indices for each token, which the model needs for positional embeddings and attention masking.

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

## 16.2 Paged KV Cache

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

### KV Cache Pool

**KV Cache Isn't Enough**. In Chapter 14, we implemented KV caching to avoid recomputing attention for previous tokens—a single request maintains its cache as a contiguous tensor growing with each decode iteration. This works well for processing one request at a time, but production serving processes dozens to hundreds of concurrent requests. If we allocate contiguous memory for each request's maximum possible length, we waste enormous amounts of GPU memory on unused capacity.

The problem is allocation granularity. Chapter 14's approach allocates `max_seq_len × hidden_dim` memory per request upfront. When requests have varying actual lengths (some use 50 tokens, others use 1500), most allocations are oversized. Even worse, growing contiguous allocations can't reuse freed memory from completed requests—fragmentation prevents efficient packing.

Production systems solve this with paged memory management, borrowing virtual memory concepts from operating systems. The KV cache is divided into fixed-size pages, and each request's cache is a collection of non-contiguous pages. This provides two critical benefits: allocate only what's actually needed (demand paging), and reuse freed pages from any completed request (no fragmentation). The pool manages a free list of pages, allocating on demand and reclaiming when requests complete.

**Implementation**. The pool manages a free list of pages and allocates on-demand:

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

## 16.3 Continuous Batching

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

### Continuous Batching Solution

To fix the static batching problem, we introduce **continuous batching**—a dynamic scheduling approach where the batch composition changes every iteration. The key insight is that decode iterations are independent and uniform: each request generates exactly one token per iteration regardless of its current sequence length or how many tokens remain until completion. This uniformity means we can safely add or remove requests from the batch between iterations without breaking the computation.

Here's how continuous batching works: After each forward pass, the scheduler checks which requests have finished (reached max tokens or generated an end-of-sequence marker) and immediately removes them from the running batch. It then admits waiting requests to fill the freed slots, up to the maximum batch size. This happens every iteration, creating a dynamic flow where new requests begin processing as soon as capacity becomes available, rather than waiting for an entire batch cycle to complete.

The algorithm maintains three request collections: a waiting queue holding new arrivals, a running batch actively generating tokens, and a finished set containing completed requests. Each iteration performs five steps in sequence.

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

**How the Five Steps Work Together**. Step 1 (admission) fills any vacant batch slots by pulling from the waiting queue, ensuring the batch size stays near its maximum. Step 2 (batch formation) packages the current running requests using the Batch abstraction from section 16.1.3, creating a single forward pass input. Step 3 (execution) runs the model, computing logits for all requests in parallel. Step 4 (update) samples the next token for each request and extends their output sequences. Step 5 (cleanup) removes finished requests, which is the critical step—by freeing slots immediately rather than waiting for all requests to complete, the system maintains high utilization.

Step 5 immediately frees slots for new requests, maintaining high GPU utilization throughout variable-length generations. The cycle then repeats with Step 1 admitting new requests to fill the newly freed capacity. This produces substantial utilization improvement over static batching:

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

## 16.4 Radix Cache

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

### Radix Tree Properties

A radix tree is a **compressed trie** where each node stores one token and its corresponding KV cache pages, each path from root to node represents a token sequence, shared prefixes are stored once as internal nodes, and unique suffixes branch off as leaf nodes.

**Key Properties**. The tree has several important properties enabling efficient prefix sharing. All requests starting with the same prefix share the same nodes—for example, requests beginning with [1, 2, 3] share those three nodes. Each complete path from root to leaf is unique, ensuring unambiguous request identification. Finding a cached prefix takes linear time in sequence length, making lookup $O(m)$ where m is the query sequence length.

**Comparison to Alternative Caching Strategies**:

| Strategy | Structure | Lookup | Memory | Best For |
|----------|-----------|--------|--------|----------|
| No cache | None | N/A | Minimal | Unique queries |
| Hash table | token[] → cache | O(N) | High | Exact matches only |
| Radix tree | Tree nodes | O(N) | Moderate | Prefix sharing |

### Arena Allocation

The implementation uses arena-based memory management, storing all nodes in a contiguous `std::vector` and referencing them by integer IDs rather than pointers. This provides several benefits: no manual memory management, cache-friendly access, and clear ownership semantics. Nodes never move in memory after allocation, so integer IDs remain valid.

### Node Structure

Each node in the radix tree represents a single token position in a cached sequence and stores four essential pieces of information: the token value at this position, the KV cache pages holding this token's attention state, a children map connecting to subsequent tokens, and a timestamp for LRU eviction.

The node provides three key APIs: `get_child(token)` looks up whether a child node exists for the given token value, returning the child's ID or INVALID_NODE if not found; `add_child(token, child_id)` creates a new edge in the tree, extending a cached path with an additional token; and `is_leaf()` checks whether this node has any children, identifying unique suffixes eligible for eviction. These simple operations compose into the tree traversal and mutation algorithms implemented by the RadixCache class.

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

**Design Rationale**. Using `std::map<int, NodeID>` for the children map handles sparse vocabularies efficiently. With 50,000+ tokens in typical vocabularies, storing a dense array per node would waste enormous memory. The map only stores edges that actually exist in cached request paths. The integer NodeID rather than a pointer provides safety—nodes never move in the arena vector, so IDs remain valid without reference counting or smart pointer overhead. The `last_access_time_` timestamp supports LRU eviction by tracking when each cached path was last reused, allowing the system to identify cold branches for removal when memory pressure occurs.

### Radix Cache Operations

The RadixCache class provides three core operations that implement prefix matching, tree growth, and memory reclamation. Understanding these operations reveals how the cache automatically detects and exploits prefix sharing across requests.

**Operation 1: match_prefix()**. This operation walks the tree to find the longest prefix of the input token sequence that exists in the cache. Starting from the root, it follows edges corresponding to each input token in sequence. When it finds a token with no matching child, the search terminates—all tokens up to that point are cached, and the remainder need computation. The operation returns both the matched length (how many tokens can skip computation) and the node ID where the match stopped (where to attach new nodes if needed). Importantly, it updates `last_access_time_` for each visited node, maintaining LRU information for eviction.

**Operation 2: insert()**. After computing KV cache for a request, this operation adds the new path to the tree for future reuse. It first calls `match_prefix()` to find any existing shared prefix, then extends the tree only for the unique suffix. Each new token gets a new node allocated from the arena, storing the token value and corresponding KV page. The nodes are linked into the tree by adding child edges. This incremental insertion is efficient—shared prefixes already exist in the tree, and only new suffixes require allocation.

**Operation 3: evict_lru_leaf()**. When memory pressure occurs, this operation frees space by removing the least recently used leaf node. It scans the tree to find the leaf with the oldest `last_access_time_`, removes it from its parent's children map, and returns the freed KV pages to the pool. Crucially, it only evicts leaves—internal nodes represent shared prefixes and must be preserved, as removing them would invalidate cache for all descendants. This selective eviction maintains high hit rates by preserving frequently reused paths.

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

The three operations work together to provide transparent prefix reuse. When a request arrives, `match_prefix()` identifies reusable computation. After prefill completes, `insert()` preserves that work for future requests with similar prefixes. When memory fills, `evict_lru_leaf()` removes cold paths while preserving hot shared prefixes. The elegance is that all this happens automatically—the serving system doesn't need complex heuristics to detect sharing patterns; the tree structure naturally captures and exploits prefix overlap.

### Usage Pattern

On request admission:
1. **Query cache**: `match_prefix()` returns how many tokens already cached
2. **Allocate pages**: Only for uncached portion of prompt
3. **After prefill**: Insert completed sequence into cache for future reuse

Subsequent requests with matching prefixes skip redundant computation.

### LRU Eviction Strategy

When GPU memory fills and no free pages remain, the system must evict cached entries to make room for new requests. The eviction policy carefully balances two objectives: free sufficient memory for immediate needs, while preserving the most valuable cached state for future reuse. The strategy is to evict leaf nodes (unique suffixes) while preserving internal nodes (shared prefixes).

This policy is optimal for workloads with prefix sharing. Internal nodes represent computation shared by multiple request patterns—for example, a system prompt used by all chatbot requests, or few-shot examples prepended to many queries. These nodes have high utility because they avoid redundant computation for many future requests. Leaf nodes represent unique continuations specific to individual completed requests. Once that request finishes, its unique suffix has low utility—unlikely to be reused unless an identical request arrives. Evicting internal nodes would break the cache for all dependent paths, forcing recomputation of the shared prefix for every subsequent request.

The eviction algorithm scans the tree to find the leaf node with the oldest `last_access_time_` timestamp. This LRU (Least Recently Used) heuristic targets cold paths that haven't been reused recently, likely representing one-time queries rather than recurring patterns. After identifying the target leaf, the algorithm removes it from its parent's children map, effectively pruning that branch from the tree. The freed KV pages return to the pool for allocation to new requests.

**Example Scenario**. Consider a tree where the root connects to an internal node representing tokens [1..10] (perhaps a system prompt). This internal node has three children: leaf [20,21,22] accessed 5 minutes ago, leaf [30,31] accessed 1 minute ago, and leaf [40..43] accessed 30 seconds ago. When eviction is needed, the algorithm identifies [20,21,22] as the LRU leaf. After eviction, the internal node [1..10] remains intact, preserving the shared prefix for future requests. The two recent leaves [30,31] and [40..43] also survive. Only the cold leaf [20,21,22] is removed, freeing 3 pages (one per token with 1-token pages, or potentially fewer with multi-token pages). This selective removal maintains cache effectiveness—the hot prefix [1..10] continues serving new requests, while cold unique suffixes are purged.

## 16.5 Chunked Prefill

Interactive LLM serving faces a fundamental scheduling challenge: long prompts monopolize GPU resources, causing short prompts to wait unacceptably long. This creates poor user experience where small queries (chatbot messages, quick questions) experience high latency due to large context processing (document analysis, multi-shot examples) ahead in the queue.

Naive FCFS (First-Come-First-Served) scheduling processes requests sequentially. A single 2000-token prompt (200ms prefill on modern GPUs) forces all subsequent requests to wait, regardless of their size. This violates the principle of fairness: small requests should not be penalized by large requests ahead in the queue.

Without chunking, request 0's 2000 tokens occupy the GPU for an extended period while request 1 (50 tokens) waits idle. The wait time is disproportionate—a request requiring minimal computation experiences massive latency due to another request's size.

**Key Insight**: Prefill computation is **divisible**—we can split it into smaller chunks and interleave execution. Unlike indivisible operations (single matrix multiplication), attention computation over 2000 tokens can be broken into 4× 500-token chunks without sacrificing correctness. The KV cache for tokens 0-499 remains valid when computing tokens 500-999. This divisibility enables fair scheduling where multiple requests make progress simultaneously.

### Chunking Strategy

The solution is to divide each prompt into fixed-size chunks (e.g., 256 tokens) and process them round-robin across all waiting requests. This provides bounded wait time—no request waits longer than one chunk duration. All requests advance at similar rates with proportional progress, though overhead slightly increases due to multiple kernel launches.

The chunking mechanism works as follows. When a request arrives with a 2000-token prompt, the system divides it into chunks of 256 tokens each (configurable parameter). Instead of processing all 2000 tokens in one blocking operation, the scheduler extracts the first 256-token chunk and adds it to the current iteration's prefill batch. The request then moves to the back of the queue with its progress updated—next iteration, it will get another 256-token chunk processed. Meanwhile, other requests (including short ones) get their turns, preventing the long request from monopolizing the GPU.

Chunk size determines the fairness-throughput tradeoff. Smaller chunks (128 tokens) provide excellent fairness with bounded latency—even very short requests never wait long. However, smaller chunks increase overhead from batch formation and kernel launches, slightly reducing overall throughput. Larger chunks (2048 tokens) maximize throughput by reducing overhead, but approach the original FCFS blocking behavior. Production systems typically use 256-512 tokens as a middle ground, achieving good fairness without excessive overhead. Interactive workloads favor smaller chunks, while batch processing workloads favor larger chunks.

**Tradeoff Analysis**:

Small chunks around 128 tokens provide excellent fairness by minimizing the maximum wait time any request experiences. If a short 50-token request arrives while a long 2000-token request is processing, it waits at most one 128-token chunk duration before getting scheduled. However, this fine-grained scheduling has lower throughput because the system spends more time on batch formation overhead and kernel launch latency relative to actual computation. This configuration suits interactive chat applications where response time matters more than aggregate throughput.

Medium chunks around 512 tokens balance fairness and throughput. Short requests still get reasonable response times—waiting at most 512 tokens worth of computation—while longer chunks amortize scheduling overhead better. This middle ground works well for mixed workloads with both interactive queries and longer document processing requests.

Large chunks around 2048 tokens prioritize throughput over fairness. The scheduler processes larger contiguous sequences, maximizing GPU compute efficiency and minimizing overhead. However, short requests may experience significant head-of-line blocking if they queue behind large chunks. This configuration suits batch processing workloads where minimizing total completion time matters more than individual request latency.

### Chunked Prefill Manager

The ChunkedPrefillManager implements the round-robin scheduling algorithm with two key parameters: `token_budget` caps the total tokens processed per iteration (preventing memory overflow), and `chunk_size` determines the maximum tokens extracted from any single request per round (ensuring fairness). Understanding the scheduling logic reveals how the manager balances throughput, fairness, and memory constraints.

**Algorithm Overview**. The `schedule()` method iterates through the prefill queue, extracting the next chunk from each request until the token budget is exhausted. For each request, it calculates the chunk boundary: starting from `cached_len` (progress so far) and extending up to `chunk_size` tokens, but not past the prompt end. If adding this chunk would exceed the token budget, the request goes back to the queue front for next iteration. Otherwise, the chunk is added to the current batch, the request's `cached_len` advances by the chunk size, and the request is requeued if more chunks remain.

The requeuing mechanism provides fairness. After extracting a chunk, the request moves to the back of the queue. Next iteration, it must wait for all other requests to get their turns before receiving another chunk. This round-robin ordering prevents any request from monopolizing the scheduler—long requests don't block short ones, and short requests don't starve long ones.

**Token Budget Management**. The budget serves two purposes: it bounds memory usage (token budget directly determines maximum memory footprint), and it bounds iteration latency (processing at most 512 tokens worth of computation per iteration provides predictable timing). The greedy packing algorithm fills the budget as much as possible without exceeding it, maximizing utilization while respecting the constraint.

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

### Execution Timeline Comparison

Comparing execution timelines reveals chunking's impact on request interleaving and latency.

**Without chunking (FCFS)**:

The naive approach processes the 2000-token request completely before starting subsequent requests. The long request occupies the GPU for its entire prefill duration, forcing short requests to wait idle in the queue. Request 1 (50 tokens) and Request 2 (70 tokens) experience wait times proportional to the long request's length, resulting in poor response time for these short, interactive queries.

**With chunking**:

Chunked scheduling divides the 2000-token request into 8 chunks of 256 tokens each (assuming 256-token chunk size). After processing the first chunk, the scheduler pauses the long request and processes other waiting requests. Request 1 completes its 50-token prefill in the next slot, followed by Request 2's 70-token prefill. The long request then receives its second chunk, and the cycle continues. This interleaving dramatically reduces wait time for short requests—they no longer sit idle while the long request monopolizes the GPU.

The improvement is most dramatic for interactive workloads. Short chatbot queries that would previously wait for large document context processing can now begin generating responses almost immediately. The long request takes slightly longer overall due to scheduling overhead, but the system maintains responsiveness for all request sizes. Decode iterations can interleave with prefill chunks, ensuring running requests continue generating tokens even while new requests are being admitted.

### Integration with Decode Phase

Chunked prefill enables **mixed-phase execution**: prefill chunks and decode iterations run in the same time slice. Each iteration processes prefill chunks (token budget limited), then decode iterations (batch size limited), ensuring both make progress.

## 16.6 Prefill-Decode Separation

Prefill and decode phases have fundamentally different performance characteristics requiring specialized optimization strategies.

**Prefill Phase**. Prefill processes many tokens per request (100-1000) with quadratic attention complexity, making it compute-bound where the bottleneck is matrix multiplication throughput. The optimization goal is maximizing floating-point operations while minimizing memory writes, typically using smaller batches of requests (2-8) with longer sequences. Memory access patterns follow sequential KV cache writes as the cache populates from left to right.

**Decode Phase**. Decode generates one token per request with linear attention complexity over cached keys and values, making it memory bandwidth-bound where the bottleneck is loading KV cache entries. The optimization goal is maximizing memory bandwidth utilization through large batch parallelism, processing many requests simultaneously where each contributes just a single token. Memory access patterns involve random KV cache reads across many pages as attention attends to all previous tokens.

Different phases need different batch sizes—small for compute-bound prefill to avoid memory overflow, large for bandwidth-bound decode to saturate memory bandwidth.

### Scheduling Fundamentals

The different phase characteristics necessitate different scheduling strategies. Prefill uses FCFS (First-Come-First-Served) with token budget constraints to maintain fairness while preventing long prompts from monopolizing resources. The token budget caps the total tokens processed in a single iteration, ensuring predictable memory usage and latency bounds. Decode uses greedy batch formation up to max batch size, maximizing throughput by saturating memory bandwidth with as many concurrent requests as possible. Since each request generates exactly one token per iteration, the batch size directly determines parallelism.

### Two-Phase Managers

The serving system implements separate manager classes for prefill and decode, each optimized for its phase's characteristics. Understanding their distinct scheduling policies and APIs reveals how the system adapts to different computational patterns.

**PrefillManager Design**. This manager implements FCFS scheduling with token budget constraints. The core API is `schedule()`, which returns a prefill batch respecting the token budget while maintaining queue order. The manager maintains a FIFO queue of requests awaiting prefill processing. When `schedule()` is called, it iterates through the queue in order, accumulating requests until adding the next one would exceed the token budget. Selected requests are removed from the queue and packaged into a prefill batch using `Batch.from_prefill()`, which concatenates their uncached prompt tokens.

The token budget constraint serves multiple purposes: it prevents out-of-memory errors by capping maximum memory usage per iteration, it bounds iteration latency by limiting computation, and it enables predictable scheduling by ensuring no iteration processes unbounded work. The greedy packing algorithm maximizes utilization within the constraint.

**DecodeManager Design**. This manager batches all running requests up to a maximum size, prioritizing throughput over fairness. The core APIs are `schedule()` which returns a decode batch, and `remove_finished()` which filters completed requests from the running set. Since decode generates exactly one token per request per iteration, all running requests have identical computational cost. The manager simply batches as many as possible (up to `max_batch_size`) to saturate memory bandwidth.

The large batch size (typically 128-256) is critical for decode performance. Memory bandwidth is the bottleneck—loading KV cache entries dominates computation time. Processing 128 requests in parallel amortizes that memory access cost across more useful work, achieving much higher throughput than smaller batches. The `remove_finished()` method filters out completed requests each iteration, implementing the continuous batching pattern from section 16.3.

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

### Integration Pattern: Two-Phase Scheduler

The TwoPhaseScheduler coordinates both managers in a unified serving loop. Its `step()` method orchestrates one complete iteration of the two-phase execution pattern. Understanding this coordination reveals how requests flow through the serving system.

**Phase 1: Prefill Execution**. The scheduler calls `prefill_mgr.schedule()` to obtain a batch of requests needing prompt processing (or None if the prefill queue is empty). If a batch exists, the model executor's `forward()` method runs with the batch's concatenated token IDs and position indices, computing KV cache entries and logits. The scheduler then checks each request's completion status: when `cached_len` equals the prompt length, that request has finished prefill. For completed requests, the scheduler samples the first output token from the logits, adds the token to the request's output, inserts the completed prompt into the radix cache for future reuse, and transitions the request to decode by calling `decode_mgr.add_request()`.

**Phase 2: Decode Execution**. The scheduler calls `decode_mgr.schedule()` to batch all running requests (or None if no requests are running). If a batch exists, the model executor runs another forward pass, but now each request contributes just one token (the last generated token). The scheduler samples next tokens from the logits, extends each request's output sequence, and increments `cached_len`. Finally, `decode_mgr.remove_finished()` filters out completed requests—those that have generated their maximum tokens or produced an end-of-sequence marker.

**Seamless Transition**. Requests transition from prefill to decode within a single iteration. When a request finishes prefill in phase 1, it immediately becomes eligible for decode in the same iteration's phase 2 (if there's remaining iteration time) or the next iteration's phase 2 at latest. This tight coupling minimizes latency—no request sits idle waiting for a phase boundary. The two-phase pattern naturally interleaves prompt processing with token generation, maintaining high GPU utilization.

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

### NanoServing Architecture

The serving engine organizes computation as a pipeline where requests flow through multiple stages, each handling a specific aspect of the serving process. Understanding this dataflow is essential to comprehending how production LLM serving works. The pipeline consists of nine sequential stages that transform incoming requests into generated responses:

**Stage 1: Request Queue (FIFO)**. New requests from clients enter a first-in-first-out queue that buffers incoming traffic. This queue absorbs bursts of requests, preventing overload when many clients submit simultaneously. The queue provides backpressure—when it fills, the system can reject new requests or apply rate limiting rather than crashing from memory exhaustion.

**Stage 2: Radix Cache Lookup**. For each request dequeued for processing, the system queries the radix cache to find the longest matching prefix already computed by previous requests. The `match_prefix()` operation walks the cache tree, identifying how many prompt tokens can skip computation. This lookup is fast (linear in prompt length) and provides transparent optimization—structurally similar requests benefit from shared work without requiring explicit cache keys or annotations.

**Stage 3: KV Pool Allocation**. Based on the cache lookup result, the system allocates memory pages for the uncached portion of the prompt plus expected generation length. The allocator may trigger LRU eviction if free memory is insufficient, removing cold cached prefixes to make room. The allocation is atomic—either the request receives all needed memory or is rejected/delayed.

**Stage 4: Chunked Prefill Manager**. Admitted requests enter the prefill manager's queue for prompt processing. The manager implements round-robin chunking with token budget constraints, extracting fixed-size chunks from each request in turn. This ensures fairness—long prompts don't monopolize the GPU, and short prompts don't starve. The chunking mechanism divides prompt processing into schedulable units that interleave with decode.

**Stage 5: Decode Manager**. Requests that have completed prefill transition to the decode manager, which batches them for token generation. The manager uses greedy packing up to maximum batch size, prioritizing throughput by processing as many concurrent requests as possible. Finished requests are filtered out each iteration, implementing continuous batching.

**Stage 6: Model Executor (MLIR-compiled)**. Both prefill and decode batches execute through the model executor, which wraps the MLIR-compiled GPT from Chapters 1-15. The executor's `forward()` method accepts flattened token IDs and position indices, runs the transformer attention and feedforward layers, and returns logits for next token prediction. This is where actual computation happens—all previous stages handle scheduling and resource management.

**Stage 7: Radix Cache Insertion**. After prefill completes for a request, the system inserts that prompt path into the radix cache tree for future reuse. The `insert()` operation extends the tree only for tokens not already cached, linking new nodes to represent the unique suffix. This preserves completed work transparently, enabling automatic prefix sharing for subsequent requests.

**Stage 8: Response Queue**. Requests that finish generation (reaching max tokens or generating end-of-sequence) move to the response queue. The queue accumulates completed requests for batch retrieval by clients. Finished requests also trigger cleanup: deallocating their KV pages (returned to the pool) and updating radix cache access timestamps.

**Pipeline Flow**. Request lifecycle follows a left-to-right flow through these stages: admission → cache lookup → memory allocation → prefill scheduling → decode scheduling → execution → cache update → completion. Requests spend most time in prefill and decode stages (the hot path), while other stages handle fast bookkeeping. The radix cache provides a cross-cutting optimization, allowing requests to skip computation by reusing cached prefixes from structurally similar predecessors. Resource management threads through the entire pipeline—the KV pool tracks memory, the radix cache manages sharing and eviction, and the schedulers enforce capacity limits through continuous coordination.

### NanoServingEngine Implementation

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

### Request Dataflow

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

### Performance Summary

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