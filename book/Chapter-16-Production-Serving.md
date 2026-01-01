# Chapter 16: Production LLM Serving

Chapters 1-15 built a complete GPT model with GPU programming concepts. The implementation demonstrates transformer architecture, KV caching, and parallel execution patterns. However, building a production LLM serving system requires more than a correct model—it demands efficient request scheduling, memory management, and multi-request batching. This chapter introduces production serving techniques used in real-world systems like vLLM, SGLang, and TensorRT-LLM.

This chapter teaches the **inference framework architecture** in its first half—the scheduling algorithms, memory management strategies, and system design patterns that enable dramatic speedups over naive implementations. The second half demonstrates **our specific implementation** combining MLIR compiler acceleration with Python inference orchestration. This structure separates concepts (portable across systems) from implementation (specific to our MLIR-based approach). The algorithms presented here are used in production systems worldwide: vLLM pioneered paged attention and continuous batching, SGLang added radix caching for prefix sharing, and TensorRT-LLM provides NVIDIA's optimized stack. Our educational implementation demonstrates these same algorithmic foundations with clarity prioritized over maximum performance.

**Why Production Serving is Different**. A research implementation processes one request at a time: encode prompt, generate tokens sequentially, return result. Production systems serve **thousands of concurrent users** with diverse requirements—some need low latency (chat), others need high throughput (batch document summarization). The naive approach fails catastrophically:

Naive implementations process one request at a time, regenerating attention for the entire sequence at each step. This approach suffers from three fundamental inefficiencies. First, the quadratic redundancy problem: each generation step recomputes attention scores for all previous tokens, creating $O(N^2)$ wasted computation where N is the sequence length. Second, the sequential bottleneck: requests queue up waiting for the current request to complete token-by-token generation, leaving the GPU idle between requests. Third, memory over-allocation: the system reserves maximum sequence length memory for every request regardless of actual usage, preventing concurrent processing of multiple requests.

The core pattern demonstrates the problem:
```python
# Each iteration recomputes attention for ALL previous tokens
for _ in range(max_tokens):
    logits = model.forward(all_tokens)  # O(N²) attention!
    next_token = sample(logits[-1])
    all_tokens.append(next_token)  # Sequence grows, computation grows quadratically
```

Production systems solve these problems with **KV caching** (O(N²) → O(N)), **continuous batching** (parallel multi-request execution), and **paged memory management** (dramatically increased capacity). The combination enables serving many concurrent users interactively.

The practical impact becomes clear when comparing user experience. Naive implementations process one request at a time with sequential token generation, creating deep queues where users wait unacceptably long for responses. Production systems like vLLM and SGLang instead batch many requests together, generating tokens in parallel across the batch. This keeps queue depth minimal and response times short, delivering the interactive experience users expect from modern AI applications. This chapter explains the algorithmic techniques—KV caching, continuous batching, paged memory, and prefix sharing—that enable this dramatic transformation from sequential processing to high-throughput concurrent serving.

## 16.1 The Request Lifecycle

Production serving systems organize computation around **requests**—user tasks with specific prompts and generation parameters. Understanding request lifecycle and state management is foundational to serving architecture.

**Request Abstraction**. Each request represents a single user generation task, tracking both input specifications and dynamic state. The core abstraction maintains the original prompt tokens, generation parameters like maximum length and sampling temperature, and evolving state throughout execution. Two key counters govern processing: `cached_len` records how many tokens have their KV cache already computed, while the total token count (`device_len`) includes both prompt and generated output. The difference between these values determines `extend_len`—the tokens requiring forward passes in the next iteration. Requests also track allocated memory pages and lifecycle status as they transition from waiting in queue, through active generation, to completion.

**Example Request Lifecycle**:

```python
# Initial state: User submits "What is MLIR?" → tokens [1, 2, 3, 4, 5]
req = Request(
    prompt=[1, 2, 3, 4, 5],
    max_tokens=10,
    cached_len=0,     # No KV cache yet
    device_len=5      # 5 prompt tokens (computed property)
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

**Request States**. The serving system manages requests through three lifecycle states. Initially, requests enter the waiting state upon submission, queuing until the scheduler admits them for processing. The scheduler considers memory availability and batch capacity before transitioning requests to the running state, where they actively generate tokens through iterative forward passes. During the running phase, requests cycle through decode iterations, extending their output one token at a time. Finally, requests reach the finished state when they either generate their maximum configured tokens or emit an end-of-sequence marker, at which point they return results and free allocated resources.

**Batching Abstraction**. Multiple requests execute together in a **batch**:

```python
class Batch:
    """Group of requests processed together"""
    
    requests: List[Request]    # Requests in this batch
    is_prefill: bool          # True for prefill, False for decode
    
    # Prepared for GPU execution
    input_ids: np.ndarray     # [total_tokens] for prefill or [batch_size] for decode
    positions: np.ndarray     # Position indices for each token (for RoPE)
    out_loc: np.ndarray       # KV cache write locations
```

**Batch Phases**. Request processing divides into two distinct phases with fundamentally different performance characteristics. The prefill phase processes all prompt tokens in the first forward pass, computing and storing the complete KV cache for the input sequence. This phase achieves parallelism across the sequence length dimension, with computation-bound matrix multiplications dominating execution time. The decode phase handles subsequent iterations, generating one new token per step while reading the cached KV entries. Decode achieves parallelism across the batch dimension, but becomes memory bandwidth-bound as it loads large cached tensors for limited computation. Production systems exploit these differences: prefill benefits from fewer concurrent requests processing longer sequences, while decode thrives on large batches where many requests amortize memory access costs across parallel operations.

**Memory Requirements**. Request memory consumption varies dramatically:

```python
# Per-request KV cache memory (float16):
mem_per_token = (
    2                    # key + value
    * num_layers         # 12 layers (GPT-2)
    * num_kv_heads       # 12 heads
    * head_dim           # dimension size
    * 2                  # float16 = 2 bytes
)

# Example calculation (varies by model architecture):
# Small model: ~36 KB/token
# Larger models: more layers/heads → more memory per token
# Long sequences + many requests → substantial memory pressure
```

This memory pressure motivates **paged memory management** (Section 16.2) and **prefix sharing** (Section 16.4).

## 16.2 Paged KV Cache: Virtual Memory for Attention

The naive approach allocates max_seq_len memory for each request—wasteful when actual prompt lengths vary widely (50-2000 tokens). Production systems use **paged memory management**: divide KV cache into fixed-size pages, allocate on-demand.

**The Fragmentation Problem**. Contiguous allocation wastes memory:

```
Contiguous allocation (naive):
Request 1 (short prompt, allocated max):  ██░░░░░░░░░░░░░░░░ (substantial waste)
Request 2 (tiny prompt, allocated max):   █░░░░░░░░░░░░░░░░░ (massive waste)
Request 3 (long prompt, allocated max):   ███████████████░░░ (modest waste)

Most requests don't use their full allocation → wasted memory
```

**Paged Solution**. Divide KV cache into fixed-size pages (page size configurable):

```
Physical memory (pages):
[0] [1] [2] [3] [4] [5] [6] [7] [8] [9] ... [N]

Request 1 (short):
  Page table: [0, 1, 2, ...]
  Waste: partial page only

Request 2 (tiny):
  Page table: [13, 14, ...]
  Waste: partial page only

Request 3 (long):
  Page table: [17, 18, ..., 110, ...]
  Waste: partial page only

Total waste: only last page of each request (minimal vs contiguous)
```

**Page Table Translation**. The paging system maintains a mapping from logical token positions to physical memory locations. When accessing the KV cache for a specific token, the system first calculates which logical page contains that token through integer division by page size. The page table then maps this logical page number to a physical page in GPU memory. Finally, the offset within that page locates the exact cache entry. This indirection enables non-contiguous allocation while maintaining $O(1)$ access time—the same performance as contiguous memory but with dramatically better utilization.

For example, accessing token 42 in a request with 16-token pages involves three simple calculations: `logical_page = 42 // 16 = 2` identifies the third page, `offset = 42 % 16 = 10` finds the position within that page, and `physical_page = page_table[request_id][2]` retrieves the actual memory location. The final address combines the physical page start with the offset.

**Implementation Pattern**. The KV cache pool manages physical pages through a simple allocator interface. At initialization, the system creates large tensor storage for all layers and marks all pages as free. The allocation method receives a token count, calculates required pages through ceiling division, and selects that many pages from the free set. Deallocation simply returns pages to the free pool for reuse. The storage method takes computed key and value tensors along with their logical positions, translates those positions through the page table to physical addresses, and writes the data. This straightforward design provides efficient memory management without complex bookkeeping—the page table indirection handles all address translation transparently.

The key operations are:
```python
# Ceiling division ensures enough pages
pages_needed = (num_tokens + page_size - 1) // page_size

# Allocate from free pool
allocated = sorted(free_pages)[:pages_needed]
for page in allocated:
    free_pages.remove(page)
```

Paged memory dramatically increases capacity compared to contiguous allocation. The exact improvement depends on your request length distribution, but the key benefit is **avoiding worst-case allocation** for every request. Instead of allocating for maximum possible length, you allocate only what's actually used.

Consider a realistic scenario: 256 concurrent requests averaging 300 tokens each, with a worst-case maximum of 2048 tokens per request. Contiguous allocation reserves space for all 256 requests at their 2048-token maximum, consuming substantial memory even though most requests use only 15% of their allocation. Paged allocation with 16-token pages allocates only the 300 tokens actually used per request, reducing memory consumption dramatically. This efficiency translates directly to increased capacity—the same hardware can serve many more concurrent requests when memory isn't wasted on unused allocations. The result is much higher GPU utilization and better throughput for real workloads where request lengths vary significantly.
**Attention Implementation**. Standard attention operates on contiguous tensors where key and value matrices reside in continuous memory. The computation proceeds through three straightforward matrix operations: compute attention scores as query-key products, normalize scores through softmax, and compute output as score-weighted value sums.

Paged attention extends this pattern by first gathering scattered KV cache entries through page table translation. For each request, the system determines which pages contain the cached data, gathers those non-contiguous pages into a temporary contiguous buffer, and then performs standard attention computation on the gathered tensors. Production kernels like FlashAttention and Flash-Decoding fuse this page table translation directly into the attention kernel, avoiding the explicit gather step and achieving efficiency comparable to contiguous access. The key insight is that page translation adds minimal overhead—modern GPUs handle indirection efficiently, making paged attention nearly as fast as contiguous attention while enabling much better memory utilization.

## 16.3 Continuous Batching: Dynamic Request Scheduling

Naive batch serving processes all requests together—waits for the slowest request before accepting new requests. **Continuous batching** dynamically adds/removes requests every iteration, maximizing GPU utilization.

**Static Batching Problem**. Traditional batching processes all requests together, waiting for the slowest to complete before accepting new work. This creates severe underutilization: if one request generates 1000 tokens while others finish at 100 tokens, the GPU processes only that single request for 900 iterations while completed requests sit idle. The batch size drops from full capacity to nearly empty, yet the system cannot admit new requests until the entire batch finishes. This convoy effect—where one slow request delays many fast ones—fundamentally limits throughput.

The pattern is simple but inefficient:
```python
# Process batch together, all requests must complete
while any(r.status != "finished" for r in batch.reqs):
    logits = model.forward(batch)
    # Finished requests generate nothing, wasting their batch slot
```

```
Time →
Req 1: ████████████████████████████████████████ (1000 tokens)
Req 2: ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (100 tokens, then waits!)
Req 3: ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ (120 tokens, then waits!)
Req 4: ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (100 tokens, then waits!)

GPU Utilization: 4/4 (100%) → 1/4 (25%) for 880 iterations
```

**Continuous Batching Solution**. The key insight is that decode generates exactly one token per request per iteration. When a request finishes, its batch slot becomes immediately available for a waiting request. The serving loop maintains a running set of active requests, adding newly admitted requests and removing completed ones after every iteration. This dynamic composition keeps batch size near its maximum regardless of individual request completion times.

The core pattern involves three steps per iteration:
```python
# 1. Add new requests up to capacity
while len(running) < MAX_BATCH and request_queue:
    running.append(request_queue.pop(0))

# 2. Execute one iteration for all running requests  
batch = Batch(running)
logits = model.forward(batch)

# 3. Update and filter - finished requests removed, slots freed
running = [r for r in running if not r.is_finished()]
```

This simple change transforms utilization. Instead of dropping to 25% efficiency when 3 out of 4 requests finish early, the system immediately admits new requests to fill those slots, maintaining near-maximum batch size continuously.

**Utilization Improvement**:

```
Time →
Req 1: ████████████████████████████████████████ (1000 tokens)
Req 2: ██████████ (100 tokens, finishes)
Req 3: ████████████ (120 tokens, finishes)
Req 4: ██████████ (100 tokens, finishes)
Req 5:           ████████████████ (150 tokens, joins after Req 2 finishes)
Req 6:                     ████████████████████ (200 tokens, joins after Req 4)
Req 7:                                    ████████ (80 tokens, joins...)

GPU Utilization: 4/4 → 3/4 → 4/4 → 4/4 → ... (consistently high!)
```

**Throughput Comparison**:

```
Static batching:
  - 4 requests × 1000 max_tokens = 4000 token budget
  - Actual: 1320 tokens generated
  - Efficiency: 33%
  - Throughput: 1320 tokens / 1000 steps = 1.32 tok/step

Continuous batching:
  - Same 4000 token budget
  - Process 7 requests (finished earlier requests free slots)
  - Actual: 1850 tokens generated
  - Efficiency: 46%
  - Throughput: 1850 tokens / 1000 steps = 1.85 tok/step
  - Speedup: 1.4×
```

**Request Pool Management**. Production systems organize requests across three collections corresponding to their lifecycle states. The waiting queue holds newly submitted requests until admission. The running list tracks actively generating requests participating in decode batches. The finished collection stores completed requests ready for result retrieval. The pool provides methods to transition requests between these states: admit moves waiting requests to running based on capacity and memory availability, finish moves running requests that reach completion conditions to the finished set, and get_batch constructs execution batches from the current running set. This clean separation of concerns simplifies the scheduling logic—each component interacts with the pool through well-defined state transitions rather than managing complex request tracking directly.

**Scheduling Loop**:

```python
def serving_loop(pool, model):
    """Main continuous batching loop"""
    
    while True:
        # 1. Admit new requests
        pool.admit_requests(capacity=MAX_BATCH_SIZE)
        
        # 2. Get current batch
        batch = pool.get_batch()
        
        if not batch.reqs:
            time.sleep(0.001)  # Nothing to do
            continue
        
        # 3. Execute forward pass
        logits = model.forward(batch)
        next_tokens = sample(logits)
        
        # 4. Update requests
        for req, token in zip(batch.reqs, next_tokens):
            req.append_token(token)
        
        # 5. Remove finished requests
        pool.finish_requests()
```

This loop runs continuously, dynamically adjusting batch size every iteration—the core of production serving.

## 16.4 Radix Cache: Automatic Prefix Sharing

Many requests share common prefixes: system prompts, few-shot examples, document context. Computing KV cache independently for each request wastes computation. **Radix cache** automatically detects and reuses KV cache for shared prefixes—2-3× speedup in realistic workloads.

**The Prefix Reuse Problem**. Consider chatbot requests with system prompt:

```python
system_prompt = "You are a helpful assistant. Answer concisely."
# Tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (10 tokens)

# User requests
req1 = system_prompt + "What is Python?"     # [1..10] + [20, 21, 22]
req2 = system_prompt + "Explain recursion."  # [1..10] + [30, 31]
req3 = system_prompt + "Debug this code:"    # [1..10] + [40, 41, 42, 43]
```

Naive approach computes KV cache for `[1..10]` three times—identical computation. If system prompt is 1000 tokens and user query is 20 tokens, we waste 98% of prefill computation!

**Prefix Detection Challenge**. How do we automatically detect shared prefixes without user annotation?

```python
# Need to detect:
[1, 2, 3, 4, 5] shared by:
  [1, 2, 3, 4, 5, 6, 7]
  [1, 2, 3, 4, 5, 8, 9]
  [1, 2, 3, 4, 5, 10]

# And handle tree structure:
[1, 2, 3]
  ├─ [1, 2, 3, 4, 5]
  │   ├─ [1, 2, 3, 4, 5, 6, 7]
  │   └─ [1, 2, 3, 4, 5, 8, 9]
  └─ [1, 2, 3, 10, 11]
```

**Radix Tree Solution**. Before diving into implementation, let's understand what radix trees are and why they're perfect for prefix caching.

**What is a Radix Tree?** A radix tree (also called radix trie or compressed trie) is a space-optimized tree data structure where nodes can represent sequences of elements, not just single elements. Unlike a standard trie where each node represents one character/token, radix tree nodes can store multiple tokens, reducing tree depth and memory overhead.

**Key Properties**:
1. **Path Compression**: Chains of nodes with single children are compressed into one node storing the entire sequence
2. **Prefix Matching**: Any path from root to node represents a shared prefix among requests
3. **Efficient Lookup**: Finding longest matching prefix is O(m) where m is the length of the query sequence
4. **Space Efficient**: Fewer nodes than standard tries, especially when sequences share long prefixes

**Comparison: Standard Trie vs Radix Tree**:

```
Standard Trie (one token per node):
         root
          |
          1
          |
          2
          |
          3
        /   \
       4     10
       |      |
       5     11
      / \
     6   8
     |   |
     7   9

Radix Tree (compressed paths):
       root
         |
    [1,2,3]
      /   \
  [4,5]  [10,11]
   / \
[6,7] [8,9]

Nodes: 14 → 6 (57% reduction!)
Height: 7 → 3 (57% reduction!)
```

**Why Radix Trees for KV Cache?** Three reasons make radix trees ideal for LLM serving:

1. **Natural Prefix Structure**: Real workloads have common prefixes (system prompts, few-shot examples, document context). A radix tree naturally exploits this structure.

2. **Storage Efficiency**: Each node can reference physical KV cache pages directly. Shared prefixes share pages—automatic deduplication.

3. **LRU Eviction**: Tree structure makes it easy to find least-recently-used leaf nodes for eviction when memory is full.

**Radix Tree for Token Sequences**. In our case, each node stores:
- Token ID it represents (single token per node in our implementation)
- KV cache pages allocated for this position
- Children map (token → child node ID)
- Metadata (last access time for LRU)

**Arena-Based Memory Management**: Modern implementation uses integer node IDs instead of pointers:
- All nodes stored in contiguous `std::vector<RadixNode>` (the "arena")
- Nodes referenced by `NodeID` (just an `int`), not pointers
- No reference counting needed—simple leaf checking for eviction
- Cache-friendly, safe, educational pattern

```
Root (empty)
 ├─ Node([1, 2, 3]) → pages [0, 1, 2]
 │   ├─ Node([4, 5]) → pages [3, 4]
 │   │   ├─ Node([6, 7]) → pages [5, 6] ← Request 1 path
 │   │   └─ Node([8, 9]) → pages [7, 8] ← Request 2 path
 │   └─ Node([10, 11]) → pages [9, 10] ← Request 3 path
 └─ Node([50, 51]) → pages [11, 12] ← Different prefix
```

When Request 2 arrives with tokens `[1, 2, 3, 4, 5, 8, 9]`, we traverse:
1. Match `[1, 2, 3]` → reuse pages [0, 1, 2]
2. Match `[4, 5]` → reuse pages [3, 4]
3. Create new branch for `[8, 9]` → allocate pages [7, 8]

**Result**: Only 2 new pages allocated instead of 7! This is the power of prefix sharing.

The essential structure:
```cpp
// Type-safe handle, not a pointer!
using NodeID = int;
const NodeID INVALID_NODE = -1;

class RadixNode {
    int token_;                        // Token this node represents  
    std::map<int, NodeID> children_;   // token → child ID (not pointer!)
    std::vector<int> kv_pages_;        // Physical pages
    double last_access_time_;          // For LRU
    
    bool is_leaf() const { return children_.empty(); }
};

class RadixCache {
    std::vector<RadixNode> nodes_;     // Arena storage
    NodeID root_id_;                   // Root is just an int
};
```

Python accesses nodes via `cache.get_node(node_id)` returning property dictionaries, keeping memory management in C++ while providing clean read access.

**Python Integration**: Thin Python wrappers call C++ radix cache methods to match prefixes, retrieve cached pages, and insert new paths. The wrapper queries `cache.match_prefix(tokens)` to find how many tokens are already cached, fetches those cached page IDs, allocates additional pages for any uncached tokens (triggering LRU eviction if needed), and inserts the complete token sequence with its full page list back into the tree.

**Request Processing with Radix Cache**:

```python
def process_request_with_cache(req, radix_cache):
    """Process request using radix cache for prefix reuse"""
    
    # 1. Find cached prefix
    matched_len, node_id = radix_cache.match_prefix(req.tokens)
    
    # 2. Set request state to skip cached tokens
    req.cached_len = matched_len  # Skip this many tokens in prefill
    
    # 3. After prefill completes, insert into cache for future reuse
    # (Done by scheduler after forward pass)
    if req.kv_pages and len(req.kv_pages) == len(req.tokens):
        radix_cache.insert(req.tokens, req.kv_pages)
    
    return req.extend_len  # Tokens needing forward pass
```

In practice, the scheduler checks for prefix matches before scheduling prefill, then inserts completed prompts into the cache after prefill finishes.

**Cache Hit Rate Analysis**. The value of radix caching depends entirely on your workload's prefix sharing patterns:

**High Hit Rate Workloads** (substantial shared prefixes):
- Chatbots with system prompts—every request shares the system message
- Few-shot prompting—same examples prepended to all queries
- Document Q&A—shared document context across questions
- Multi-turn conversations—growing shared history

**Low Hit Rate Workloads** (mostly unique requests):
- Diverse document summarization—each document is unique
- Creative writing—no prefix overlap between stories
- Random query workload—no structure to exploit

**Key Insight**: When a prefix is shared, it's computed once and reused by all subsequent requests with that prefix. The speedup comes from skipping redundant computation on the shared portion. Workloads with substantial prefix sharing see meaningful benefits; workloads with little sharing gain little from radix caching.

**LRU Eviction Implementation**. When memory is full, the system evicts least-recently-used branches:

```cpp
// C++ implementation (simplified)
pair<NodeID, vector<NodeID>> RadixCache::find_lru_leaf() {
    // Recursively traverse tree to find leaf with oldest access time
    return find_lru_recursive(root_id_, {});
}

vector<int> RadixCache::evict_leaf(NodeID leaf_id, vector<NodeID> path) {
    RadixNode& leaf = get_node(leaf_id);
    
    if (leaf.is_leaf()) {  // Leaf nodes can be evicted
        vector<int> freed_pages = leaf.get_kv_pages();
        
        // Remove leaf from parent
        NodeID parent_id = path.back();
        get_node(parent_id).remove_child(leaf.get_token());
        
        // Mark node as inactive (arena slot can be reused)
        free_node(leaf_id);
        
        return freed_pages;
    }
    return {};
}
```

**Eviction Strategy**:
- Leaf nodes (no children) can be evicted—they represent unique suffixes not shared by other requests
- Internal nodes (have children) represent shared prefixes—preserve them
- LRU ordering ensures recently-used paths survive longest
- The algorithm preserves hot prefixes (like system prompts) while removing cold unique continuations

Radix cache with LRU eviction balances cache hit rate (keep hot prefixes) with memory efficiency (evict cold prefixes).

## 16.5 Chunked Prefill: Fair Scheduling for Long Contexts

Prefill phase processes all prompt tokens at once—for 2000-token prompts, this blocks GPU for 100-200ms. During this time, short prompts (50 tokens, 5ms) wait in queue. **Chunked prefill** splits long prompts into chunks, interleaving with short prompts and decode batches for fair scheduling.

**The Long Prompt Problem**. Consider serving scenario:

```
Queue at t=0:
  Req 1: 2000-token prompt (200ms prefill)
  Req 2: 50-token prompt (5ms prefill)
  Req 3: 100-token prompt (10ms prefill)
  Req 4-20: Decode iterations (1ms each)

Naive scheduling (FCFS):
  t=0-200ms: Req 1 prefill (GPU 100% busy)
  t=200ms: Req 2 prefill (waited 200ms!)
  t=205ms: Req 3 prefill (waited 205ms!)
  ...
```

Short requests starve—unacceptable for interactive workloads.

**Chunking Strategy**. Split long prompts into fixed-size chunks (e.g., 256 tokens). The system wraps requests to track progress through long prompts, dividing the full token sequence into fixed-size chunks and maintaining a counter of which chunk to process next. This simple state enables resumable processing where each scheduling round advances one chunk and increments the position counter.

**Chunked Scheduling**. The scheduler maintains a queue of requests needing prefill, each potentially spanning multiple chunks. A token budget limits the total tokens processed per iteration, preventing memory overflow and bounding latency. The algorithm iterates through the prefill queue in round-robin fashion: for each request, extract its next chunk, check if adding it would exceed the token budget, and if space remains, add the chunk to the current batch and move the request to the back of the queue for fairness. This continues until the budget fills or all requests have contributed their next chunk, ensuring short requests don't starve behind long ones.

**Scheduling Timeline with Chunking**:

```
Time →
Req 1 (long):  [chunk] .... [chunk] ... [chunk] ... [chunk] ... [chunk] ...
Req 2 (short):        [short]
Req 3 (mid):                 [mid]
Decode batch:         [decode] [decode] [decode] [decode] [decode]

Req 2 wait time: One chunk (vs full long prefill blocking)
Req 3 wait time: Chunk + short request (vs full blocking)
```

**Fairness Benefits**:

Chunked prefill dramatically improves latency for short requests in mixed workloads. The exact improvement depends on hardware, model size, and batch composition—but the principle is universal: **avoid blocking the GPU with long-running operations when short requests are waiting**.

Production systems use adaptive chunk sizes based on current load:
- More running requests → smaller chunks (prioritize decode and fairness)
- Fewer running requests → larger chunks (maximize throughput)

The tradeoff is typically a small throughput reduction (batch efficiency) for large latency improvements on short requests—essential for interactive serving.

**Token Budget Management**. Prefill and decode compete for GPU time:

```python
def adaptive_token_budget(running_reqs, target_latency_ms=50):
    """Dynamically adjust prefill token budget based on load"""
    
    # More running requests → smaller prefill budget (prioritize decode)
    # Fewer running requests → larger prefill budget (maximize throughput)
    
    if len(running_reqs) > 64:
        budget = 128  # High load: small chunks
    elif len(running_reqs) > 32:
        budget = 256  # Medium load: medium chunks
    else:
        budget = 512  # Low load: large chunks
    
    return budget
```

Production systems adaptively adjust budget based on current load—balancing fairness (small chunks) with efficiency (large chunks).

## 16.6 Prefill-Decode Separation

Prefill and decode phases have fundamentally different performance characteristics requiring specialized optimization strategies. Prefill processes many tokens per request with quadratic attention complexity, making it compute-bound where the bottleneck is matrix multiplication throughput. The optimization goal is maximizing floating-point operations while minimizing memory writes, typically using smaller batches of requests with longer sequences. Decode generates one token per request with linear attention complexity over cached keys and values, making it memory bandwidth-bound where the bottleneck is loading KV cache entries. The optimization goal is maximizing memory bandwidth utilization through large batch parallelism, processing many requests simultaneously where each contributes just a single token.

**Scheduling Fundamentals**. LLM serving schedulers decide which requests to execute and in what order at each iteration, managing GPU compute and memory resources rather than assigning CPU time like traditional OS schedulers. The scheduler makes four key decisions: selecting which waiting requests should start prefill, forming batches to group requests for parallel execution, allocating memory for KV cache and compute tokens per batch, and balancing throughput against latency to achieve fairness across requests.

**Scheduling Strategies**. Production systems adapt traditional OS scheduling to GPU resource management. Prefill uses First-Come-First-Served with token budgets to maintain arrival order fairness while preventing long prompts from blocking short ones. Decode uses all-at-once batching to maximize parallelism, processing all running requests simultaneously since each generates exactly one token. The phase separation is necessary because prefill has variable compute requirements depending on prompt length and memory-constrained admission, while decode has fixed per-request compute and benefits from maximizing batch parallelism.

**Resource Management**. Token budgets limit total tokens processed in a single prefill batch, providing memory predictability since the budget translates directly to maximum memory footprint, bounding prefill latency, and ensuring no single long prompt monopolizes the GPU. Maximum batch size limits concurrent requests in decode phase, preventing out-of-memory errors since each request holds KV cache, controlling per-iteration latency, and respecting hardware limits. Production systems tune these parameters based on workload characteristics and hardware capacity.

**Scheduler Implementation**. Production systems implement specialized schedulers for each phase, allowing independent optimization without complex conditional logic. The prefill scheduler implements FCFS with token budget constraints, iterating through the waiting queue and adding requests until the cumulative token count would exceed the budget, thus maintaining arrival order fairness while preventing convoy effects. The decode scheduler simply batches all running requests up to the maximum batch size, since decode generates exactly one token per request regardless of sequence length. This architectural separation enables each scheduler to focus on its phase-specific constraints and optimization goals.

**Execution Loop**. The serving loop coordinates both schedulers in each iteration: query the prefill scheduler for waiting requests that fit within the token budget, query the decode scheduler for all running requests up to batch size limit, execute any prefill batch and transition completed requests to decode phase, execute any decode batch and remove finished requests, then repeat. This two-phase coordination naturally interleaves prompt processing with token generation, maintaining high GPU utilization.

**Phase-Specific Optimization**. Each phase can use specialized attention kernels optimized for its bottleneck. Prefill typically uses FlashAttention to minimize memory traffic during compute-bound operations, while decode uses specialized kernels that parallelize across the sequence dimension to maximize memory bandwidth utilization when loading KV cache entries.

## 16.7 Production System Architecture

Production LLM serving systems layer components with clean interfaces: API servers handle HTTP requests, tokenizers convert text to token IDs, schedulers manage request queues with admission control, and execution engines run model forward passes while coordinating KV cache and radix cache. The scheduler's main loop coordinates three phases—prefill processing with chunking, decode execution for running requests, and admission of waiting requests to fill capacity. Communication uses ZMQ/gRPC for inter-process messaging and direct Python/C++ interfaces for model execution.

**Note**: Our nano-serving implementation (sections 16.10-16.19) uses simplified single-process design with direct `ContinuousBatcher` to model executor calls, demonstrating core algorithms without production complexity.

## 16.8 Performance Analysis

These complementary optimizations deliver 30-100× throughput improvements: parallel batching eliminates sequential bottlenecks, paged KV cache removes fragmentation, continuous batching maintains GPU utilization, and radix cache reduces redundant prefix computation. Effectiveness varies with workload characteristics.

## 16.9 Summary

The framework architecture (sections 16.1-16.6) covered five core techniques enabling production LLM serving: **paged KV cache** treats attention memory like OS virtual memory with on-demand allocation, eliminating fragmentation; **continuous batching** dynamically adds/removes requests every step, maintaining GPU utilization; **radix cache** shares computed KV entries across requests with common prefixes; **chunked prefill** splits long prompts into chunks to prevent starvation of short requests; and **prefill-decode separation** optimizes each phase independently (compute-bound vs memory-bound).

These algorithms, pioneered by vLLM, SGLang, and TensorRT-LLM, deliver substantial throughput improvements through complementary optimizations. Production systems compose API servers, tokenizers, schedulers, and execution engines with clean interfaces, coordinating request lifecycle management from submission through completion.

The following sections demonstrate these concepts through nano-serving, combining Python orchestration with MLIR-compiled execution to validate the algorithms and provide a foundation for understanding production system implementations.

## 16.10 Implementation Overview

The nano-serving implementation demonstrates production LLM serving concepts—paged KV cache, continuous batching, radix cache, chunked prefill—through **our specific implementation** combining MLIR compiler acceleration with Python inference orchestration.

This chapter builds nano-serving in six progressive phases, covering both **core components** (Phases 0-2) and **advanced features** (Phases 3-6). By the end, you'll understand how Python scheduling coordinates with C++ model execution through clean interfaces, and how production systems achieve performance through algorithmic innovation.

**Implementation Philosophy**: Production systems prioritize throughput—CUDA kernels, GPU-optimized attention, multi-GPU tensor parallelism. Our implementation prioritizes **clarity**—readable Python, clean C++ bindings, educational structure. The algorithms are identical; the optimization level differs. Understanding our implementation prepares you to read vLLM/SGLang source code.

**Architecture Overview**. The nano-serving implementation splits responsibilities between two layers connected through pybind11 bindings. The Python layer handles orchestration and control flow, managing request and batch abstractions, wrapping the KV pool for memory management, implementing prefill and decode scheduling managers, maintaining the radix cache prefix tree, and coordinating the continuous batching main loop. The C++/MLIR layer focuses on performance-critical execution, implementing the paged KV cache pool for efficient memory allocation, integrating the GPT model JIT-compiled from Chapter 14, and optimizing forward pass execution. This separation leverages Python's expressiveness for complex scheduling logic while delegating computation-heavy operations to compiled C++ for maximum performance.

**Why This Split?** Python excels at high-level logic (tree operations, scheduling decisions, request management). C++ excels at performance-critical code (matrix operations, memory management, tight loops). Combining both leverages each language's strengths.

**Development Approach**: Build incrementally with tests at each phase. Each component has clear inputs/outputs and can be tested independently before integration. This matches real-world development—validate components before assembling the system.

## 16.11 Phase 0: Request and Batch Abstractions

Before implementing schedulers or memory management, we need data structures representing user tasks and batched execution.

**Request Class**: Tracks a single generation task through its lifecycle. Each request maintains state distinguishing between cached tokens (already have KV cache computed) and extend tokens (need processing this iteration). The key state properties include the request identifier, prompt token list, maximum generation length, current cached length tracking how many tokens have computed KV cache, growing output token list, allocated KV page list for memory management, and optional per-layer K and V cache tensors. The `extend_len` property encapsulates the prefill-to-decode transition elegantly: it returns the full uncached prompt length during prefill, then drops to exactly 1 once generation begins, representing just the most recently generated token needing processing.

The lifecycle progression demonstrates the state evolution:
```python
# Initial: User submits "What is MLIR?" → tokens [1,2,3,4,5]
req.cached_len == 0         # No KV cache yet
req.extend_len == 5         # Need to process 5 prompt tokens

# After prefill completes
req.cached_len = 5          # All prompt cached
req.output_tokens = [42]    # First token generated  
req.extend_len == 1         # Only new token needs processing
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

## 16.12 Phase 1: KV Cache Pool (C++ Implementation)

KV cache stores attention keys and values for all previous tokens. Naive allocation (max_seq_len per request) wastes memory. **Paged allocation** divides cache into fixed-size pages, allocating on-demand.

**C++ Implementation** (performance-critical memory management). The KV cache pool implements paged memory allocation in C++ for performance, managing allocation through a free page set and ceiling division for page calculation. The essential paging algorithm tracks available pages in a set, computes required pages through `(num_tokens + page_size - 1) / page_size` for ceiling division, checks availability, and allocates by removing pages from the free set. Deallocation simply returns pages for reuse. The `std::set<int>` provides $O(\log n)$ operations; production systems optimize with $O(1)$ free lists.

The core pattern:
```cpp
class KVCachePool {
    std::set<int> free_pages_;  // Track available pages
    
    std::vector<int> allocate(int num_tokens) {
        int pages_needed = (num_tokens + page_size_ - 1) / page_size_;
        if (free_pages_.size() < pages_needed)
            throw std::runtime_error("Out of memory");
        
        // Allocate from free pool
        std::vector<int> allocated;
        auto it = free_pages_.begin();
        for (int i = 0; i < pages_needed; i++)
            allocated.push_back(*it++);
        return allocated;
    }
};
```

**Python Bindings**: The pybind11 module exposes C++ classes to Python with automatic type conversion, allowing Python code to instantiate the KV pool, call allocation and deallocation methods, and query available capacity.

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

## 16.13 Phase 2: Prefill and Decode Managers

Prefill (process prompt) and decode (generate tokens) have different characteristics. Separate managers optimize each phase independently.

**Prefill Manager** schedules prompt processing using FCFS with token budget constraints. The token budget prevents out-of-memory errors from processing too many long prompts simultaneously while maintaining arrival order fairness. The greedy packing algorithm iterates through the FIFO queue, adding requests whose token counts fit within the remaining budget. This approach is optimal for FCFS: it serves the maximum number of waiting requests while respecting budget limits and preserving submission order.

The scheduling logic:
```python
class PrefillManager:
    def schedule(self):
        selected, total_tokens = [], 0
        while self.queue:
            if total_tokens + self.queue[0].extend_len <= self.token_budget:
                selected.append(self.queue.pop(0))
                total_tokens += selected[-1].extend_len
            else:
                break  # Would exceed budget
        return Batch.from_prefill(selected) if selected else None
```

**Decode Manager** batches all running requests since each generates exactly one token. The batch size is bounded only by the maximum configured capacity, not token count like prefill. The scheduling logic is straightforward: collect all currently running requests up to the maximum batch size, form a decode batch, execute, then filter out finished requests. Decode throughput scales linearly with batch size since GPU parallelism spreads across the batch dimension—256 concurrent requests generate 256 tokens per iteration.

The pattern:
```python
class DecodeManager:
    def schedule(self):
        if not self.running:
            return None
        batch_reqs = self.running[:self.max_batch_size]
        return Batch.from_decode(batch_reqs)
    
    def remove_finished(self):
        self.running = [r for r in self.running if not r.is_finished]
```

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

## 16.14 Model Executor: MLIR Integration

The executor wraps the GPT model from Chapter 14, providing a clean interface for prefill and decode execution.

**Model Executor**: Wraps the GPT model from Chapter 14, providing unified interfaces for both prefill and decode execution. The key complexity lies in extracting per-request logits from batched output. Prefill returns many tokens per request but only the last token's logits matter for sampling—the executor must slice the concatenated output appropriately. Decode returns one token per request, so all logits are used directly. This asymmetry reflects the different batch construction patterns between the two phases.

The extraction logic:
```python
class ModelExecutor:
    def execute_prefill(self, batch):
        logits = self.model.forward(batch.input_ids, batch.positions)
        # Extract last token logits for each request
        result, offset = [], 0
        for req in batch.requests:
            result.append(logits[offset + req.extend_len - 1])
            offset += req.extend_len
        return np.array(result)
    
    def execute_decode(self, batch):
        return self.model.forward(batch.input_ids, batch.positions)
```

## 16.15 Integration Testing and Data Flow

Phase 2 components work together through well-defined interfaces. Integration tests validate the complete data flow.

**Data Flow**: User requests become Python objects that flow through the prefill manager into batches for model execution. The Python/C++ boundary crossing happens during model forward passes, where MLIR-compiled code executes and writes to KV cache. Logits return to Python for token sampling, then decode processing iterates until completion. Scheduling and lifecycle management remain in Python for flexibility, while performance-critical computation happens in compiled C++.

**Integration Validation**: The test suite verifies end-to-end flow from request creation through prefill→decode transitions to completion. Key invariants: `cached_len` increases monotonically, `extend_len` transitions from prompt length to 1, KV pages are allocated before prefill and freed after completion.

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

**Core Chunking State**. The system tracks progress through long prompts using a thin wrapper that maintains how many tokens have been processed and provides methods to retrieve subsequent chunks. Each chunked request references the underlying full request while adding a `tokens_processed` counter starting from the request's cached length. The `has_more_chunks` property simply checks if processing has reached the end of the prompt. The `get_next_chunk` method slices the next chunk worth of tokens, typically 256-512 based on configuration, advancing from the current position. This straightforward design enables resumption from any point—the scheduler iterates through chunked requests, processing one chunk per iteration until all prompts complete.

The essential pattern:
```python
class ChunkedRequest:
    def __init__(self, request, chunk_size=512):
        self.request = request
        self.tokens_processed = request.cached_len  # Resume point
    
    def get_next_chunk(self):
        start = self.tokens_processed
        end = min(start + self.chunk_size, len(self.request.prompt_tokens))
        return self.request.prompt_tokens[start:end]
```

Round-robin scheduling is simple but not optimal for all scenarios. Production systems often use priority queues where each request receives a score based on estimated completion time divided by chunk size, balancing fairness for small requests with throughput for large ones. This prevents both starvation of large requests by many small requests and blocking of small requests by large contexts. Prefill and decode phases can coexist in the same serving loop—while chunking processes a new request's prompt, the system simultaneously generates output tokens for requests already in decode phase.

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

Each radix tree node represents a single token position in the sequence and stores the corresponding KV cache pages in physical memory. The node maintains a map from token IDs to child node IDs (integers, not pointers!), enabling fast traversal when looking up prefixes.

**Educational Design Choice**: Our implementation uses **arena-based allocation** with integer node IDs instead of pointers. This demonstrates production-quality patterns without manual memory management:
- All nodes stored in `std::vector<RadixNode>` (contiguous arena)
- Nodes referenced by `NodeID` (type-safe integer handle)
- No reference counting—eviction checks if node is a leaf (has no children)
- No `shared_ptr`, no `use_count()`, no pointer arithmetic

The last access timestamp supports LRU eviction by identifying which branches of the tree have gone unused for the longest time.

**Core Node Structure** (from [`radix_cache.h`](../ch.16.Nano-Serving/src/radix_cache.h)):

```cpp
// Type-safe node handle — not a pointer!
using NodeID = int;
constexpr NodeID INVALID_NODE = -1;

class RadixNode {
    int token_;                        // Token this node represents
    std::vector<int> kv_pages_;        // Physical page indices
    std::map<int, NodeID> children_;   // token → child node ID (not pointer!)
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
    // No reference counting!
};

class RadixCache {
    std::vector<RadixNode> nodes_;      // Arena: all nodes live here
    std::vector<bool> node_active_;     // Track active slots
    NodeID root_id_;                    // Root is just an integer
    
    NodeID allocate_node(int token);    // Arena allocation
    void free_node(NodeID id);          // Mark slot inactive
    RadixNode& get_node(NodeID id);     // Safe access with bounds check
};
```

The simplicity is deliberate: nodes are lightweight containers stored in a contiguous arena. No pointers, no manual memory management, no reference counting. Python accesses nodes via `cache.get_node(node_id)` which returns a dictionary of properties.

### 16.17.3 Cache Lookup Algorithm

When a new request arrives with tokens `[t1, t2, t3, ...]`, walk the tree to find the longest prefix match:

**Prefix Matching Logic** (from [`radix_cache.cpp`](../ch.16.Nano-Serving/src/radix_cache.cpp)):

```cpp
std::pair<int, NodeID> RadixCache::match_prefix(const std::vector<int>& tokens) {
    NodeID current_id = root_id_;
    int matched_len = 0;

    for (size_t i = 0; i < tokens.size(); ++i) {
        RadixNode& node = get_node(current_id);  // Arena access, not pointer!
        NodeID child_id = node.get_child(tokens[i]);
        
        if (child_id != INVALID_NODE) {
            current_id = child_id;     // Advance in tree (just integer assignment)
            matched_len = i + 1;       // Count matched tokens
            get_node(current_id).update_access_time();  // Update LRU
        } else {
            break;  // No match, stop here
        }
    }

    return {matched_len, current_id};
}
```

**Python sees**:
```python
matched_len, node_id = radix_cache.match_prefix(tokens)
# node_id is just an int, not a pointer!
```

**Example execution**:
- Tree contains path [1, 2, 3]
- New request: [1, 2, 3, 6, 7]
- Lookup walks: root → [1] → [2] → [3] → (no child for 6)
- Returns: `matched_length=3`, reuse KV cache for tokens [1,2,3]
- Need to compute: tokens [6, 7] starting from node representing [1,2,3]

The algorithm is optimal: $O(n)$ where $n$ is the shorter of (token sequence length, tree depth). No backtracking, no hash lookups—pure tree traversal.

### 16.17.4 Memory Management and Eviction

**Physical Memory Layout**: KV cache stored in **pages** (e.g., 16 tokens per page). Each radix node holds references to physical pages in the KV pool.

**Problem**: Tree grows unbounded as requests add new paths. GPU memory is limited (40GB on A100, stores ~100K tokens with Llama-70B).

**Solution**: **LRU eviction** removes unused branches when memory pressure occurs:

```cpp
// Find LRU leaf — no reference counting needed!
std::pair<NodeID, std::vector<NodeID>> RadixCache::find_lru_leaf() {
    return find_lru_recursive(root_id_, {});
}

// Evict a leaf node
std::vector<int> RadixCache::evict_leaf(NodeID leaf_id, std::vector<NodeID> path) {
    RadixNode& leaf = get_node(leaf_id);
    
    if (leaf.is_leaf()) {  // Simple check — no use_count()!
        std::vector<int> freed_pages = leaf.get_kv_pages();
        
        // Remove from parent
        NodeID parent_id = path.back();
        get_node(parent_id).remove_child(leaf.get_token());
        
        // Mark arena slot as inactive (can be reused)
        free_node(leaf_id);
        
        return freed_pages;
    }
    return {};
}
```

**Why This Works Without Reference Counting**:
1. Leaf nodes (no children) can be evicted — they're not shared!
2. Internal nodes (have children) represent shared prefixes — keep them
3. LRU traversal finds oldest leaf by comparing access times
4. No manual tracking of "who uses this node" needed

The eviction policy preserves hot paths while removing cold ones. Frequently accessed sequences like system prompts remain in the cache because their access timestamps constantly update. One-time queries with unique continuations become eviction candidates because they're leaves. The leaf-first removal strategy ensures shared prefixes near the tree root survive longer than unique continuations near the leaves, maintaining the most valuable cached state.

The memory efficiency gain can be dramatic. Consider one hundred requests sharing a five-hundred-token prefix. Without radix caching, the system stores fifty thousand token entries redundantly—each request maintains its own copy of the identical prefix. With radix caching, only five hundred tokens need storage, shared across all requests. This represents a ninety-nine percent reduction in memory consumption for the shared portion.

### 16.17.5 Integration with Request Lifecycle

During prefill, the system first looks up the request's token sequence in the radix tree to find the longest matching prefix. It reuses the existing KV cache entries for all matched tokens and computes new entries only for the unmatched portion. As computation proceeds, the system inserts new nodes into the tree (allocating them in the arena) representing the unique continuation of this request.

The decode phase extends the cached sequence with each generated token. As the model produces new tokens, the system creates corresponding child nodes in the radix tree and stores their KV vectors. If multiple requests share the same generation path (rare but possible in beam search or when temperature is zero), they share these decode-phase nodes as well.

When a request completes, cleanup is automatic—nodes not used by any active request become leaf nodes over time. The LRU eviction process runs periodically, examining leaf nodes and freeing the least recently accessed ones when memory pressure requires it. No manual tracking needed!

### 16.17.6 Implementation Notes

**Arena Allocation Benefits**: Our pointer-free design offers several advantages for education and production:

1. **Cache-Friendly**: All nodes stored contiguously in `std::vector` — better CPU cache utilization
2. **No Fragmentation**: No malloc/free per node — arena manages memory in bulk
3. **Slot Reuse**: Inactive slots marked with `node_active_[i] = false`, reused by later allocations
4. **Safe Bounds Checking**: Every access validates node ID is in range and active
5. **Educational Clarity**: Students see modern C++ without pointer gymnastics

**Python Binding Strategy**: C++ owns the arena and exposes node IDs (integers) to Python. When Python needs node properties, it calls `cache.get_node(node_id)` which returns a dictionary:

```python
node = cache.get_node(node_id)  # Returns dict
print(node["token"])            # Access properties
print(node["is_leaf"])          
print(node["children"])         # Dict of {token: child_id}
```

This keeps memory management in C++ while giving Python clean read access.

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

**Continuous Batching Main Loop**. The serving engine's `step()` method orchestrates one complete iteration of the continuous batching cycle. First, it schedules prefill chunks for new or partially-processed requests, executing those chunks and transitioning completed prefills to decode phase. Next comes decode scheduling for all running requests—generating one token per request, sampling from the logits, and removing finished requests. Finally, the system admits waiting requests to fill available batch capacity. This cycle repeats continuously with no global synchronization—each iteration rebuilds the batch from current state, allowing finished requests to leave and new ones to enter seamlessly.

The core pattern:
```python
class ContinuousBatcher:
    def step(self):
        tokens_generated = 0
        
        # Phase 1: Prefill chunks
        if prefill_batch := self.prefill_mgr.get_next_batch():
            self.executor.prefill(prefill_batch)
            for req in prefill_batch.finished:
                self.request_pool.move_to_running(req)
        
        # Phase 2: Decode running
        if decode_batch := self.decode_mgr.get_batch(self.request_pool.running):
            logits = self.executor.decode(decode_batch)
            for i, req in enumerate(decode_batch.requests):
                token = sample_token(logits[i])
                req.append_token(token)
                if req.is_finished():
                    self.request_pool.mark_finished(req)
                    tokens_generated += 1
        
        # Phase 3: Admit waiting
        while self.can_admit_more():
            if waiting := self.request_pool.get_next_waiting():
                self.radix_mgr.allocate_for_request(waiting)
                self.request_pool.move_to_running(waiting)
        
        return tokens_generated
```

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

Nano-serving demonstrates how each algorithmic technique contributes to serving efficiency. Parallel batching processes multiple requests simultaneously rather than sequentially, providing the foundation for all further optimizations. Paged KV cache eliminates memory fragmentation that would otherwise waste GPU capacity, enabling higher concurrency. Continuous batching dynamically adds and removes requests to maintain high utilization regardless of varying completion times. Radix cache reuses computation for shared prefixes when workloads exhibit overlap, though effectiveness depends on actual prefix patterns. The relative impact of each optimization varies with workload characteristics including batch sizes, prompt lengths, prefix overlap patterns, and generation lengths.

**Optimization Breakdown**: Each algorithmic component contributes multiplicatively to overall throughput. Consider 32 requests with 100-token prompts and 20-token generation:

- **Baseline (sequential)**: 32 × 120 = 3,840 computation steps, processing requests one by one
- **+ Parallel batching**: Reduces to ~480 steps through 8-way parallelism, providing the foundation
- **+ KV cache**: Further reduces to ~672 steps (32 prefills + 32×20 decode iterations), eliminating quadratic recomputation
- **+ Continuous batching**: Halves effective steps to ~336 through doubled utilization by filling idle cycles
- **+ Radix cache**: With 60% prefix sharing, drops to ~134 steps, as most prefills reuse cached computation

The cumulative effect yields roughly 30× speedup from combining these techniques. Each optimization addresses a different inefficiency, making them complementary rather than redundant.

**Token Sampling**. The system samples next tokens from model logits using temperature-controlled probability distributions. Greedy sampling (temperature=0) simply selects the highest-probability token via argmax. Stochastic sampling applies temperature scaling to the logits before softmax normalization, controlling randomness: lower temperatures concentrate probability on top tokens for conservative generation, while higher temperatures flatten the distribution for creative variation. The temperature parameter T scales logits as $\text{probs}_i = \frac{\exp(\text{logits}_i / T)}{\sum_j \exp(\text{logits}_j / T)}$, with numerical stability achieved by subtracting the maximum logit before exponentiation.

The core implementation:
```python
def sample_token(logits, temperature=1.0):
    if temperature == 0.0:
        return int(np.argmax(logits))  # Greedy
    
    logits = logits / temperature
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    probs = exp_logits / np.sum(exp_logits)
    return int(np.random.choice(len(probs), p=probs))
```

**Memory Efficiency**: Paged allocation eliminates fragmentation. For 32 concurrent requests with 2048-token maximum but 120-token average usage, contiguous allocation wastes memory on unused capacity, while paged allocation tracks actual usage, enabling 17× higher concurrency.

**Optimization Analysis**: Each technique addresses a different bottleneck. Parallel batching provides the foundation by processing multiple requests simultaneously. Paged KV cache eliminates memory waste. Continuous batching maintains high utilization by dynamically managing the batch. Radix cache reduces redundant computation on workloads with prefix overlap. The effectiveness of each depends on workload characteristics—prompt lengths, batch sizes, and prefix sharing patterns all influence relative impact.

## 16.21 Conclusion

This chapter demonstrated the core serving algorithms through nano-serving, an educational implementation combining Python orchestration with MLIR-compiled execution. The implementation validates that paged attention, continuous batching, radix caching, and chunked prefill work as described in the framework architecture sections. Understanding these algorithms at the implementation level prepares you to read production systems like vLLM and SGLang, which use the same fundamental techniques with additional GPU optimizations and distributed coordination.

Chapters 1-15 built the compiler foundation: MLIR's multi-level IR, dynamic shapes, bufferization, custom dialects, and GPU concepts. Chapter 16 completed the picture by showing how compiled models integrate into production serving systems. The progression from MLIR internals through transformer implementation to serving infrastructure demonstrates the full ML systems stack—compiler techniques enable efficient model execution, while serving algorithms enable efficient multi-request orchestration.

Modern ML systems achieve remarkable performance through layered optimization: compiler transformations optimize individual operations, while serving algorithms optimize resource utilization across concurrent requests. MLIR provides systematic infrastructure for the former; continuous batching, paged memory, and prefix caching provide the algorithmic foundation for the latter. Together, these techniques enable the interactive AI applications users experience today.

This educational implementation prioritizes clarity over maximum performance, making the algorithms transparent and comprehensible. Production systems like vLLM and SGLang apply the same fundamental techniques but with CUDA-optimized kernels, multi-GPU coordination, and hardware-specific tuning. Understanding the algorithmic core—which you now have from this chapter—provides the foundation for exploring those production implementations with confidence.

This educational implementation prioritizes clarity over maximum performance, making the algorithms transparent and comprehensible. Production systems like vLLM and SGLang apply the same fundamental techniques but with CUDA-optimized kernels, multi-GPU coordination, and hardware-specific tuning. Understanding the algorithmic core—which you now have from this chapter—provides the foundation for exploring those production implementations with confidence.