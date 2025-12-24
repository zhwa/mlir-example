# Chapter 16 Part 1: Production LLM Serving - Framework Architecture

Chapters 1-15 built a complete GPT model with GPU programming concepts. The implementation demonstrates transformer architecture, KV caching, and parallel execution patterns. However, building a production LLM serving system requires more than a correct model—it demands efficient request scheduling, memory management, and multi-request batching. This chapter introduces production serving techniques used in real-world systems like vLLM, SGLang, and TensorRT-LLM.

Chapter 16 is divided into two parts. Part 1 (this chapter) teaches the **inference framework architecture**—the scheduling algorithms, memory management strategies, and system design patterns that enable 100-1000× speedups over naive implementations. Part 2 (next chapter) demonstrates **our specific implementation** combining MLIR compiler acceleration with Python inference orchestration. This two-part structure separates concepts (portable across systems) from implementation (specific to our MLIR-based approach).

**Why Production Serving is Different**. A research implementation processes one request at a time: encode prompt, generate tokens sequentially, return result. Production systems serve **thousands of concurrent users** with diverse requirements—some need low latency (chat), others need high throughput (batch document summarization). The naive approach fails catastrophically:

```python
# Naive serving (100-1000× slower than production!)
def serve_naive(requests):
    for req in requests:
        prompt_tokens = tokenize(req.prompt)
        
        # Full forward pass for entire sequence every iteration
        for _ in range(req.max_tokens):
            logits = model.forward(prompt_tokens)
            next_token = sample(logits[-1])
            prompt_tokens.append(next_token)
            
        return prompt_tokens
```

**Three Fatal Problems**:

1. **O(N²) Redundancy**: Each generation step recomputes attention for all previous tokens—at token 100, we've processed token 0 one hundred times
2. **Sequential Processing**: Requests wait in queue while current request generates token-by-token
3. **Memory Waste**: Each request allocates max_seq_len memory (2048 tokens), but average prompt is 200 tokens—90% waste

Production systems solve these problems with **KV caching** (O(N²) → O(N)), **continuous batching** (parallel multi-request execution), and **paged memory management** (10-30× memory capacity). The combination achieves 100-1000× speedup.

**Real-World Impact**. Consider a production serving system handling 1,000 requests/second (typical for a mid-size company):

```
Naive implementation:
  - 1 request processes at a time
  - 100 tokens/sec generation
  - Queue depth: 1,000 concurrent users
  - Average wait time: 10 seconds per request
  - User experience: unacceptable

Production implementation (vLLM/SGLang):
  - 32 requests batch together
  - 10,000 tokens/sec generation (100× faster)
  - Queue depth: minimal
  - Average wait time: <100ms
  - User experience: interactive
```

This chapter explains how production systems achieve this transformation.

## 16.1 The Request Lifecycle

Production serving systems organize computation around **requests**—user tasks with specific prompts and generation parameters. Understanding request lifecycle and state management is foundational to serving architecture.

**Request Abstraction**. A request represents a single user generation task:

```python
class Request:
    """Represents one user generation task"""
    
    # Input
    prompt: List[int]          # Token IDs [1, 2, 3, ...]
    max_tokens: int            # Generate up to N tokens
    temperature: float         # Sampling temperature
    top_k: int                 # Top-k sampling
    
    # State tracking
    cached_len: int            # Tokens with computed KV cache
    device_len: int            # Tokens in GPU memory
    max_device_len: int        # Maximum capacity (prompt_len + max_tokens)
    
    # Memory management
    table_idx: int             # Row in page table
    page_ids: List[int]        # Allocated physical pages
    
    # Lifecycle
    status: str                # "waiting" → "running" → "finished"
    output_tokens: List[int]   # Generated tokens
```

**Key Invariants**:
- `0 ≤ cached_len < device_len ≤ max_device_len`
- `cached_len`: tokens with KV cache computed and stored
- `device_len`: tokens loaded in GPU memory (prompt + generated)
- `extend_len = device_len - cached_len`: tokens needing forward pass

**Example Request Lifecycle**:

```python
# Initial state: User submits "What is MLIR?" → tokens [1, 2, 3, 4, 5]
req = Request(
    prompt=[1, 2, 3, 4, 5],
    max_tokens=10,
    cached_len=0,     # No KV cache yet
    device_len=5,     # 5 prompt tokens
    max_device_len=15 # 5 prompt + 10 generation
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

**Request States**. Requests transition through three states:

```
┌──────────┐  scheduler   ┌─────────┐  completion  ┌──────────┐
│ Waiting  │─────────────>│ Running │─────────────>│ Finished │
└──────────┘  admission   └─────────┘  (EOS/limit) └──────────┘
                              ▲ │
                              └─┘ decode iterations
```

- **Waiting**: Request received, not yet executing (queued)
- **Running**: Actively generating tokens (forward passes)
- **Finished**: Completed (reached max_tokens or EOS)

**Batching Abstraction**. Multiple requests execute together in a **batch**:

```python
class Batch:
    """Group of requests processed together"""
    
    reqs: List[Request]        # Requests in this batch
    phase: str                 # "prefill" or "decode"
    
    # Prepared for GPU execution
    input_ids: Tensor          # [batch_size] or [total_tokens]
    positions: Tensor          # Position indices for each token
    out_loc: Tensor            # KV cache write locations
    
    # Attention metadata
    attn_metadata: AttentionMetadata  # Backend-specific attention info
```

**Batch Phases**. Requests have two execution phases with different characteristics:

```
Prefill Phase (First Forward Pass):
  Input: All prompt tokens
  Operation: Compute KV cache for entire prompt
  Parallelism: Across tokens (seq_len dimension)
  Bottleneck: Computation (matmul)
  
Decode Phase (Subsequent Passes):
  Input: One new token
  Operation: Generate next token using cached KV
  Parallelism: Across requests (batch dimension)
  Bottleneck: Memory bandwidth (KV cache loads)
```

Production systems optimize each phase differently:
- **Prefill**: Large batch_size hurts (memory), process few requests with many tokens
- **Decode**: Large batch_size helps (parallelism), process many requests with one token each

**Memory Requirements**. Request memory consumption varies dramatically:

```python
# Per-request KV cache memory (float16):
mem_per_token = (
    2                    # key + value
    * num_layers         # 12 layers (GPT-2)
    * num_kv_heads       # 12 heads
    * head_dim           # 64 dimensions
    * 2                  # float16 = 2 bytes
)

# Example: GPT-2 (12 layers, 12 heads, 64 head_dim)
mem_per_token = 2 * 12 * 12 * 64 * 2 = 36,864 bytes ≈ 36 KB/token

# 1000-token sequence: 36 MB
# 32 concurrent 1000-token requests: 1.15 GB
# 256 concurrent requests: 9.2 GB (exceeds typical GPU memory!)
```

This memory pressure motivates **paged memory management** (Section 16.2) and **prefix sharing** (Section 16.4).

## 16.2 Paged KV Cache: Virtual Memory for Attention

The naive approach allocates max_seq_len memory for each request—wasteful when actual prompt lengths vary widely (50-2000 tokens). Production systems use **paged memory management**: divide KV cache into fixed-size pages, allocate on-demand.

**The Fragmentation Problem**. Contiguous allocation wastes memory:

```
Contiguous allocation (naive):
Request 1 (200 tokens, allocated 2048): ████░░░░░░░░░░░░░░░░ (90% waste)
Request 2 (50 tokens, allocated 2048):  █░░░░░░░░░░░░░░░░░░░ (97% waste)
Request 3 (1500 tokens, allocated 2048): ███████████████░░░░ (27% waste)

Total allocated: 6,144 tokens
Total used: 1,750 tokens
Waste: 71.5%
```

**Paged Solution**. Divide KV cache into fixed-size pages (16-32 tokens per page):

```
Physical memory (pages):
[0] [1] [2] [3] [4] [5] [6] [7] [8] [9] ... [1023]

Request 1 (200 tokens = 13 pages):
  Page table: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  Waste: 8 tokens (4% vs 90%)

Request 2 (50 tokens = 4 pages):
  Page table: [13, 14, 15, 16]
  Waste: 14 tokens (21% vs 97%)

Request 3 (1500 tokens = 94 pages):
  Page table: [17, 18, ..., 110]
  Waste: 4 tokens (0.3% vs 27%)

Total allocated: 1,777 tokens (111 pages × 16 tokens/page)
Total used: 1,750 tokens
Waste: 1.5% (vs 71.5%!)
```

**Page Table Translation**. Each request has a page table mapping logical token positions to physical pages:

```python
# Page table: logical_page → physical_page
page_size = 16
page_table = [
    [0, 1, 2, 3],      # Request 0: pages 0-3
    [4, 5, 6, 7, 8],   # Request 1: pages 4-8
    ...
]

# Access KV cache for request_id=0, token_id=42
logical_page = 42 // page_size  # = 2
offset = 42 % page_size         # = 10
physical_page = page_table[request_id][logical_page]  # = 2
physical_address = physical_page * page_size + offset  # = 42

# Load KV from cache
key = k_cache[layer, physical_address, :]
value = v_cache[layer, physical_address, :]
```

**Implementation Pattern**:

```python
class KVCachePool:
    """Paged memory allocator for KV cache"""
    
    def __init__(self, num_pages, num_layers, num_heads, head_dim):
        # Physical storage: [num_layers, total_pages, num_heads, head_dim]
        self.k_cache = torch.zeros(num_layers, num_pages, num_heads, head_dim)
        self.v_cache = torch.zeros(num_layers, num_pages, num_heads, head_dim)
        
        # Free page tracking
        self.free_pages = set(range(num_pages))
    
    def allocate(self, num_pages):
        """Allocate contiguous physical pages"""
        if len(self.free_pages) < num_pages:
            return None  # Out of memory
        
        pages = sorted(self.free_pages)[:num_pages]
        for page in pages:
            self.free_pages.remove(page)
        
        return pages
    
    def free(self, pages):
        """Return pages to free pool"""
        self.free_pages.update(pages)
    
    def store_kv(self, keys, values, positions, layer_id):
        """Store KV for given positions"""
        # positions: [batch_size] logical addresses
        # Translate through page table to physical addresses
        for i, pos in enumerate(positions):
            self.k_cache[layer_id, pos] = keys[i]
            self.v_cache[layer_id, pos] = values[i]
```

**Memory Efficiency Gains**:

| Approach | Memory for 256 requests (avg 300 tokens, max 2048) | Capacity |
|----------|-----------------------------------------------------|----------|
| Contiguous | 256 × 2048 × 36 KB = 18.9 GB | 256 requests |
| Paged (page_size=16) | 256 × 300 × 36 KB = 2.77 GB (98% reduction!) | 1,752 requests (6.8×) |

Paged memory enables **6-10× more concurrent requests** on the same hardware.

**Attention Kernel Modification**. Standard attention assumes contiguous KV:

```python
# Standard attention (contiguous)
def attention(q, k, v):
    scores = q @ k.transpose(-2, -1)
    weights = softmax(scores)
    output = weights @ v
    return output
```

**Paged attention** translates logical addresses:

```python
# Paged attention (non-contiguous)
def paged_attention(q, page_table, k_cache, v_cache, seq_lens):
    batch_size = len(page_table)
    outputs = []
    
    for i in range(batch_size):
        # Gather keys and values through page table
        seq_len = seq_lens[i]
        pages = page_table[i][:ceil(seq_len / PAGE_SIZE)]
        
        # Gather scattered KV
        k = gather_pages(k_cache, pages)[:seq_len]
        v = gather_pages(v_cache, pages)[:seq_len]
        
        # Standard attention on gathered KV
        scores = q[i] @ k.transpose(-1, -2)
        weights = softmax(scores)
        out = weights @ v
        outputs.append(out)
    
    return torch.stack(outputs)
```

Production kernels (FlashAttention, Flash-Decoding) fuse page table translation directly into attention computation for efficiency.

## 16.3 Continuous Batching: Dynamic Request Scheduling

Naive batch serving processes all requests together—waits for the slowest request before accepting new requests. **Continuous batching** dynamically adds/removes requests every iteration, maximizing GPU utilization.

**Static Batching Problem**. Traditional batching waits for all requests to complete:

```python
# Static batching (inefficient)
def serve_static_batch(requests):
    # Process entire batch together
    batch = Batch(requests)
    
    while any(r.status != "finished" for r in batch.reqs):
        # All requests generate one token
        logits = model.forward(batch)
        next_tokens = sample(logits)
        
        for req, token in zip(batch.reqs, next_tokens):
            req.output_tokens.append(token)
            if req.is_finished():
                req.status = "finished"
    
    # Return all results at once
    return [r.output_tokens for r in batch.reqs]
```

**Problem**: If one request generates 1000 tokens and others finish at 100 tokens, GPU sits 90% idle for 900 iterations while that one request completes.

```
Time →
Req 1: ████████████████████████████████████████ (1000 tokens)
Req 2: ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (100 tokens, then waits!)
Req 3: ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ (120 tokens, then waits!)
Req 4: ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (100 tokens, then waits!)

GPU Utilization: 4/4 (100%) → 1/4 (25%) for 880 iterations
```

**Continuous Batching Solution**. Add/remove requests dynamically:

```python
# Continuous batching (efficient)
def serve_continuous_batch(request_queue):
    running = []  # Currently executing requests
    
    while request_queue or running:
        # 1. Add new requests up to capacity
        while len(running) < MAX_BATCH_SIZE and request_queue:
            new_req = request_queue.pop(0)
            running.append(new_req)
        
        if not running:
            continue  # Nothing to process
        
        # 2. Execute one iteration
        batch = Batch(running)
        logits = model.forward(batch)
        next_tokens = sample(logits)
        
        # 3. Update requests and remove finished ones
        still_running = []
        for req, token in zip(running, next_tokens):
            req.output_tokens.append(token)
            if not req.is_finished():
                still_running.append(req)
            else:
                # Request finished, can immediately return result
                yield req
        
        running = still_running
```

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

**Request Pool Management**. Production systems organize requests in pools:

```python
class RequestPool:
    """Manages request lifecycle"""
    
    def __init__(self):
        self.waiting = []    # Queued requests
        self.running = []    # Executing requests
        self.finished = []   # Completed requests
    
    def add_request(self, req):
        """User submits new request"""
        self.waiting.append(req)
    
    def admit_requests(self, capacity):
        """Move waiting → running (up to capacity)"""
        admitted = []
        while len(self.running) < capacity and self.waiting:
            req = self.waiting.pop(0)
            self.running.append(req)
            admitted.append(req)
        return admitted
    
    def finish_requests(self):
        """Move running → finished (completed requests)"""
        still_running = []
        for req in self.running:
            if req.is_finished():
                self.finished.append(req)
            else:
                still_running.append(req)
        self.running = still_running
    
    def get_batch(self):
        """Create batch from running requests"""
        return Batch(reqs=self.running.copy())
```

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

**Radix Tree Solution**. A **radix tree** (prefix tree / trie) stores KV cache nodes indexed by token sequences:

```
Root
 ├─ [1, 2, 3] (cached)
 │   ├─ [4, 5] (cached)
 │   │   ├─ [6, 7] (cached) ← Request 1
 │   │   └─ [8, 9] (cached) ← Request 2
 │   └─ [10, 11] (cached) ← Request 3
 └─ [50, 51] (cached) ← Different prefix
```

**Radix Node Structure**:

```python
class RadixNode:
    """Node in radix tree storing KV cache"""
    
    def __init__(self):
        self.tokens: List[int] = []       # Token sequence this node represents
        self.page_ids: List[int] = []     # Physical pages storing KV cache
        self.children: Dict[int, RadixNode] = {}  # token → child node
        
        self.last_access_time: float = 0  # For LRU eviction
        self.ref_count: int = 0           # Number of requests using this node
```

**Tree Operations**:

```python
class RadixCache:
    """Prefix-sharing KV cache using radix tree"""
    
    def __init__(self, kv_pool):
        self.root = RadixNode()
        self.kv_pool = kv_pool
    
    def match_prefix(self, tokens):
        """Find longest matching prefix in tree"""
        node = self.root
        matched_len = 0
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Try to follow edge with this token
            if token in node.children:
                child = node.children[token]
                
                # Match tokens in child node
                for j, child_token in enumerate(child.tokens):
                    if i + j >= len(tokens) or tokens[i + j] != child_token:
                        # Partial match within node
                        return node, matched_len
                
                # Full match of child node
                matched_len += len(child.tokens)
                i += len(child.tokens)
                node = child
            else:
                # No matching edge
                break
        
        return node, matched_len
    
    def insert(self, tokens, page_ids):
        """Insert new token sequence with KV cache pages"""
        node, matched_len = self.match_prefix(tokens)
        
        if matched_len == len(tokens):
            # Full match, reuse existing node
            node.ref_count += 1
            node.last_access_time = time.time()
            return node, 0  # 0 new tokens cached
        
        # Create new node for unmatched suffix
        new_tokens = tokens[matched_len:]
        new_node = RadixNode(tokens=new_tokens, page_ids=page_ids)
        
        # Link new node
        first_token = new_tokens[0]
        node.children[first_token] = new_node
        
        new_node.ref_count = 1
        new_node.last_access_time = time.time()
        
        return new_node, len(new_tokens)
```

**Request Processing with Radix Cache**:

```python
def process_request_with_cache(req, radix_cache):
    """Process request using radix cache for prefix reuse"""
    
    # 1. Find cached prefix
    node, cached_len = radix_cache.match_prefix(req.tokens)
    
    # 2. Set request state
    req.cached_len = cached_len
    req.device_len = len(req.tokens)
    req.cache_node = node
    
    # 3. Allocate pages for new tokens only
    new_token_count = req.device_len - req.cached_len
    if new_token_count > 0:
        new_pages = kv_pool.allocate(pages_needed(new_token_count))
        
        # 4. Insert new tokens into radix tree
        radix_cache.insert(req.tokens, new_pages)
    
    # 5. Build page table from tree path
    req.page_table = collect_pages_from_tree_path(req.cache_node)
    
    return req.extend_len  # Tokens needing forward pass
```

**Cache Hit Rate**. Real-world workload (chatbots with system prompts):

```
1000 requests:
  - System prompt: 500 tokens
  - User queries: 20-100 tokens (avg 50)
  - Total tokens without sharing: 550,000 tokens
  - Total tokens with sharing: 500 + (1000 × 50) = 50,500 tokens
  - Computation reduction: 91% (10.9× speedup!)
  - Cache hit rate: 91%
```

Even with diverse queries, practical workloads show **40-60% hit rates**—translating to **2-3× speedup**.

**LRU Eviction**. Limited memory requires evicting unused cache nodes:

```python
def evict_lru_nodes(radix_cache, required_pages):
    """Evict least-recently-used nodes to free memory"""
    
    # 1. Collect evictable nodes (ref_count == 0)
    evictable = []
    
    def collect_evictable(node):
        if node.ref_count == 0 and node.page_ids:
            evictable.append(node)
        for child in node.children.values():
            collect_evictable(child)
    
    collect_evictable(radix_cache.root)
    
    # 2. Sort by last_access_time (LRU first)
    evictable.sort(key=lambda n: n.last_access_time)
    
    # 3. Evict until enough memory freed
    freed_pages = 0
    for node in evictable:
        if freed_pages >= required_pages:
            break
        
        # Free node's pages
        radix_cache.kv_pool.free(node.page_ids)
        freed_pages += len(node.page_ids)
        
        # Remove node from tree
        # (implementation detail: unlink from parent)
        node.page_ids = []
    
    return freed_pages
```

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

**Chunked Prefill Strategy**. Split long prompts into fixed-size chunks (e.g., 256 tokens):

```python
class ChunkedRequest:
    """Request split into chunks for fair scheduling"""
    
    def __init__(self, tokens, chunk_size=256):
        self.tokens = tokens
        self.chunk_size = chunk_size
        
        # Divide into chunks
        self.chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            self.chunks.append(chunk_tokens)
        
        self.current_chunk = 0
    
    def next_chunk(self):
        """Get next chunk for processing"""
        if self.current_chunk < len(self.chunks):
            chunk = self.chunks[self.current_chunk]
            self.current_chunk += 1
            return chunk
        return None
    
    def is_prefill_done(self):
        return self.current_chunk >= len(self.chunks)
```

**Chunked Scheduling**:

```python
def chunked_prefill_schedule(prefill_queue, decode_batch, token_budget=512):
    """Schedule chunks from prefill queue, interleaved with decode"""
    
    batch_tokens = []
    batch_reqs = []
    current_budget = token_budget
    
    # 1. Round-robin through prefill requests
    while prefill_queue and current_budget > 0:
        req = prefill_queue[0]  # Peek first
        chunk = req.next_chunk()
        
        if chunk is None:
            # Request fully prefilled, remove from queue
            prefill_queue.pop(0)
            continue
        
        # Add chunk to batch if fits budget
        if len(chunk) <= current_budget:
            batch_tokens.extend(chunk)
            batch_reqs.append(req)
            current_budget -= len(chunk)
            
            # Move request to back of queue (round-robin)
            prefill_queue.append(prefill_queue.pop(0))
        else:
            break  # Chunk too large for remaining budget
    
    # 2. Form prefill batch
    if batch_tokens:
        prefill_batch = Batch(reqs=batch_reqs, phase="prefill")
        yield prefill_batch
    
    # 3. Yield decode batch
    if decode_batch.reqs:
        yield decode_batch
```

**Scheduling Timeline with Chunking**:

```
Time →
Req 1 (2000 tok): [256] .............. [256] ... [256] ... [256] ... [256] ... [256] ... [256] ... [256]
Req 2 (50 tok):         [50]
Req 3 (100 tok):              [100]
Decode batch:          [32×1]  [32×1]  [32×1]  [32×1]  [32×1]  [32×1]  [32×1]  [32×1]

Req 2 wait time: 256 tokens = 15ms (vs 200ms without chunking!)
Req 3 wait time: 256 + 50 = 20ms (vs 205ms!)
```

**Fairness Metrics**:

| Metric | Without Chunking | With Chunking (chunk_size=256) |
|--------|------------------|--------------------------------|
| Avg wait (short requests) | 150ms | 18ms (8.3× better) |
| P99 wait (short requests) | 300ms | 45ms (6.7× better) |
| Throughput | 10,000 tok/s | 9,500 tok/s (5% cost) |

Chunked prefill trades **5% throughput** for **6-8× latency improvement** on short requests—essential for interactive serving.

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

Prefill and decode have fundamentally different characteristics. Production systems separate them for specialized optimization.

**Performance Characteristics**:

```
Prefill Phase:
  - Input: seq_len tokens (100-2000)
  - Compute: O(seq_len²) attention + O(seq_len) FFN
  - Memory: Write seq_len KV cache entries
  - Bottleneck: Computation (matmul)
  - Optimization: Minimize memory writes, maximize FLOPS
  - Batch size: Small (2-8 requests) × large seq_len

Decode Phase:
  - Input: 1 token per request
  - Compute: O(seq_len) attention (read cached K/V) + O(1) FFN
  - Memory: Read seq_len KV cache entries
  - Bottleneck: Memory bandwidth (KV cache loads)
  - Optimization: Maximize memory bandwidth, batch parallelism
  - Batch size: Large (32-256 requests) × 1 token each
```

**Separate Schedulers**:

```python
class PrefillManager:
    """Schedules prefill phase (prompt processing)"""
    
    def __init__(self, token_budget=512):
        self.queue = []
        self.token_budget = token_budget
    
    def add_request(self, req):
        """Add new request needing prefill"""
        self.queue.append(req)
    
    def get_batch(self):
        """Select requests for next prefill batch"""
        batch_reqs = []
        total_tokens = 0
        
        # FCFS with token budget
        while self.queue and total_tokens < self.token_budget:
            req = self.queue[0]
            req_tokens = req.extend_len
            
            if total_tokens + req_tokens <= self.token_budget:
                batch_reqs.append(self.queue.pop(0))
                total_tokens += req_tokens
            else:
                break  # Would exceed budget
        
        if batch_reqs:
            return Batch(reqs=batch_reqs, phase="prefill")
        return None


class DecodeManager:
    """Schedules decode phase (token generation)"""
    
    def __init__(self, max_batch_size=256):
        self.running = []
        self.max_batch_size = max_batch_size
    
    def add_request(self, req):
        """Add request that finished prefill"""
        self.running.append(req)
    
    def get_batch(self):
        """Batch all running requests (up to max_batch_size)"""
        if not self.running:
            return None
        
        # Batch all running requests
        batch_reqs = self.running[:self.max_batch_size]
        return Batch(reqs=batch_reqs, phase="decode")
    
    def remove_finished(self):
        """Remove completed requests"""
        self.running = [r for r in self.running if not r.is_finished()]
```

**Scheduling Loop with Separation**:

```python
def two_phase_serving_loop(prefill_mgr, decode_mgr, model):
    """Serving loop with separate prefill/decode scheduling"""
    
    while True:
        # 1. Get prefill batch (if any requests waiting)
        prefill_batch = prefill_mgr.get_batch()
        
        # 2. Get decode batch (if any requests running)
        decode_batch = decode_mgr.get_batch()
        
        # 3. Execute prefill (if exists)
        if prefill_batch:
            _ = model.forward(prefill_batch)
            
            # Move prefilled requests to decode
            for req in prefill_batch.reqs:
                if req.extend_len == 0:  # Prefill complete
                    decode_mgr.add_request(req)
        
        # 4. Execute decode (if exists)
        if decode_batch:
            logits = model.forward(decode_batch)
            next_tokens = sample(logits)
            
            for req, token in zip(decode_batch.reqs, next_tokens):
                req.append_token(token)
            
            decode_mgr.remove_finished()
        
        # 5. If nothing to do, wait briefly
        if not prefill_batch and not decode_batch:
            time.sleep(0.001)
```

**Optimization Opportunities**:

```python
# Prefill: Use FlashAttention for compute efficiency
class PrefillAttention:
    def forward(self, q, k, v):
        # FlashAttention optimizes for compute
        return flash_attention(q, k, v)

# Decode: Use Flash-Decoding for bandwidth efficiency
class DecodeAttention:
    def forward(self, q, k_cache, v_cache, seq_lens):
        # Flash-Decoding optimizes for memory bandwidth
        return flash_decoding(q, k_cache, v_cache, seq_lens)
```

Different attention kernels for different phases—FlashAttention (prefill) minimizes memory traffic, Flash-Decoding (decode) parallelizes across sequence dimension.

## 16.7 Production System Architecture

Modern LLM serving systems combine all techniques into a cohesive architecture. This section describes the overall system structure inspired by SGLang, vLLM, and TensorRT-LLM.

**Component Overview**:

```
┌─────────────────────────────────────────────────────┐
│ API Server (FastAPI/OpenAI-compatible)              │
│ • HTTP endpoints (/v1/chat/completions)             │
│ • Request validation and queueing                   │
└────────────────┬────────────────────────────────────┘
                 │ (ZMQ/gRPC)
┌────────────────▼────────────────────────────────────┐
│ Tokenizer/Detokenizer Workers                       │
│ • Text → token IDs (parallel tokenization)          │
│ • Token IDs → text (streaming detokenization)       │
└────────────────┬────────────────────────────────────┘
                 │ (ZMQ)
┌────────────────▼────────────────────────────────────┐
│ Scheduler Worker (Main orchestration)               │
│ • Request pool management                           │
│ • Radix cache management                            │
│ • Prefill/Decode scheduling                         │
│ • KV cache allocation                               │
└────────────────┬────────────────────────────────────┘
                 │ (Python/C++ interface)
┌────────────────▼────────────────────────────────────┐
│ Engine (Model execution)                            │
│ • GPU model inference                               │
│ • KV cache pool (paged memory)                      │
│ • Attention backends (FlashAttention/FlashInfer)    │
│ • CUDA graph optimization                           │
└─────────────────────────────────────────────────────┘
```

**Data Flow (Single Request)**:

```
1. User sends prompt to API Server
   ↓
2. API Server → Tokenizer: Convert text to token IDs
   ↓
3. Tokenizer → Scheduler: [1, 2, 3, 4, 5] (prompt tokens)
   ↓
4. Scheduler:
   - Check radix cache for prefix match
   - Allocate KV cache pages
   - Add to prefill queue
   ↓
5. Scheduler creates prefill batch
   ↓
6. Engine executes prefill forward pass
   - Compute attention for all prompt tokens
   - Write KV cache
   - Return logits for last position
   ↓
7. Scheduler samples first token
   - Add request to decode manager
   ↓
8. Engine executes decode iterations (loop):
   - Forward pass for 1 token
   - Read KV cache (via page table)
   - Write new KV cache entry
   - Return logits
   ↓
9. Scheduler samples next token
   - Check if EOS or max_tokens reached
   - If not done, repeat decode
   ↓
10. Request finishes → Scheduler → Detokenizer: [42, 73, 99] (output tokens)
    ↓
11. Detokenizer → API Server: "MLIR is..." (text)
    ↓
12. API Server → User: Stream response
```

**Scheduler State Machine**:

```python
class ContinuousBatcher:
    """Main scheduling loop coordinating all components"""
    
    def __init__(self, model, kv_pool, radix_cache):
        self.request_pool = RequestPool()
        self.prefill_mgr = PrefillManager()
        self.decode_mgr = DecodeManager()
        self.kv_pool = kv_pool
        self.radix_cache = radix_cache
        self.model = model
    
    def add_request(self, tokens, sampling_params):
        """User submits new request"""
        # 1. Find cached prefix
        cache_node, cached_len = self.radix_cache.match_prefix(tokens)
        
        # 2. Allocate pages for new tokens
        extend_len = len(tokens) - cached_len
        pages = self.kv_pool.allocate(pages_needed(extend_len))
        
        if pages is None:
            # Out of memory, try evicting LRU nodes
            self.radix_cache.evict_lru(pages_needed(extend_len))
            pages = self.kv_pool.allocate(pages_needed(extend_len))
        
        # 3. Create request
        req = Request(
            tokens=tokens,
            cached_len=cached_len,
            device_len=len(tokens),
            pages=pages,
            sampling_params=sampling_params
        )
        
        # 4. Add to prefill queue
        self.request_pool.add_waiting(req)
        self.prefill_mgr.add_request(req)
    
    def step(self):
        """Execute one scheduling iteration"""
        # 1. Prefill batch
        prefill_batch = self.prefill_mgr.get_batch()
        if prefill_batch:
            self.model.forward(prefill_batch)
            
            # Move to decode
            for req in prefill_batch.reqs:
                if req.prefill_done():
                    self.decode_mgr.add_request(req)
        
        # 2. Decode batch
        decode_batch = self.decode_mgr.get_batch()
        if decode_batch:
            logits = self.model.forward(decode_batch)
            next_tokens = self.sample(logits, decode_batch)
            
            for req, token in zip(decode_batch.reqs, next_tokens):
                req.append_token(token)
            
            self.decode_mgr.remove_finished()
        
        # 3. Collect finished requests
        return self.request_pool.collect_finished()
    
    def run_loop(self):
        """Main continuous batching loop"""
        while True:
            finished = self.step()
            
            # Return finished requests
            for req in finished:
                yield req
```

**Multi-GPU with Tensor Parallelism**:

```python
# Initialize TP group
torch.distributed.init_process_group(backend="nccl", world_size=4, rank=rank)

# Shard model weights across GPUs
model = create_model_with_tp(config, tp_size=4, tp_rank=rank)

# All-reduce after attention output
class ParallelAttention:
    def forward(self, x):
        # Each rank computes subset of heads
        local_output = self.attention(x)  # [batch, seq_len, hidden_size/tp_size]
        
        # All-reduce to combine results
        output = torch.distributed.all_reduce(local_output)
        return output
```

Each GPU processes subset of attention heads/FFN columns—4 GPUs provide 4× parallelism for large models.

## 16.8 Performance Analysis

Combining all optimizations yields dramatic speedups. This section quantifies each technique's contribution.

**Baseline vs Production**:

```python
# Baseline (naive serving):
def baseline_serve(requests):
    results = []
    for req in requests:
        tokens = req.prompt
        for _ in range(req.max_tokens):
            # O(N²) attention (no KV cache)
            logits = model.forward_no_cache(tokens)
            next_token = sample(logits[-1])
            tokens.append(next_token)
        results.append(tokens)
    return results

# Production (all optimizations):
def production_serve(requests):
    engine = NanoServingEngine(
        model=model,
        kv_cache_pages=1024,
        max_batch_size=256,
        radix_cache=True,
        chunked_prefill=True
    )
    return engine.generate(requests)
```

**Speedup Breakdown** (100 requests, avg 300 tokens prompt, 50 tokens generation):

| Optimization | Speedup | Cumulative |
|--------------|---------|------------|
| Baseline | 1.0× | 1.0× |
| + KV Caching | 10× | 10× |
| + Continuous Batching | 2.5× | 25× |
| + Paged Memory | 1.5× | 37.5× |
| + Radix Cache (60% hit) | 2.6× | 97.5× |
| + Chunked Prefill | 1.0× | 97.5× (no throughput cost) |
| **Total** | **~100×** | **~100×** |

**Individual Technique Impact**:

```
1. KV Caching (10×):
   - Eliminates O(N²) recomputation
   - Applies to all requests
   - No memory overhead with paging

2. Continuous Batching (2.5×):
   - GPU utilization: 33% → 82%
   - Dynamic batch size maintains high throughput
   - Immediate benefit as requests finish

3. Paged Memory (1.5×):
   - Memory efficiency: 29% → 97% (3.3× more capacity)
   - Capacity: 256 requests → 845 requests (3.3×)
   - Throughput scales with capacity

4. Radix Cache (2.6×):
   - Cache hit rate: 60% (realistic workload)
   - Computation reduction: 60% on prefill
   - Memory saving: 60% fewer pages

5. Chunked Prefill (1.0×):
   - Throughput: -5% (small cost)
   - Latency: -83% (6× better for short requests)
   - Fairness: essential for interactive serving
```

**Realistic Workload** (chatbot with system prompt):

```
Setup:
  - System prompt: 800 tokens (reused across requests)
  - User queries: 50-200 tokens (avg 100)
  - Generations: 50-300 tokens (avg 150)
  - Concurrency: 100 requests/sec

Without Radix Cache:
  Prefill compute: 900 tokens × 100 req/sec = 90,000 tok/sec
  Decode compute: 150 tokens × 100 req/sec = 15,000 tok/sec
  Total: 105,000 tok/sec

With Radix Cache (80% hit rate on system prompt):
  Prefill compute: (800 × 0.2 + 100) × 100 = 26,000 tok/sec
  Decode compute: 15,000 tok/sec
  Total: 41,000 tok/sec
  Reduction: 61% (2.6× speedup)
```

**Memory Capacity**:

```
GPU: 40 GB A100

Without Paging (contiguous allocation, max_seq_len=2048):
  Per request: 2048 tokens × 36 KB/token = 73.7 MB
  Capacity: 40 GB / 73.7 MB = 542 requests

With Paging (page_size=16, avg usage 350 tokens):
  Per request: 350 tokens × 36 KB/token = 12.6 MB
  Capacity: 40 GB / 12.6 MB = 3,174 requests
  Improvement: 5.9×
```

Paged memory unlocks **6-10× capacity**, enabling higher concurrency and throughput.

## 16.9 Summary

Chapter 16 Part 1 introduced production LLM serving framework architecture—the scheduling algorithms, memory management, and system design patterns that power real-world systems like vLLM, SGLang, and TensorRT-LLM.

**Core Techniques**:

1. **Request Lifecycle**: Abstraction managing user tasks through waiting → running → finished states with KV cache tracking
2. **Paged KV Cache**: Virtual memory for attention cache achieving 6-10× memory efficiency through fixed-size pages and page tables
3. **Continuous Batching**: Dynamic request scheduling that maximizes GPU utilization by adding/removing requests every iteration (2.5× throughput)
4. **Radix Cache**: Automatic prefix sharing via radix tree enabling 2-3× speedup on workloads with common prefixes (40-60% hit rate)
5. **Chunked Prefill**: Fair scheduling splitting long prompts into chunks, reducing short-request latency by 6-8× with minimal throughput cost
6. **Prefill-Decode Separation**: Specialized optimization for each phase leveraging their different compute vs bandwidth bottlenecks

**Performance Impact** (combined):
- **100-1000× speedup** vs naive sequential serving
- **6-10× memory capacity** through paging
- **2-3× additional speedup** via radix cache (prefix sharing)
- **6-8× latency reduction** for short requests (chunked prefill)

**Production Architecture**:
- Multi-component system: API server, tokenizer/detokenizer, scheduler, execution engine
- Request pool management coordinating waiting, running, and finished requests
- Radix cache with LRU eviction balancing hit rate and memory
- Adaptive scheduling responding to load conditions

**Looking Ahead**. Chapter 16 Part 2 demonstrates our nano-serving implementation combining these concepts with MLIR compiler acceleration. We'll see how:

- **Python framework** implements scheduling logic (request pool, radix cache, continuous batcher)
- **C++/MLIR compiler** implements model execution (KV cache pool, forward passes, JIT compilation)
- **Integration** coordinates Python scheduling with C++ execution through efficient interfaces
- **Testing** validates each component independently and end-to-end throughput

The concepts in Part 1 are portable—they apply to any serving system (PyTorch, TensorFlow, JAX). Part 2 demonstrates MLIR's unique advantages: JIT compilation for custom operations, efficient memory management, and seamless Python/C++ integration. Together, these chapters complete the book's journey from MLIR fundamentals (Chapter 1) to production serving (Chapter 16).

Modern LLM serving is algorithmic optimization (KV caching, radix cache), systems engineering (batching, scheduling, memory management), and careful implementation (kernel fusion, tensor parallelism). Understanding these layers—from high-level scheduling to low-level memory access—empowers you to build and optimize production ML systems. Chapter 16 Part 1 provided the conceptual foundation; Part 2 brings it to life with working code.
