# Chapter 16: Nano LLM Serving

## Quick Start

```bash
# Build the project
cmake --preset x64-release
cmake --build build/x64-release --target ch16

# Run all tests (68 comprehensive tests across 6 phases)
cd ch.16.Nano-Serving
python3 test_all.py

# Interactive demo (shows complete serving pipeline)
python3 demo.py

# Individual phase breakdown:
# - Phase 0: Request & Batch Abstraction (9 tests)
# - Phase 1: KV Cache Pool (8 tests)
# - Phase 2: Prefill vs Decode (11 tests)
# - Phase 3: Chunked Prefill (11 tests)
# - Phase 4: Radix Cache (13 tests)
# - Phase 5: Continuous Batching (7 tests)
# - Phase 6: Integration (9 tests)
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Python (Scheduling & Control)                      │
│  • Request/Batch management                         │
│  • Radix cache (prefix tree)                        │
│  • Prefill/Decode schedulers                        │
│  • Continuous batching loop                         │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│  C++/MLIR (Model Execution)                         │
│  • GPT model (MLIR JIT from Ch.14)                  │
│  • KV cache pool (paged memory)                     │
│  • Forward pass optimization                        │
└─────────────────────────────────────────────────────┘
```

## Usage Example

```python
from nano_engine import NanoServingEngine, SamplingParams
from executor import ModelConfig

# Initialize engine
config = ModelConfig(vocab_size=256, n_layer=2, n_head=4, n_embd=64)
weights = load_weights("model.pt")

engine = NanoServingEngine(
    config=config,
    weights=weights,
    kv_cache_pages=256,
    max_chunk_size=256
)

# Generate completions
prompts = [
    [1, 2, 3, 4, 5],       # "What is MLIR"
    [1, 2, 3, 6, 7],       # "What is Python" (shares prefix!)
    [10, 20, 30]           # "Hello world"
]

params = [
    SamplingParams(max_tokens=10, temperature=0.7)
    for _ in prompts
]

# Run serving!
finished = engine.generate(prompts, params)

# Check results
for req in finished:
    print(f"Output: {req.output_tokens}")

# View statistics
stats = engine.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")  # ~60%
print(f"Throughput: {stats['avg_tokens_per_step']:.1f} tok/step")
```

## Learning Path

1. **Read [TUTORIAL.md](TUTORIAL.md)** (comprehensive guide with explanations)
2. **Run tests** (understand each component)
3. **Modify parameters** (experiment with batch sizes, chunk sizes)
4. **Try ebook chapters** (book/Chapter-16-Production-Serving-Part1.md and Part2.md)
2. **Run tests** (`python3 test_all.py` - 68 tests across 6 phases)
3. **Try demo** (`python3 demo.py` - interactive serving)
4. **Experiment** (modify batch sizes, chunk sizes, cache parameters

### Paged KV Cache (Phase 1)
```
Like OS virtual memory:
- Divide KV cache into pages (16 tokens each)
- Allocate/free pages dynamically
- Share pages between requests (radix cache)
```

### Continuous Batching (Phase 5)
```
Dynamic batch that changes every step:
1. Remove finished requests → free memory
2. Add new requests → fill empty slots
3. Generate tokens for all running requests
4. Repeat!
```

### Radix Cache (Phase 4)
```
Pointer-free prefix tree for automatic sharing:
           [root:0]
             |
         "What is"
        /         \
   "MLIR"      "Python"
   
Two requests share "What is" → reuse KV cache!

Implementation: Arena-based with integer node IDs
- No pointers, no reference counting
- Safe, educational, production-quality patterns
```

### Chunked Prefill (Phase 3)
```
Long prompt: [1000 tokens]
↓
Chunks: [256] [256] [256] [232]
↓
Process one chunk per step
→ Interleave with decode batches
→ Fair scheduling!
```