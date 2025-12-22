// Phase 7: Transformer Block & Nano-GPT
//
// Complete transformer implementation with:
// - Causal masked attention
// - Feed-forward network (2 linear + GELU)
// - Residual connections
// - Layer normalization
// - Full forward pass
//
// This is a real neural network! 🚀

#include "common.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace mlir;

// ============================================================================
// Kernel 1: Embedding Lookup
// ============================================================================
// Simple version: output[i] = embedding_table[token_ids[i]]
// Input: token_ids[seq_len] (int32), embedding_table[vocab_size, d_model]
// Output: embedded[seq_len, d_model]

extern "C" void embedding_lookup(
    int* token_ids,           // [seq_len]
    float* embedding_table,   // [vocab_size, d_model]
    float* output,            // [seq_len, d_model]
    int seq_len,
    int vocab_size,
    int d_model
) {
    // Simple CPU implementation - each token looks up its embedding
    for (int i = 0; i < seq_len; i++) {
        int token_id = token_ids[i];

        // Bounds check
        if (token_id < 0 || token_id >= vocab_size) {
            // Zero out invalid tokens
            for (int j = 0; j < d_model; j++) {
                output[i * d_model + j] = 0.0f;
            }
            continue;
        }

        // Copy embedding: output[i,:] = embedding_table[token_id,:]
        for (int j = 0; j < d_model; j++) {
            output[i * d_model + j] = embedding_table[token_id * d_model + j];
        }
    }
}

// ============================================================================
// Kernel 2: Causal Masked Attention
// ============================================================================
// Like attention_kernel but with causal masking: mask out future positions

// Forward declarations of kernels we'll reuse
extern "C" {
    void matmul_kernel(float* A, float* B, float* C, int M, int N, int K);
    void transpose_kernel(float* input, float* output, int M, int N);
    void scale_kernel(float* input, float* output, int N, float scale_factor);
    void softmax_kernel(float* input, float* output, int N);
}

extern "C" void causal_attention_kernel(
    float* Q,      // Query: [seq_len, d_k]
    float* K,      // Key:   [seq_len, d_k]
    float* V,      // Value: [seq_len, d_v]
    float* output, // Output: [seq_len, d_v]
    int seq_len,
    int d_k,
    int d_v
) {
    // Allocate temporary buffers
    int scores_size = seq_len * seq_len;
    int kT_size = d_k * seq_len;

    float* K_T = new float[kT_size];
    float* scores = new float[scores_size];
    float* scaled_scores = new float[scores_size];
    float* masked_scores = new float[scores_size];
    float* attn_weights = new float[scores_size];

    // Step 1: Transpose K to get K^T
    transpose_kernel(K, K_T, seq_len, d_k);

    // Step 2: Compute scores = Q @ K^T
    matmul_kernel(Q, K_T, scores, seq_len, seq_len, d_k);

    // Step 3: Scale by 1/√d_k
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
    scale_kernel(scores, scaled_scores, scores_size, scale_factor);

    // Step 4: Apply causal mask (mask out future positions)
    // For position i, can only attend to positions <= i
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            if (j > i) {
                // Future position - mask with large negative number
                masked_scores[i * seq_len + j] = -1e9f;
            } else {
                masked_scores[i * seq_len + j] = scaled_scores[i * seq_len + j];
            }
        }
    }

    // Step 5: Apply softmax row-wise
    for (int i = 0; i < seq_len; i++) {
        softmax_kernel(masked_scores + i * seq_len, 
                      attn_weights + i * seq_len, 
                      seq_len);
    }

    // Step 6: Compute output = attn_weights @ V
    matmul_kernel(attn_weights, V, output, seq_len, d_v, seq_len);

    // Cleanup
    delete[] K_T;
    delete[] scores;
    delete[] scaled_scores;
    delete[] masked_scores;
    delete[] attn_weights;
}

// ============================================================================
// Kernel 3: Feed-Forward Network
// ============================================================================
// FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
// Typically d_ff = 4 * d_model

extern "C" {
    void gelu_kernel(float* input, float* output, int N);
    void bias_add_kernel(float* input, float bias, float* output, int N);
}

extern "C" void feedforward_kernel(
    float* input,   // [seq_len, d_model]
    float* W1,      // [d_model, d_ff]
    float* b1,      // [d_ff] (we'll broadcast manually)
    float* W2,      // [d_ff, d_model]
    float* b2,      // [d_model]
    float* output,  // [seq_len, d_model]
    int seq_len,
    int d_model,
    int d_ff
) {
    // Allocate temporary buffers
    float* hidden1 = new float[seq_len * d_ff];        // After W1
    float* hidden2 = new float[seq_len * d_ff];        // After bias
    float* hidden3 = new float[seq_len * d_ff];        // After GELU
    float* hidden4 = new float[seq_len * d_model];     // After W2

    // Step 1: hidden1 = input @ W1  [seq_len, d_model] @ [d_model, d_ff] -> [seq_len, d_ff]
    matmul_kernel(input, W1, hidden1, seq_len, d_ff, d_model);

    // Step 2: Add bias b1 (broadcast over seq_len)
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_ff; j++) {
            hidden2[i * d_ff + j] = hidden1[i * d_ff + j] + b1[j];
        }
    }

    // Step 3: Apply GELU activation
    gelu_kernel(hidden2, hidden3, seq_len * d_ff);

    // Step 4: hidden4 = hidden3 @ W2  [seq_len, d_ff] @ [d_ff, d_model] -> [seq_len, d_model]
    matmul_kernel(hidden3, W2, hidden4, seq_len, d_model, d_ff);

    // Step 5: Add bias b2
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            output[i * d_model + j] = hidden4[i * d_model + j] + b2[j];
        }
    }

    // Cleanup
    delete[] hidden1;
    delete[] hidden2;
    delete[] hidden3;
    delete[] hidden4;
}

// ============================================================================
// Kernel 4: Transformer Block
// ============================================================================
// Complete transformer layer:
//   x = LayerNorm(x)
//   x = x + CausalAttention(x)      # Residual connection
//   x = LayerNorm(x)
//   x = x + FFN(x)                   # Residual connection

extern "C" {
    void layernorm_kernel(float* input, float* output, int N, float epsilon);
    void add_kernel(float* A, float* B, float* C, int N);
}

extern "C" void transformer_block(
    float* input,        // [seq_len, d_model]
    // Attention weights
    float* W_q,          // [d_model, d_k]
    float* W_k,          // [d_model, d_k]
    float* W_v,          // [d_model, d_v]
    float* W_o,          // [d_v, d_model]
    // FFN weights
    float* W1,           // [d_model, d_ff]
    float* b1,           // [d_ff]
    float* W2,           // [d_ff, d_model]
    float* b2,           // [d_model]
    float* output,       // [seq_len, d_model]
    int seq_len,
    int d_model,
    int d_k,
    int d_v,
    int d_ff
) {
    // Allocate buffers
    float* x = new float[seq_len * d_model];          // Working buffer
    float* norm1 = new float[seq_len * d_model];      // After first layernorm
    float* Q = new float[seq_len * d_k];
    float* K = new float[seq_len * d_k];
    float* V = new float[seq_len * d_v];
    float* attn_out = new float[seq_len * d_v];
    float* attn_proj = new float[seq_len * d_model];  // After output projection
    float* residual1 = new float[seq_len * d_model];  // After first residual
    float* norm2 = new float[seq_len * d_model];      // After second layernorm
    float* ffn_out = new float[seq_len * d_model];

    // Copy input to working buffer
    std::copy(input, input + seq_len * d_model, x);

    // ===== ATTENTION BLOCK =====

    // 1. LayerNorm
    for (int i = 0; i < seq_len; i++) {
        layernorm_kernel(x + i * d_model, norm1 + i * d_model, d_model, 1e-5f);
    }

    // 2. Project to Q, K, V
    matmul_kernel(norm1, W_q, Q, seq_len, d_k, d_model);
    matmul_kernel(norm1, W_k, K, seq_len, d_k, d_model);
    matmul_kernel(norm1, W_v, V, seq_len, d_v, d_model);

    // 3. Causal self-attention
    causal_attention_kernel(Q, K, V, attn_out, seq_len, d_k, d_v);

    // 4. Output projection
    matmul_kernel(attn_out, W_o, attn_proj, seq_len, d_model, d_v);

    // 5. Residual connection
    add_kernel(x, attn_proj, residual1, seq_len * d_model);

    // ===== FFN BLOCK =====

    // 6. LayerNorm
    for (int i = 0; i < seq_len; i++) {
        layernorm_kernel(residual1 + i * d_model, norm2 + i * d_model, d_model, 1e-5f);
    }

    // 7. Feed-forward network
    feedforward_kernel(norm2, W1, b1, W2, b2, ffn_out, seq_len, d_model, d_ff);

    // 8. Residual connection
    add_kernel(residual1, ffn_out, output, seq_len * d_model);

    // Cleanup
    delete[] x;
    delete[] norm1;
    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] attn_out;
    delete[] attn_proj;
    delete[] residual1;
    delete[] norm2;
    delete[] ffn_out;
}

// ============================================================================
// Kernel 5: KV-Cached Attention (for efficient generation)
// ============================================================================
// Optimized attention that reuses cached K, V from previous tokens
//
// During generation:
//   Step 1: Q, K, V for token 0 → cache K, V
//   Step 2: Q for token 1, use cached K[0] + new K[1], cached V[0] + new V[1]
//   Step 3: Q for token 2, use cached K[0:1] + new K[2], cached V[0:1] + new V[2]
//   ...
//
// This avoids recomputing attention for all previous tokens!

extern "C" void kv_cached_attention(
    float* Q_new,        // Query for new token(s): [new_tokens, d_k]
    float* K_new,        // Key for new token(s): [new_tokens, d_k]
    float* V_new,        // Value for new token(s): [new_tokens, d_v]
    float* K_cache,      // Cached keys: [cache_len, d_k]
    float* V_cache,      // Cached values: [cache_len, d_v]
    float* output,       // Output: [new_tokens, d_v]
    int new_tokens,      // Number of new tokens (usually 1 during generation)
    int cache_len,       // Number of cached tokens
    int d_k,
    int d_v
) {
    int total_len = cache_len + new_tokens;

    // Allocate buffers
    float* K_full = new float[total_len * d_k];
    float* V_full = new float[total_len * d_v];
    float* K_full_T = new float[d_k * total_len];
    float* scores = new float[new_tokens * total_len];
    float* scaled_scores = new float[new_tokens * total_len];
    float* attn_weights = new float[new_tokens * total_len];

    // 1. Concatenate cached and new K, V
    // K_full = [K_cache; K_new]  (vertical concat)
    if (cache_len > 0) {
        std::copy(K_cache, K_cache + cache_len * d_k, K_full);
        std::copy(V_cache, V_cache + cache_len * d_v, V_full);
    }
    std::copy(K_new, K_new + new_tokens * d_k, K_full + cache_len * d_k);
    std::copy(V_new, V_new + new_tokens * d_v, V_full + cache_len * d_v);

    // 2. Transpose K_full to get K_full^T
    transpose_kernel(K_full, K_full_T, total_len, d_k);

    // 3. Compute scores = Q_new @ K_full^T
    // [new_tokens, d_k] @ [d_k, total_len] -> [new_tokens, total_len]
    matmul_kernel(Q_new, K_full_T, scores, new_tokens, total_len, d_k);

    // 4. Scale by 1/√d_k
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
    scale_kernel(scores, scaled_scores, new_tokens * total_len, scale_factor);

    // Note: No causal masking needed here - cache_len positions are all valid,
    // and new_tokens can attend to cache + themselves (already computed)

    // 5. Apply softmax row-wise
    for (int i = 0; i < new_tokens; i++) {
        softmax_kernel(scaled_scores + i * total_len, 
                      attn_weights + i * total_len, 
                      total_len);
    }

    // 6. Compute output = attn_weights @ V_full
    // [new_tokens, total_len] @ [total_len, d_v] -> [new_tokens, d_v]
    matmul_kernel(attn_weights, V_full, output, new_tokens, d_v, total_len);

    // Cleanup
    delete[] K_full;
    delete[] V_full;
    delete[] K_full_T;
    delete[] scores;
    delete[] scaled_scores;
    delete[] attn_weights;
}

// ============================================================================
// Kernel 6: Autoregressive Generation with KV Cache
// ============================================================================
// Generate tokens one at a time using KV cache for efficiency

extern "C" void generate_with_kv_cache(
    int* prompt_tokens,       // Initial prompt: [prompt_len]
    float* token_embeddings,  // [vocab_size, d_model]
    float* pos_embeddings,    // [max_seq_len, d_model]
    // Attention weights
    float* W_q, float* W_k, float* W_v, float* W_o,
    // FFN weights
    float* W1, float* b1, float* W2, float* b2,
    // Output projection
    float* W_out,             // [d_model, vocab_size]
    int* output_tokens,       // [max_len] - generated sequence
    int prompt_len,
    int max_len,              // Maximum sequence length to generate
    int vocab_size,
    int d_model,
    int d_k,
    int d_v,
    int d_ff
) {
    // Allocate KV cache (grows as we generate)
    std::vector<float> K_cache;
    std::vector<float> V_cache;
    K_cache.reserve(max_len * d_k);
    V_cache.reserve(max_len * d_v);

    // Copy prompt to output
    for (int i = 0; i < prompt_len; i++) {
        output_tokens[i] = prompt_tokens[i];
    }

    // Working buffers
    float* embedded = new float[d_model];
    float* x = new float[d_model];
    float* norm1 = new float[d_model];
    float* Q = new float[d_k];
    float* K_new = new float[d_k];
    float* V_new = new float[d_v];
    float* attn_out = new float[d_v];
    float* attn_proj = new float[d_model];
    float* residual1 = new float[d_model];
    float* norm2 = new float[d_model];
    float* ffn_out = new float[d_model];
    float* residual2 = new float[d_model];
    float* final_norm = new float[d_model];
    float* logits = new float[vocab_size];

    // Process prompt tokens first (build initial cache)
    for (int pos = 0; pos < prompt_len; pos++) {
        int token = output_tokens[pos];

        // 1. Embed token
        embedding_lookup(&token, token_embeddings, embedded, 1, vocab_size, d_model);

        // 2. Add positional embedding
        for (int j = 0; j < d_model; j++) {
            x[j] = embedded[j] + pos_embeddings[pos * d_model + j];
        }

        // 3. Transformer block with KV cache

        // LayerNorm
        layernorm_kernel(x, norm1, d_model, 1e-5f);

        // Project to Q, K, V (single token)
        matmul_kernel(norm1, W_q, Q, 1, d_k, d_model);
        matmul_kernel(norm1, W_k, K_new, 1, d_k, d_model);
        matmul_kernel(norm1, W_v, V_new, 1, d_v, d_model);

        // Append K, V to cache
        K_cache.insert(K_cache.end(), K_new, K_new + d_k);
        V_cache.insert(V_cache.end(), V_new, V_new + d_v);

        // KV-cached attention (attend to all previous + current)
        kv_cached_attention(Q, K_new, V_new, 
                           K_cache.data(), V_cache.data(),
                           attn_out, 1, pos, d_k, d_v);

        // Output projection
        matmul_kernel(attn_out, W_o, attn_proj, 1, d_model, d_v);

        // Residual
        for (int j = 0; j < d_model; j++) {
            residual1[j] = x[j] + attn_proj[j];
        }

        // LayerNorm
        layernorm_kernel(residual1, norm2, d_model, 1e-5f);

        // FFN
        feedforward_kernel(norm2, W1, b1, W2, b2, ffn_out, 1, d_model, d_ff);

        // Residual
        for (int j = 0; j < d_model; j++) {
            residual2[j] = residual1[j] + ffn_out[j];
        }
    }

    // Generate new tokens
    for (int pos = prompt_len; pos < max_len; pos++) {
        // Use last residual2 as input for next token

        // Final LayerNorm
        layernorm_kernel(residual2, final_norm, d_model, 1e-5f);

        // Output projection
        matmul_kernel(final_norm, W_out, logits, 1, vocab_size, d_model);

        // Sample next token (greedy: argmax)
        int next_token = 0;
        float max_logit = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                next_token = i;
            }
        }

        output_tokens[pos] = next_token;

        // If we're not at the last position, compute next state
        if (pos + 1 < max_len) {
            int token = next_token;

            // Embed
            embedding_lookup(&token, token_embeddings, embedded, 1, vocab_size, d_model);

            // Add positional embedding
            for (int j = 0; j < d_model; j++) {
                x[j] = embedded[j] + pos_embeddings[pos * d_model + j];
            }

            // Transformer block with KV cache
            layernorm_kernel(x, norm1, d_model, 1e-5f);

            matmul_kernel(norm1, W_q, Q, 1, d_k, d_model);
            matmul_kernel(norm1, W_k, K_new, 1, d_k, d_model);
            matmul_kernel(norm1, W_v, V_new, 1, d_v, d_model);

            // Append to cache
            K_cache.insert(K_cache.end(), K_new, K_new + d_k);
            V_cache.insert(V_cache.end(), V_new, V_new + d_v);

            // KV-cached attention
            kv_cached_attention(Q, K_new, V_new,
                               K_cache.data(), V_cache.data(),
                               attn_out, 1, pos, d_k, d_v);

            matmul_kernel(attn_out, W_o, attn_proj, 1, d_model, d_v);

            for (int j = 0; j < d_model; j++) {
                residual1[j] = x[j] + attn_proj[j];
            }

            layernorm_kernel(residual1, norm2, d_model, 1e-5f);

            feedforward_kernel(norm2, W1, b1, W2, b2, ffn_out, 1, d_model, d_ff);

            for (int j = 0; j < d_model; j++) {
                residual2[j] = residual1[j] + ffn_out[j];
            }
        }
    }

    // Cleanup
    delete[] embedded;
    delete[] x;
    delete[] norm1;
    delete[] Q;
    delete[] K_new;
    delete[] V_new;
    delete[] attn_out;
    delete[] attn_proj;
    delete[] residual1;
    delete[] norm2;
    delete[] ffn_out;
    delete[] residual2;
    delete[] final_norm;
    delete[] logits;
}

// ============================================================================
// Kernel 7: Nano-GPT Forward Pass (Original - for training/full sequence)
// ============================================================================
// Full model: Embedding -> N x Transformer Block -> Output Projection

extern "C" void nanogpt_forward(
    int* token_ids,           // [seq_len]
    float* token_embeddings,  // [vocab_size, d_model]
    float* pos_embeddings,    // [max_seq_len, d_model]
    // Layer 1 weights
    float* W_q1, float* W_k1, float* W_v1, float* W_o1,
    float* W1_1, float* b1_1, float* W2_1, float* b2_1,
    // Output projection
    float* W_out,             // [d_model, vocab_size]
    float* logits,            // [seq_len, vocab_size] - output
    int seq_len,
    int vocab_size,
    int d_model,
    int d_k,
    int d_v,
    int d_ff
) {
    // Allocate buffers
    float* embedded = new float[seq_len * d_model];
    float* pos_embed = new float[seq_len * d_model];
    float* x = new float[seq_len * d_model];
    float* block_out = new float[seq_len * d_model];

    // 1. Token embedding lookup
    embedding_lookup(token_ids, token_embeddings, embedded, 
                    seq_len, vocab_size, d_model);

    // 2. Add positional embeddings
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            x[i * d_model + j] = embedded[i * d_model + j] + pos_embeddings[i * d_model + j];
        }
    }

    // 3. Transformer block(s)
    transformer_block(x, W_q1, W_k1, W_v1, W_o1, W1_1, b1_1, W2_1, b2_1,
                     block_out, seq_len, d_model, d_k, d_v, d_ff);

    // 4. Final layer norm
    float* normalized = new float[seq_len * d_model];
    for (int i = 0; i < seq_len; i++) {
        layernorm_kernel(block_out + i * d_model, normalized + i * d_model, d_model, 1e-5f);
    }

    // 5. Output projection: [seq_len, d_model] @ [d_model, vocab_size] -> [seq_len, vocab_size]
    matmul_kernel(normalized, W_out, logits, seq_len, vocab_size, d_model);

    // Cleanup
    delete[] embedded;
    delete[] pos_embed;
    delete[] x;
    delete[] block_out;
    delete[] normalized;
}