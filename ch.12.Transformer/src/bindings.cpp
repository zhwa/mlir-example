//===- bindings.cpp - Python bindings for Transformer dialect -----*- C++ -*-===//
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>
#include <cmath>

namespace py = pybind11;

//===----------------------------------------------------------------------===//
// Computation Graph
//===----------------------------------------------------------------------===//

enum class OpType {
  Input,
  LayerNorm,
  Linear,
  Gelu,
  Add,
  Matmul,
  Transpose,
  Softmax,
  Scale
};

struct GraphNode {
  OpType type;
  std::vector<std::shared_ptr<GraphNode>> inputs;
  py::array_t<float> data; // For Input nodes

  // Operation parameters
  py::array_t<float> gamma; // LayerNorm scale
  py::array_t<float> beta;  // LayerNorm bias
  float epsilon = 1e-5f;     // LayerNorm epsilon

  py::array_t<float> weight; // Linear weight
  py::array_t<float> bias;   // Linear bias

  float scale_factor = 1.0f;  // Scale operation factor

  std::vector<int64_t> shape;

  GraphNode(OpType t) : type(t) {}
};

//===----------------------------------------------------------------------===//
// Tensor API
//===----------------------------------------------------------------------===//

class Tensor {
public:
  std::shared_ptr<GraphNode> node;

  Tensor(py::array_t<float> data) {
    node = std::make_shared<GraphNode>(OpType::Input);
    node->data = data;
    auto info = data.request();
    for (int i = 0; i < info.ndim; i++) {
      node->shape.emplace_back(info.shape[i]);
    }
  }

  Tensor(std::shared_ptr<GraphNode> n) : node(n) {}

  const std::vector<int64_t>& shape() const { return node->shape; }

  // Operator overloading for clean composition
  Tensor operator+(const Tensor& other) const {
    auto add_node = std::make_shared<GraphNode>(OpType::Add);
    add_node->inputs.push_back(node);
    add_node->inputs.push_back(other.node);
    add_node->shape = node->shape;
    return Tensor(add_node);
  }
};

//===----------------------------------------------------------------------===//
// Operation Builders
//===----------------------------------------------------------------------===//

Tensor layer_norm(const Tensor& input, py::array_t<float> gamma, 
                  py::array_t<float> beta, float epsilon = 1e-5f) {
  auto node = std::make_shared<GraphNode>(OpType::LayerNorm);
  node->inputs.push_back(input.node);
  node->gamma = gamma;
  node->beta = beta;
  node->epsilon = epsilon;
  node->shape = input.node->shape; // Output shape same as input
  return Tensor(node);
}

Tensor linear(const Tensor& input, py::array_t<float> weight, py::array_t<float> bias) {
  auto node = std::make_shared<GraphNode>(OpType::Linear);
  node->inputs.push_back(input.node);
  node->weight = weight;
  node->bias = bias;

  // Output shape: [seq_len, out_features]
  auto weight_info = weight.request();
  node->shape = {input.node->shape[0], weight_info.shape[0]};
  return Tensor(node);
}

Tensor gelu(const Tensor& input) {
  auto node = std::make_shared<GraphNode>(OpType::Gelu);
  node->inputs.push_back(input.node);
  node->shape = input.node->shape;
  return Tensor(node);
}

Tensor add(const Tensor& lhs, const Tensor& rhs) {
  auto node = std::make_shared<GraphNode>(OpType::Add);
  node->inputs.push_back(lhs.node);
  node->inputs.push_back(rhs.node);
  node->shape = lhs.node->shape;
  return Tensor(node);
}

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
  auto node = std::make_shared<GraphNode>(OpType::Matmul);
  node->inputs.push_back(lhs.node);
  node->inputs.push_back(rhs.node);

  // Output shape: [M, N] where lhs is [M, K] and rhs is [K, N]
  node->shape = {lhs.node->shape[0], rhs.node->shape[1]};
  return Tensor(node);
}

Tensor transpose(const Tensor& input) {
  auto node = std::make_shared<GraphNode>(OpType::Transpose);
  node->inputs.push_back(input.node);

  // Swap last two dimensions
  node->shape = {input.node->shape[1], input.node->shape[0]};
  return Tensor(node);
}

Tensor softmax(const Tensor& input) {
  auto node = std::make_shared<GraphNode>(OpType::Softmax);
  node->inputs.push_back(input.node);
  node->shape = input.node->shape;
  return Tensor(node);
}

Tensor scale(const Tensor& input, float scale_factor) {
  auto node = std::make_shared<GraphNode>(OpType::Scale);
  node->inputs.push_back(input.node);
  node->scale_factor = scale_factor;
  node->shape = input.node->shape;
  return Tensor(node);
}

//===----------------------------------------------------------------------===//
// High-Level Compositions
//===----------------------------------------------------------------------===//

Tensor ffn(const Tensor& input, py::array_t<float> w1, py::array_t<float> b1,
           py::array_t<float> w2, py::array_t<float> b2) {
  // FFN: Linear → GELU → Linear
  Tensor hidden = linear(input, w1, b1);
  Tensor activated = gelu(hidden);
  return linear(activated, w2, b2);
}

Tensor scaled_dot_product_attention(const Tensor& Q, const Tensor& K, const Tensor& V) {
  // Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
  int64_t d_k = K.node->shape[1];
  float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));

  Tensor K_T = transpose(K);
  Tensor scores = matmul(Q, K_T);  // [seq_len, seq_len]
  Tensor scaled_scores = scale(scores, scale_factor);
  Tensor attn_weights = softmax(scaled_scores);
  return matmul(attn_weights, V);  // [seq_len, d_model]
}

Tensor multi_head_attention(const Tensor& input,
                             py::array_t<float> w_q, py::array_t<float> b_q,
                             py::array_t<float> w_k, py::array_t<float> b_k,
                             py::array_t<float> w_v, py::array_t<float> b_v,
                             py::array_t<float> w_o, py::array_t<float> b_o) {
  // Project to Q, K, V
  Tensor Q = linear(input, w_q, b_q);
  Tensor K = linear(input, w_k, b_k);
  Tensor V = linear(input, w_v, b_v);

  // Apply scaled dot-product attention
  Tensor attn_output = scaled_dot_product_attention(Q, K, V);

  // Output projection
  return linear(attn_output, w_o, b_o);
}

Tensor transformer_block(const Tensor& input,
                         // Attention weights
                         py::array_t<float> w_q, py::array_t<float> b_q,
                         py::array_t<float> w_k, py::array_t<float> b_k,
                         py::array_t<float> w_v, py::array_t<float> b_v,
                         py::array_t<float> w_o, py::array_t<float> b_o,
                         // LayerNorm 1 params
                         py::array_t<float> gamma1, py::array_t<float> beta1,
                         // FFN weights
                         py::array_t<float> w1, py::array_t<float> b1,
                         py::array_t<float> w2, py::array_t<float> b2,
                         // LayerNorm 2 params
                         py::array_t<float> gamma2, py::array_t<float> beta2) {
  // Pre-norm architecture (like GPT-2)
  // x = x + attention(layer_norm(x))
  Tensor normed1 = layer_norm(input, gamma1, beta1);
  Tensor attn_out = multi_head_attention(normed1, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o);
  Tensor residual1 = input + attn_out;  // Using operator overloading!

  // x = x + ffn(layer_norm(x))
  Tensor normed2 = layer_norm(residual1, gamma2, beta2);
  Tensor ffn_out = ffn(normed2, w1, b1, w2, b2);
  return residual1 + ffn_out;  // Using operator overloading!
}

Tensor multi_layer_transformer(const Tensor& input,
                                const std::vector<py::array_t<float>>& all_weights) {
  // Expect 16 weights per layer (8 attention + 2 LN1 + 4 FFN + 2 LN2)
  if (all_weights.size() % 16 != 0) {
    throw std::runtime_error("Expected 16 weights per layer");
  }

  int num_layers = all_weights.size() / 16;
  Tensor output = input;

  for (int layer = 0; layer < num_layers; layer++) {
    int base = layer * 16;
    output = transformer_block(
        output,
        all_weights[base + 0],  all_weights[base + 1],   // w_q, b_q
        all_weights[base + 2],  all_weights[base + 3],   // w_k, b_k
        all_weights[base + 4],  all_weights[base + 5],   // w_v, b_v
        all_weights[base + 6],  all_weights[base + 7],   // w_o, b_o
        all_weights[base + 8],  all_weights[base + 9],   // gamma1, beta1
        all_weights[base + 10], all_weights[base + 11],  // w1, b1
        all_weights[base + 12], all_weights[base + 13],  // w2, b2
        all_weights[base + 14], all_weights[base + 15]   // gamma2, beta2
    );
  }

  return output;
}

//===----------------------------------------------------------------------===//
// Forward declarations for C++ reference implementations
//===----------------------------------------------------------------------===//

py::array_t<float> add_ref(py::array_t<float> lhs, py::array_t<float> rhs);
py::array_t<float> layer_norm_ref(py::array_t<float> input, py::array_t<float> gamma, py::array_t<float> beta, float epsilon);
py::array_t<float> linear_ref(py::array_t<float> input, py::array_t<float> weight, py::array_t<float> bias);
py::array_t<float> gelu_ref(py::array_t<float> input);
py::array_t<float> matmul_ref(py::array_t<float> lhs, py::array_t<float> rhs);
py::array_t<float> transpose_ref(py::array_t<float> input);
py::array_t<float> softmax_ref(py::array_t<float> input);
py::array_t<float> scale_ref(py::array_t<float> input, float scale_factor);

//===----------------------------------------------------------------------===//
// Graph Interpreter
//===----------------------------------------------------------------------===//

py::array_t<float> forward(const Tensor& output) {
  // Execute computation graph using C++ reference implementations

  std::map<GraphNode*, py::array_t<float>> resultMap;

  std::function<py::array_t<float>(std::shared_ptr<GraphNode>)> execute;
  execute = [&](std::shared_ptr<GraphNode> node) -> py::array_t<float> {
    if (resultMap.count(node.get())) {
      return resultMap[node.get()];
    }

    py::array_t<float> result;

    switch (node->type) {
      case OpType::Input:
        result = node->data;
        break;

      case OpType::Add: {
        auto lhs = execute(node->inputs[0]);
        auto rhs = execute(node->inputs[1]);
        result = add_ref(lhs, rhs);
        break;
      }

      case OpType::LayerNorm: {
        auto input = execute(node->inputs[0]);
        result = layer_norm_ref(input, node->gamma, node->beta, node->epsilon);
        break;
      }

      case OpType::Linear: {
        auto input = execute(node->inputs[0]);
        result = linear_ref(input, node->weight, node->bias);
        break;
      }

      case OpType::Gelu: {
        auto input = execute(node->inputs[0]);
        result = gelu_ref(input);
        break;
      }

      case OpType::Matmul: {
        auto lhs = execute(node->inputs[0]);
        auto rhs = execute(node->inputs[1]);
        result = matmul_ref(lhs, rhs);
        break;
      }

      case OpType::Transpose: {
        auto input = execute(node->inputs[0]);
        result = transpose_ref(input);
        break;
      }

      case OpType::Softmax: {
        auto input = execute(node->inputs[0]);
        result = softmax_ref(input);
        break;
      }

      case OpType::Scale: {
        auto input = execute(node->inputs[0]);
        result = scale_ref(input, node->scale_factor);
        break;
      }
    }

    resultMap[node.get()] = result;
    return result;
  };

  return execute(output.node);
}

//===----------------------------------------------------------------------===//
// C++ Reference Implementation (for interpreter)
//===----------------------------------------------------------------------===//

py::array_t<float> add_ref(py::array_t<float> lhs, py::array_t<float> rhs) {
  auto lhs_buf = lhs.request();
  auto rhs_buf = rhs.request();

  if (lhs_buf.ndim != rhs_buf.ndim) {
    throw std::runtime_error("Dimension mismatch in add");
  }

  // Allocate output
  py::array_t<float> result(lhs_buf.shape);
  auto result_buf = result.request();

  float* lhs_ptr = static_cast<float*>(lhs_buf.ptr);
  float* rhs_ptr = static_cast<float*>(rhs_buf.ptr);
  float* result_ptr = static_cast<float*>(result_buf.ptr);

  size_t size = 1;
  for (ssize_t i = 0; i < lhs_buf.ndim; i++) {
    size *= lhs_buf.shape[i];
  }

  for (size_t i = 0; i < size; i++) {
    result_ptr[i] = lhs_ptr[i] + rhs_ptr[i];
  }

  return result;
}

py::array_t<float> layer_norm_ref(py::array_t<float> input, 
                                   py::array_t<float> gamma,
                                   py::array_t<float> beta,
                                   float epsilon = 1e-5f) {
  auto input_buf = input.request();
  auto gamma_buf = gamma.request();
  auto beta_buf = beta.request();

  if (input_buf.ndim != 2) {
    throw std::runtime_error("Input must be 2D");
  }

  int64_t seq_len = input_buf.shape[0];
  int64_t d_model = input_buf.shape[1];

  auto output = py::array_t<float>({seq_len, d_model});
  auto output_buf = output.request();

  float* input_ptr = static_cast<float*>(input_buf.ptr);
  float* gamma_ptr = static_cast<float*>(gamma_buf.ptr);
  float* beta_ptr = static_cast<float*>(beta_buf.ptr);
  float* output_ptr = static_cast<float*>(output_buf.ptr);

  for (int64_t i = 0; i < seq_len; i++) {
    // Compute mean
    float mean = 0.0f;
    for (int64_t j = 0; j < d_model; j++) {
      mean += input_ptr[i * d_model + j];
    }
    mean /= d_model;

    // Compute variance
    float variance = 0.0f;
    for (int64_t j = 0; j < d_model; j++) {
      float diff = input_ptr[i * d_model + j] - mean;
      variance += diff * diff;
    }
    variance /= d_model;

    // Normalize
    float inv_std = 1.0f / std::sqrt(variance + epsilon);
    for (int64_t j = 0; j < d_model; j++) {
      float normalized = (input_ptr[i * d_model + j] - mean) * inv_std;
      output_ptr[i * d_model + j] = normalized * gamma_ptr[j] + beta_ptr[j];
    }
  }

  return output;
}

py::array_t<float> linear_ref(py::array_t<float> input,
                               py::array_t<float> weight,
                               py::array_t<float> bias) {
  auto input_buf = input.request();
  auto weight_buf = weight.request();
  auto bias_buf = bias.request();

  if (input_buf.ndim != 2 || weight_buf.ndim != 2) {
    throw std::runtime_error("Input and weight must be 2D");
  }

  int64_t seq_len = input_buf.shape[0];
  int64_t in_features = input_buf.shape[1];
  int64_t out_features = weight_buf.shape[0];

  if (weight_buf.shape[1] != in_features) {
    throw std::runtime_error("Input/weight dimension mismatch");
  }

  auto output = py::array_t<float>({seq_len, out_features});
  auto output_buf = output.request();

  float* input_ptr = static_cast<float*>(input_buf.ptr);
  float* weight_ptr = static_cast<float*>(weight_buf.ptr);
  float* bias_ptr = static_cast<float*>(bias_buf.ptr);
  float* output_ptr = static_cast<float*>(output_buf.ptr);

  // output[i, j] = sum_k(input[i, k] * weight[j, k]) + bias[j]
  for (int64_t i = 0; i < seq_len; i++) {
    for (int64_t j = 0; j < out_features; j++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < in_features; k++) {
        sum += input_ptr[i * in_features + k] * weight_ptr[j * in_features + k];
      }
      output_ptr[i * out_features + j] = sum + bias_ptr[j];
    }
  }

  return output;
}

py::array_t<float> gelu_ref(py::array_t<float> input) {
  auto input_buf = input.request();
  auto output = py::array_t<float>(input_buf.shape);
  auto output_buf = output.request();

  float* input_ptr = static_cast<float*>(input_buf.ptr);
  float* output_ptr = static_cast<float*>(output_buf.ptr);

  size_t size = 1;
  for (ssize_t i = 0; i < input_buf.ndim; i++) {
    size *= input_buf.shape[i];
  }

  // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  const float sqrt_2_over_pi = 0.7978845608f;
  const float coeff = 0.044715f;

  for (size_t i = 0; i < size; i++) {
    float x = input_ptr[i];
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    float tanh_val = std::tanh(inner);
    output_ptr[i] = 0.5f * x * (1.0f + tanh_val);
  }

  return output;
}

py::array_t<float> ffn_ref(py::array_t<float> input,
                            py::array_t<float> w1, py::array_t<float> b1,
                            py::array_t<float> w2, py::array_t<float> b2) {
  auto hidden = linear_ref(input, w1, b1);
  auto activated = gelu_ref(hidden);
  return linear_ref(activated, w2, b2);
}

py::array_t<float> matmul_ref(py::array_t<float> lhs, py::array_t<float> rhs) {
  auto lhs_buf = lhs.request();
  auto rhs_buf = rhs.request();

  if (lhs_buf.ndim != 2 || rhs_buf.ndim != 2) {
    throw std::runtime_error("Inputs must be 2D");
  }

  int64_t M = lhs_buf.shape[0];
  int64_t K = lhs_buf.shape[1];
  int64_t N = rhs_buf.shape[1];

  if (rhs_buf.shape[0] != K) {
    throw std::runtime_error("Matrix dimension mismatch");
  }

  auto output = py::array_t<float>({M, N});
  auto output_buf = output.request();

  float* lhs_ptr = static_cast<float*>(lhs_buf.ptr);
  float* rhs_ptr = static_cast<float*>(rhs_buf.ptr);
  float* output_ptr = static_cast<float*>(output_buf.ptr);

  for (int64_t i = 0; i < M; i++) {
    for (int64_t j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += lhs_ptr[i * K + k] * rhs_ptr[k * N + j];
      }
      output_ptr[i * N + j] = sum;
    }
  }

  return output;
}

py::array_t<float> transpose_ref(py::array_t<float> input) {
  auto input_buf = input.request();

  if (input_buf.ndim != 2) {
    throw std::runtime_error("Input must be 2D");
  }

  int64_t M = input_buf.shape[0];
  int64_t N = input_buf.shape[1];

  auto output = py::array_t<float>({N, M});
  auto output_buf = output.request();

  float* input_ptr = static_cast<float*>(input_buf.ptr);
  float* output_ptr = static_cast<float*>(output_buf.ptr);

  for (int64_t i = 0; i < M; i++) {
    for (int64_t j = 0; j < N; j++) {
      output_ptr[j * M + i] = input_ptr[i * N + j];
    }
  }

  return output;
}

py::array_t<float> softmax_ref(py::array_t<float> input) {
  auto input_buf = input.request();

  if (input_buf.ndim != 2) {
    throw std::runtime_error("Input must be 2D");
  }

  int64_t rows = input_buf.shape[0];
  int64_t cols = input_buf.shape[1];

  auto output = py::array_t<float>({rows, cols});
  auto output_buf = output.request();

  float* input_ptr = static_cast<float*>(input_buf.ptr);
  float* output_ptr = static_cast<float*>(output_buf.ptr);

  for (int64_t i = 0; i < rows; i++) {
    // Find max for numerical stability
    float max_val = input_ptr[i * cols];
    for (int64_t j = 1; j < cols; j++) {
      max_val = std::max(max_val, input_ptr[i * cols + j]);
    }

    // Compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int64_t j = 0; j < cols; j++) {
      sum_exp += std::exp(input_ptr[i * cols + j] - max_val);
    }

    // Normalize
    for (int64_t j = 0; j < cols; j++) {
      output_ptr[i * cols + j] = std::exp(input_ptr[i * cols + j] - max_val) / sum_exp;
    }
  }

  return output;
}

py::array_t<float> scale_ref(py::array_t<float> input, float scale_factor) {
  auto input_buf = input.request();
  auto output = py::array_t<float>(input_buf.shape);
  auto output_buf = output.request();

  float* input_ptr = static_cast<float*>(input_buf.ptr);
  float* output_ptr = static_cast<float*>(output_buf.ptr);

  size_t size = 1;
  for (ssize_t i = 0; i < input_buf.ndim; i++) {
    size *= input_buf.shape[i];
  }

  for (size_t i = 0; i < size; i++) {
    output_ptr[i] = input_ptr[i] * scale_factor;
  }

  return output;
}

py::array_t<float> scaled_dot_product_attention_ref(py::array_t<float> Q,
                                                     py::array_t<float> K,
                                                     py::array_t<float> V) {
  // Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
  int64_t d_k = K.request().shape[1];
  float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));

  auto K_T = transpose_ref(K);
  auto scores = matmul_ref(Q, K_T);
  auto scaled_scores = scale_ref(scores, scale_factor);
  auto attn_weights = softmax_ref(scaled_scores);
  return matmul_ref(attn_weights, V);
}

py::array_t<float> multi_head_attention_ref(py::array_t<float> input,
                                             py::array_t<float> w_q, py::array_t<float> b_q,
                                             py::array_t<float> w_k, py::array_t<float> b_k,
                                             py::array_t<float> w_v, py::array_t<float> b_v,
                                             py::array_t<float> w_o, py::array_t<float> b_o) {
  auto Q = linear_ref(input, w_q, b_q);
  auto K = linear_ref(input, w_k, b_k);
  auto V = linear_ref(input, w_v, b_v);

  auto attn_output = scaled_dot_product_attention_ref(Q, K, V);
  return linear_ref(attn_output, w_o, b_o);
}

py::array_t<float> transformer_block_ref(py::array_t<float> input,
                                          // Attention weights
                                          py::array_t<float> w_q, py::array_t<float> b_q,
                                          py::array_t<float> w_k, py::array_t<float> b_k,
                                          py::array_t<float> w_v, py::array_t<float> b_v,
                                          py::array_t<float> w_o, py::array_t<float> b_o,
                                          // LayerNorm 1 params
                                          py::array_t<float> gamma1, py::array_t<float> beta1,
                                          // FFN weights
                                          py::array_t<float> w1, py::array_t<float> b1,
                                          py::array_t<float> w2, py::array_t<float> b2,
                                          // LayerNorm 2 params
                                          py::array_t<float> gamma2, py::array_t<float> beta2) {
  // Pre-norm architecture
  // x = x + attention(layer_norm(x))
  auto normed1 = layer_norm_ref(input, gamma1, beta1);
  auto attn_out = multi_head_attention_ref(normed1, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o);

  // Add residual
  auto input_buf = input.request();
  auto attn_buf = attn_out.request();
  auto residual1 = py::array_t<float>(input_buf.shape);
  auto res1_buf = residual1.request();

  float* input_ptr = static_cast<float*>(input_buf.ptr);
  float* attn_ptr = static_cast<float*>(attn_buf.ptr);
  float* res1_ptr = static_cast<float*>(res1_buf.ptr);

  size_t size = 1;
  for (ssize_t i = 0; i < input_buf.ndim; i++) {
    size *= input_buf.shape[i];
  }

  for (size_t i = 0; i < size; i++) {
    res1_ptr[i] = input_ptr[i] + attn_ptr[i];
  }

  // x = x + ffn(layer_norm(x))
  auto normed2 = layer_norm_ref(residual1, gamma2, beta2);
  auto ffn_out = ffn_ref(normed2, w1, b1, w2, b2);

  // Add residual
  auto ffn_buf = ffn_out.request();
  auto output = py::array_t<float>(input_buf.shape);
  auto output_buf = output.request();

  float* ffn_ptr = static_cast<float*>(ffn_buf.ptr);
  float* output_ptr = static_cast<float*>(output_buf.ptr);

  for (size_t i = 0; i < size; i++) {
    output_ptr[i] = res1_ptr[i] + ffn_ptr[i];
  }

  return output;
}

py::array_t<float> multi_layer_transformer_ref(py::array_t<float> input,
                                                const std::vector<py::array_t<float>>& all_weights) {
  // Expect 16 weights per layer
  if (all_weights.size() % 16 != 0) {
    throw std::runtime_error("Expected 16 weights per layer");
  }

  int num_layers = all_weights.size() / 16;
  auto output = input;

  for (int layer = 0; layer < num_layers; layer++) {
    int base = layer * 16;
    output = transformer_block_ref(
        output,
        all_weights[base + 0],  all_weights[base + 1],   // w_q, b_q
        all_weights[base + 2],  all_weights[base + 3],   // w_k, b_k
        all_weights[base + 4],  all_weights[base + 5],   // w_v, b_v
        all_weights[base + 6],  all_weights[base + 7],   // w_o, b_o
        all_weights[base + 8],  all_weights[base + 9],   // gamma1, beta1
        all_weights[base + 10], all_weights[base + 11],  // w1, b1
        all_weights[base + 12], all_weights[base + 13],  // w2, b2
        all_weights[base + 14], all_weights[base + 15]   // gamma2, beta2
    );
  }

  return output;
}

//===----------------------------------------------------------------------===//
// Python Module
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(ch12, m) {
  m.doc() = "Chapter 12: Transformer with Tensor API";

  py::class_<Tensor>(m, "Tensor")
      .def(py::init<py::array_t<float>>())
      .def("shape", &Tensor::shape)
      .def("__add__", &Tensor::operator+, "Add two tensors (operator overloading)");

  m.def("layer_norm", &layer_norm, 
        py::arg("input"), 
        py::arg("gamma"), 
        py::arg("beta"), 
        py::arg("epsilon") = 1e-5f,
        "Layer normalization");

  m.def("linear", &linear,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        "Linear transformation");

  m.def("gelu", &gelu,
        py::arg("input"),
        "GELU activation");

  m.def("add", &add,
        py::arg("lhs"),
        py::arg("rhs"),
        "Element-wise addition");

  m.def("matmul", &matmul,
        py::arg("lhs"),
        py::arg("rhs"),
        "Matrix multiplication");

  m.def("transpose", &transpose,
        py::arg("input"),
        "Matrix transpose");

  m.def("softmax", &softmax,
        py::arg("input"),
        "Softmax activation");

  m.def("scale", &scale,
        py::arg("input"),
        py::arg("scale_factor"),
        "Scale by constant");

  m.def("ffn", &ffn,
        py::arg("input"),
        py::arg("w1"),
        py::arg("b1"),
        py::arg("w2"),
        py::arg("b2"),
        "Feed-forward network (Linear -> GELU -> Linear)");

  m.def("scaled_dot_product_attention", &scaled_dot_product_attention,
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        "Scaled dot-product attention");

  m.def("multi_head_attention", &multi_head_attention,
        py::arg("input"),
        py::arg("w_q"),
        py::arg("b_q"),
        py::arg("w_k"),
        py::arg("b_k"),
        py::arg("w_v"),
        py::arg("b_v"),
        py::arg("w_o"),
        py::arg("b_o"),
        "Multi-head attention");

  m.def("transformer_block", &transformer_block,
        py::arg("input"),
        py::arg("w_q"),
        py::arg("b_q"),
        py::arg("w_k"),
        py::arg("b_k"),
        py::arg("w_v"),
        py::arg("b_v"),
        py::arg("w_o"),
        py::arg("b_o"),
        py::arg("gamma1"),
        py::arg("beta1"),
        py::arg("w1"),
        py::arg("b1"),
        py::arg("w2"),
        py::arg("b2"),
        py::arg("gamma2"),
        py::arg("beta2"),
        "Transformer block (pre-norm architecture)");

  m.def("multi_layer_transformer", &multi_layer_transformer,
        py::arg("input"),
        py::arg("all_weights"),
        "Multi-layer transformer (stack of N blocks)");

  m.def("forward", &forward,
        py::arg("output"),
        "Execute computation graph using interpreter");

  // C++ reference implementations
  m.def("layer_norm_ref", &layer_norm_ref,
        py::arg("input"),
        py::arg("gamma"),
        py::arg("beta"),
        py::arg("epsilon") = 1e-5f,
        "Layer normalization (C++ reference)");

  m.def("linear_ref", &linear_ref,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        "Linear transformation (C++ reference)");

  m.def("gelu_ref", &gelu_ref,
        py::arg("input"),
        "GELU activation (C++ reference)");

  m.def("ffn_ref", &ffn_ref,
        py::arg("input"),
        py::arg("w1"),
        py::arg("b1"),
        py::arg("w2"),
        py::arg("b2"),
        "Feed-forward network (C++ reference)");

  m.def("matmul_ref", &matmul_ref,
        py::arg("lhs"),
        py::arg("rhs"),
        "Matrix multiplication (C++ reference)");

  m.def("transpose_ref", &transpose_ref,
        py::arg("input"),
        "Matrix transpose (C++ reference)");

  m.def("softmax_ref", &softmax_ref,
        py::arg("input"),
        "Softmax activation (C++ reference)");

  m.def("scale_ref", &scale_ref,
        py::arg("input"),
        py::arg("scale_factor"),
        "Scale by constant (C++ reference)");

  m.def("scaled_dot_product_attention_ref", &scaled_dot_product_attention_ref,
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        "Scaled dot-product attention (C++ reference)");

  m.def("multi_head_attention_ref", &multi_head_attention_ref,
        py::arg("input"),
        py::arg("w_q"),
        py::arg("b_q"),
        py::arg("w_k"),
        py::arg("b_k"),
        py::arg("w_v"),
        py::arg("b_v"),
        py::arg("w_o"),
        py::arg("b_o"),
        "Multi-head attention (C++ reference)");

  m.def("transformer_block_ref", &transformer_block_ref,
        py::arg("input"),
        py::arg("w_q"),
        py::arg("b_q"),
        py::arg("w_k"),
        py::arg("b_k"),
        py::arg("w_v"),
        py::arg("b_v"),
        py::arg("w_o"),
        py::arg("b_o"),
        py::arg("gamma1"),
        py::arg("beta1"),
        py::arg("w1"),
        py::arg("b1"),
        py::arg("w2"),
        py::arg("b2"),
        py::arg("gamma2"),
        py::arg("beta2"),
        "Transformer block (C++ reference)");

  m.def("multi_layer_transformer_ref", &multi_layer_transformer_ref,
        py::arg("input"),
        py::arg("all_weights"),
        "Multi-layer transformer (C++ reference)");
}