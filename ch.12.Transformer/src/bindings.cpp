//===- bindings.cpp - Python bindings for Transformer dialect -----*- C++ -*-===//
#include "TransformerDialect.h"
#include "TransformerOps.h"
#include "TransformerPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

#include <llvm/Support/TargetSelect.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ffi.h>

#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>

namespace py = pybind11;
using namespace mlir;

//===----------------------------------------------------------------------===//
// LLVM Initialization
//===----------------------------------------------------------------------===//

static struct LLVMInit {
  LLVMInit() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }
} llvmInit;

//===----------------------------------------------------------------------===//
// Compiler Infrastructure
//===----------------------------------------------------------------------===//

class TransformerCompiler {
public:
  TransformerCompiler() {
    context_.loadDialect<transformer::TransformerDialect,
                         func::FuncDialect, arith::ArithDialect,
                         memref::MemRefDialect, scf::SCFDialect, math::MathDialect,
                         LLVM::LLVMDialect>();
  }

  MLIRContext& getContext() { return context_; }

  bool lowerToLLVM(ModuleOp module) {
    PassManager pm(&context_);

    // Lower transformer dialect to standard dialects
    pm.addNestedPass<func::FuncOp>(createLowerTransformerToStandardPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createCSEPass());

    // Lower to LLVM
    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(createConvertMathToLibmPass());
    pm.addPass(createConvertSCFToCFPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

    return succeeded(pm.run(module));
  }

  void* compileAndGetFunctionPtr(ModuleOp module, const std::string& funcName) {
    registerBuiltinDialectTranslation(*module.getContext());
    registerLLVMDialectTranslation(*module.getContext());

    if (!lowerToLLVM(module)) {
      llvm::errs() << "Failed to lower to LLVM dialect\n";
      return nullptr;
    }

    ExecutionEngineOptions options;
    options.transformer = makeOptimizingTransformer(3, 0, nullptr);

    auto maybeEngine = ExecutionEngine::create(module, options);
    if (!maybeEngine) {
      llvm::errs() << "Failed to create ExecutionEngine: " 
                   << maybeEngine.takeError() << "\n";
      return nullptr;
    }

    auto engine = std::move(*maybeEngine);
    auto expectedFPtr = engine->lookup(funcName);
    if (!expectedFPtr) {
      llvm::errs() << "Failed to lookup function: " << funcName << "\n";
      return nullptr;
    }

    // Keep engine alive
    engines_.emplace_back(engine.release());
    return reinterpret_cast<void*>(*expectedFPtr);
  }

private:
  MLIRContext context_;
  std::vector<ExecutionEngine*> engines_;  // Keep engines alive
};

static TransformerCompiler& getCompiler() {
  static TransformerCompiler compiler;
  return compiler;
}

//===----------------------------------------------------------------------===//
// Memref Marshalling
//===----------------------------------------------------------------------===//

void marshal_memref_2d(std::vector<void*>& args, py::array_t<float> arr) {
  auto buf = arr.request();
  float* data = static_cast<float*>(buf.ptr);
  args.emplace_back(data);
  args.emplace_back(data);
  args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(0)));
  args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[0])));
  args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[1])));
  args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[1])));
  args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(1)));
}

void marshal_memref_1d(std::vector<void*>& args, py::array_t<float> arr) {
  auto buf = arr.request();
  float* data = static_cast<float*>(buf.ptr);
  args.emplace_back(data);
  args.emplace_back(data);
  args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(0)));
  args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[0])));
  args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(1)));
}

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
  py::array_t<float> gamma, beta;  // LayerNorm
  float epsilon = 1e-5f;
  py::array_t<float> weight, bias; // Linear
  float scale_factor = 1.0f;        // Scale

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

  Tensor operator+(const Tensor& other) const {
    auto add_node = std::make_shared<GraphNode>(OpType::Add);
    add_node->inputs.emplace_back(node);
    add_node->inputs.emplace_back(other.node);
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
  node->inputs.emplace_back(input.node);
  node->gamma = gamma;
  node->beta = beta;
  node->epsilon = epsilon;
  node->shape = input.node->shape;
  return Tensor(node);
}

Tensor linear(const Tensor& input, py::array_t<float> weight, py::array_t<float> bias) {
  auto node = std::make_shared<GraphNode>(OpType::Linear);
  node->inputs.emplace_back(input.node);
  node->weight = weight;
  node->bias = bias;
  auto weight_info = weight.request();
  node->shape = {input.node->shape[0], weight_info.shape[0]};
  return Tensor(node);
}

Tensor gelu(const Tensor& input) {
  auto node = std::make_shared<GraphNode>(OpType::Gelu);
  node->inputs.emplace_back(input.node);
  node->shape = input.node->shape;
  return Tensor(node);
}

Tensor add(const Tensor& lhs, const Tensor& rhs) {
  auto node = std::make_shared<GraphNode>(OpType::Add);
  node->inputs.emplace_back(lhs.node);
  node->inputs.emplace_back(rhs.node);
  node->shape = lhs.node->shape;
  return Tensor(node);
}

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
  auto node = std::make_shared<GraphNode>(OpType::Matmul);
  node->inputs.emplace_back(lhs.node);
  node->inputs.emplace_back(rhs.node);
  node->shape = {lhs.node->shape[0], rhs.node->shape[1]};
  return Tensor(node);
}

Tensor transpose(const Tensor& input) {
  auto node = std::make_shared<GraphNode>(OpType::Transpose);
  node->inputs.emplace_back(input.node);
  node->shape = {input.node->shape[1], input.node->shape[0]};
  return Tensor(node);
}

Tensor softmax(const Tensor& input) {
  auto node = std::make_shared<GraphNode>(OpType::Softmax);
  node->inputs.emplace_back(input.node);
  node->shape = input.node->shape;
  return Tensor(node);
}

Tensor scale(const Tensor& input, float scale_factor) {
  auto node = std::make_shared<GraphNode>(OpType::Scale);
  node->inputs.emplace_back(input.node);
  node->scale_factor = scale_factor;
  node->shape = input.node->shape;
  return Tensor(node);
}

//===----------------------------------------------------------------------===//
// High-Level Compositions
//===----------------------------------------------------------------------===//

Tensor ffn(const Tensor& input, py::array_t<float> w1, py::array_t<float> b1,
           py::array_t<float> w2, py::array_t<float> b2) {
  Tensor hidden = linear(input, w1, b1);
  Tensor activated = gelu(hidden);
  return linear(activated, w2, b2);
}

Tensor scaled_dot_product_attention(const Tensor& Q, const Tensor& K, const Tensor& V) {
  int64_t d_k = K.node->shape[1];
  float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));

  Tensor K_T = transpose(K);
  Tensor scores = matmul(Q, K_T);
  Tensor scaled_scores = scale(scores, scale_factor);
  Tensor attn_weights = softmax(scaled_scores);
  return matmul(attn_weights, V);
}

Tensor multi_head_attention(const Tensor& input,
                             py::array_t<float> w_q, py::array_t<float> b_q,
                             py::array_t<float> w_k, py::array_t<float> b_k,
                             py::array_t<float> w_v, py::array_t<float> b_v,
                             py::array_t<float> w_o, py::array_t<float> b_o) {
  Tensor Q = linear(input, w_q, b_q);
  Tensor K = linear(input, w_k, b_k);
  Tensor V = linear(input, w_v, b_v);
  Tensor attn_output = scaled_dot_product_attention(Q, K, V);
  return linear(attn_output, w_o, b_o);
}

Tensor transformer_block(const Tensor& input,
                         py::array_t<float> w_q, py::array_t<float> b_q,
                         py::array_t<float> w_k, py::array_t<float> b_k,
                         py::array_t<float> w_v, py::array_t<float> b_v,
                         py::array_t<float> w_o, py::array_t<float> b_o,
                         py::array_t<float> gamma1, py::array_t<float> beta1,
                         py::array_t<float> w1, py::array_t<float> b1,
                         py::array_t<float> w2, py::array_t<float> b2,
                         py::array_t<float> gamma2, py::array_t<float> beta2) {
  Tensor normed1 = layer_norm(input, gamma1, beta1);
  Tensor attn_out = multi_head_attention(normed1, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o);
  Tensor residual1 = input + attn_out;

  Tensor normed2 = layer_norm(residual1, gamma2, beta2);
  Tensor ffn_out = ffn(normed2, w1, b1, w2, b2);
  return residual1 + ffn_out;
}

Tensor multi_layer_transformer(const Tensor& input,
                                const std::vector<py::array_t<float>>& all_weights) {
  if (all_weights.size() % 16 != 0) {
    throw std::runtime_error("Expected 16 weights per layer");
  }

  int num_layers = all_weights.size() / 16;
  Tensor output = input;

  for (int layer = 0; layer < num_layers; layer++) {
    int base = layer * 16;
    output = transformer_block(
        output,
        all_weights[base + 0],  all_weights[base + 1],
        all_weights[base + 2],  all_weights[base + 3],
        all_weights[base + 4],  all_weights[base + 5],
        all_weights[base + 6],  all_weights[base + 7],
        all_weights[base + 8],  all_weights[base + 9],
        all_weights[base + 10], all_weights[base + 11],
        all_weights[base + 12], all_weights[base + 13],
        all_weights[base + 14], all_weights[base + 15]
    );
  }

  return output;
}

//===----------------------------------------------------------------------===//
// MLIR IR Generation
//===----------------------------------------------------------------------===//

class IRBuilder {
public:
  OpBuilder& builder;
  std::unordered_map<GraphNode*, Value> valueMap;
  std::unordered_map<void*, int>* paramIndex; // Map param pointer → arg index
  int parameterArgOffset; // Offset of first parameter argument

public:
  IRBuilder(OpBuilder& b) : builder(b), paramIndex(nullptr), parameterArgOffset(0) {}

  Value createAlloc(const std::vector<int64_t>& shape) {
    auto memrefType = MemRefType::get(shape, builder.getF32Type());
    return builder.create<memref::AllocOp>(builder.getUnknownLoc(), memrefType);
  }

  Value getParameter(void* ptr) {
    if (!paramIndex || paramIndex->count(ptr) == 0) {
      throw std::runtime_error("Parameter not found in function arguments");
    }
    int idx = (*paramIndex)[ptr];
    auto func = builder.getBlock()->getParentOp();
    return cast<func::FuncOp>(func).getArgument(parameterArgOffset + idx);
  }

  Value compileNode(std::shared_ptr<GraphNode> node) {
    if (valueMap.count(node.get())) {
      return valueMap[node.get()];
    }

    Value result;
    Location loc = builder.getUnknownLoc();

    switch (node->type) {
      case OpType::Input: {
        result = createAlloc(node->shape);
        break;
      }

      case OpType::Add: {
        Value lhs = compileNode(node->inputs[0]);
        Value rhs = compileNode(node->inputs[1]);
        Value output = createAlloc(node->shape);
        builder.create<transformer::AddOp>(loc, lhs, rhs, output);
        result = output;
        break;
      }

      case OpType::LayerNorm: {
        Value input = compileNode(node->inputs[0]);
        Value output = createAlloc(node->shape);
        Value gamma = getParameter(node->gamma.request().ptr);
        Value beta = getParameter(node->beta.request().ptr);
        builder.create<transformer::LayerNormOp>(loc, input, gamma, beta, output);
        result = output;
        break;
      }

      case OpType::Linear: {
        Value input = compileNode(node->inputs[0]);
        Value output = createAlloc(node->shape);
        Value weight = getParameter(node->weight.request().ptr);
        Value bias = getParameter(node->bias.request().ptr);
        builder.create<transformer::LinearOp>(loc, input, weight, bias, output);
        result = output;
        break;
      }

      case OpType::Gelu: {
        Value input = compileNode(node->inputs[0]);
        Value output = createAlloc(node->shape);
        builder.create<transformer::GeluOp>(loc, input, output);
        result = output;
        break;
      }

      case OpType::Matmul: {
        Value lhs = compileNode(node->inputs[0]);
        Value rhs = compileNode(node->inputs[1]);
        Value output = createAlloc(node->shape);
        builder.create<transformer::MatmulOp>(loc, lhs, rhs, output);
        result = output;
        break;
      }

      case OpType::Transpose: {
        Value input = compileNode(node->inputs[0]);
        Value output = createAlloc(node->shape);
        builder.create<transformer::TransposeOp>(loc, input, output);
        result = output;
        break;
      }

      case OpType::Softmax: {
        Value input = compileNode(node->inputs[0]);
        Value output = createAlloc(node->shape);
        builder.create<transformer::SoftmaxOp>(loc, input, output);
        result = output;
        break;
      }

      case OpType::Scale: {
        Value input = compileNode(node->inputs[0]);
        Value output = createAlloc(node->shape);
        Value scale = createAlloc({1});
        builder.create<transformer::ScaleOp>(loc, input, scale, output);
        result = output;
        break;
      }
    }

    valueMap[node.get()] = result;
    return result;
  }
};

//===----------------------------------------------------------------------===//
// JIT Execution (Forward Pass)
//===----------------------------------------------------------------------===//

py::array_t<float> forward(const Tensor& output) {
  TransformerCompiler& compiler = getCompiler();

  // Collect inputs and parameters via topological traversal
  std::vector<std::shared_ptr<GraphNode>> inputs;
  std::vector<py::array_t<float>> parameters; // gamma, beta, weight, bias
  std::unordered_map<GraphNode*, int> visited;
  std::unordered_map<void*, int> paramIndex; // Map param pointer → arg index

  std::function<void(std::shared_ptr<GraphNode>)> collect;
  collect = [&](std::shared_ptr<GraphNode> node) {
    if (visited.count(node.get())) return;
    visited[node.get()] = 1;

    if (node->type == OpType::Input) {
      inputs.emplace_back(node);
    } else {
      // Collect parameters
      if (node->gamma.size() > 0 && paramIndex.count(node->gamma.request().ptr) == 0) {
        paramIndex[node->gamma.request().ptr] = parameters.size();
        parameters.emplace_back(node->gamma);
      }
      if (node->beta.size() > 0 && paramIndex.count(node->beta.request().ptr) == 0) {
        paramIndex[node->beta.request().ptr] = parameters.size();
        parameters.emplace_back(node->beta);
      }
      if (node->weight.size() > 0 && paramIndex.count(node->weight.request().ptr) == 0) {
        paramIndex[node->weight.request().ptr] = parameters.size();
        parameters.emplace_back(node->weight);
      }
      if (node->bias.size() > 0 && paramIndex.count(node->bias.request().ptr) == 0) {
        paramIndex[node->bias.request().ptr] = parameters.size();
        parameters.emplace_back(node->bias);
      }

      for (auto& inp : node->inputs) {
        collect(inp);
      }
    }
  };
  collect(output.node);

  // Build MLIR module
  OpBuilder builder(&compiler.getContext());
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  // Create function signature: (inputs..., parameters..., output) -> ()
  std::vector<Type> funcInputTypes;
  for (auto& inp : inputs) {
    auto memrefType = MemRefType::get(inp->shape, builder.getF32Type());
    funcInputTypes.emplace_back(memrefType);
  }
  for (auto& param : parameters) {
    auto buf = param.request();
    std::vector<int64_t> shape;
    for (ssize_t i = 0; i < buf.ndim; i++) {
      shape.emplace_back(buf.shape[i]);
    }
    auto memrefType = MemRefType::get(shape, builder.getF32Type());
    funcInputTypes.emplace_back(memrefType);
  }
  auto outputType = MemRefType::get(output.node->shape, builder.getF32Type());
  funcInputTypes.emplace_back(outputType); // Output as last argument

  auto funcType = builder.getFunctionType(funcInputTypes, {});
  auto func = func::FuncOp::create(builder.getUnknownLoc(), "compute", funcType);
  auto& entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // Map inputs to function arguments
  IRBuilder irBuilder(builder);
  irBuilder.paramIndex = &paramIndex;
  irBuilder.parameterArgOffset = inputs.size();
  for (size_t i = 0; i < inputs.size(); i++) {
    irBuilder.valueMap[inputs[i].get()] = entryBlock.getArgument(i);
  }

  // Compile computation graph
  Value resultValue = irBuilder.compileNode(output.node);
  Value outputArg = entryBlock.getArgument(inputs.size() + parameters.size());

  // Copy result to output
  builder.create<memref::CopyOp>(builder.getUnknownLoc(), resultValue, outputArg);
  builder.create<func::ReturnOp>(builder.getUnknownLoc());

  module.push_back(func);

  // Compile to native code
  void* fnPtr = compiler.compileAndGetFunctionPtr(module, "compute");
  if (!fnPtr) {
    throw std::runtime_error("Failed to compile function");
  }

  // Prepare arguments: inputs, parameters, output
  std::vector<void*> args;
  for (auto& inp : inputs) {
    marshal_memref_2d(args, inp->data);
  }
  for (auto& param : parameters) {
    auto buf = param.request();
    if (buf.ndim == 1) {
      marshal_memref_1d(args, param);
    } else if (buf.ndim == 2) {
      marshal_memref_2d(args, param);
    } else {
      throw std::runtime_error("Only 1D and 2D parameters supported");
    }
  }

  // Allocate output
  py::array_t<float> result(output.node->shape);
  marshal_memref_2d(args, result);

  // Execute using libffi
  size_t num_args = args.size();
  std::vector<ffi_type*> arg_types(num_args, &ffi_type_pointer);
  std::vector<void*> arg_values(num_args);
  for (size_t i = 0; i < num_args; ++i) {
    arg_values[i] = &args[i];
  }

  ffi_cif cif;
  if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, num_args, &ffi_type_void, arg_types.data()) != FFI_OK) {
    throw std::runtime_error("libffi ffi_prep_cif failed");
  }

  ffi_call(&cif, FFI_FN(fnPtr), nullptr, arg_values.data());

  return result;
}

//===----------------------------------------------------------------------===//
// Python Bindings
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(ch12, m) {
  m.doc() = "Chapter 12: Transformer with Pure MLIR JIT";

  py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
      .def(py::init<py::array_t<float>>())
      .def("__add__", &Tensor::operator+)
      .def("shape", &Tensor::shape);

  m.def("layer_norm", &layer_norm,
        py::arg("input"), py::arg("gamma"), py::arg("beta"), py::arg("epsilon") = 1e-5f);
  m.def("linear", &linear);
  m.def("gelu", &gelu);
  m.def("add", &add);
  m.def("matmul", &matmul);
  m.def("transpose", &transpose);
  m.def("softmax", &softmax);
  m.def("scale", &scale);

  m.def("ffn", &ffn);
  m.def("scaled_dot_product_attention", &scaled_dot_product_attention);
  m.def("attention", &multi_head_attention);
  m.def("transformer_block", &transformer_block);
  m.def("multi_layer_transformer", &multi_layer_transformer);

  m.def("forward", &forward,
        py::arg("output"),
        "JIT compile and execute computation graph");
}