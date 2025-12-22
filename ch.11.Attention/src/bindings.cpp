//===- bindings.cpp - Python bindings for Chapter 11 (Pure MLIR JIT) ------===//
#include "TransformerDialect.h"
#include "TransformerOps.h"
#include "TransformerToStandard.h"

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
using namespace mlir::transformer;

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
    context_.getOrLoadDialect<TransformerDialect>();
    context_.getOrLoadDialect<func::FuncDialect>();
    context_.getOrLoadDialect<arith::ArithDialect>();
    context_.getOrLoadDialect<memref::MemRefDialect>();
    context_.getOrLoadDialect<scf::SCFDialect>();
    context_.getOrLoadDialect<math::MathDialect>();
    context_.getOrLoadDialect<LLVM::LLVMDialect>();
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
    auto transformer = makeOptimizingTransformer(3, 0, nullptr);
    options.transformer = std::move(transformer);

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
  std::vector<ExecutionEngine*> engines_;
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

//===----------------------------------------------------------------------===//
// Computation Graph
//===----------------------------------------------------------------------===//

enum class OpType {
  Input,
  Matmul,
  Add,
  Transpose,
  Softmax,
  Scale
};

struct GraphNode {
  OpType type;
  std::vector<std::shared_ptr<GraphNode>> inputs;
  py::array_t<float> data;
  float scale_factor = 1.0f;
  std::vector<int64_t> shape;

  GraphNode(OpType t) : type(t) {}
};

//===----------------------------------------------------------------------===//
// Tensor API
//===----------------------------------------------------------------------===//

class Tensor {
public:
  std::shared_ptr<GraphNode> node;

  Tensor(std::shared_ptr<GraphNode> n) : node(n) {}

  Tensor(py::array_t<float> data) {
    node = std::make_shared<GraphNode>(OpType::Input);
    node->data = data;
    auto buf = data.request();
    node->shape.resize(buf.ndim);
    for (int i = 0; i < buf.ndim; i++) {
      node->shape[i] = static_cast<int64_t>(buf.shape[i]);
    }
  }

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

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
  auto node = std::make_shared<GraphNode>(OpType::Matmul);
  node->inputs.emplace_back(lhs.node);
  node->inputs.emplace_back(rhs.node);
  node->shape = {lhs.node->shape[0], rhs.node->shape[1]};
  return Tensor(node);
}

Tensor add(const Tensor& lhs, const Tensor& rhs) {
  auto node = std::make_shared<GraphNode>(OpType::Add);
  node->inputs.emplace_back(lhs.node);
  node->inputs.emplace_back(rhs.node);
  node->shape = lhs.node->shape;
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
// High-Level Attention
//===----------------------------------------------------------------------===//

Tensor attention(const Tensor& Q, const Tensor& K, const Tensor& V) {
  int64_t d_k = K.node->shape[1];
  float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));

  Tensor K_T = transpose(K);
  Tensor scores = matmul(Q, K_T);
  Tensor scaled_scores = scale(scores, scale_factor);
  Tensor attn_weights = softmax(scaled_scores);
  return matmul(attn_weights, V);
}

//===----------------------------------------------------------------------===//
// MLIR IR Generation
//===----------------------------------------------------------------------===//

class IRBuilder {
public:
  OpBuilder& builder;
  std::unordered_map<GraphNode*, Value> valueMap;

public:
  IRBuilder(OpBuilder& b) : builder(b) {}

  Value createAlloc(const std::vector<int64_t>& shape) {
    auto memrefType = MemRefType::get(shape, builder.getF32Type());
    return builder.create<memref::AllocOp>(builder.getUnknownLoc(), memrefType);
  }

  Value compileNode(std::shared_ptr<GraphNode> node) {
    if (valueMap.count(node.get())) {
      return valueMap[node.get()];
    }

    Location loc = builder.getUnknownLoc();
    Value result;

    switch (node->type) {
      case OpType::Input:
        throw std::runtime_error("Input nodes should be mapped to function arguments");

      case OpType::Matmul: {
        Value lhs = compileNode(node->inputs[0]);
        Value rhs = compileNode(node->inputs[1]);
        Value output = createAlloc(node->shape);
        builder.create<MatmulOp>(loc, lhs, rhs, output);
        result = output;
        break;
      }

      case OpType::Add: {
        Value lhs = compileNode(node->inputs[0]);
        Value rhs = compileNode(node->inputs[1]);
        Value output = createAlloc(node->shape);
        builder.create<AddOp>(loc, lhs, rhs, output);
        result = output;
        break;
      }

      case OpType::Transpose: {
        Value input = compileNode(node->inputs[0]);
        Value output = createAlloc(node->shape);
        builder.create<TransposeOp>(loc, input, output);
        result = output;
        break;
      }

      case OpType::Softmax: {
        Value input = compileNode(node->inputs[0]);
        Value output = createAlloc(node->shape);
        builder.create<SoftmaxOp>(loc, input, output);
        result = output;
        break;
      }

      case OpType::Scale: {
        Value input = compileNode(node->inputs[0]);
        Value output = createAlloc(node->shape);

        // Create constant scale factor memref (same shape as input)
        Value scale = createAlloc(node->shape);
        Value scaleConst = builder.create<arith::ConstantFloatOp>(
            loc, llvm::APFloat(node->scale_factor), builder.getF32Type());

        // Fill scale memref with constant
        SmallVector<Value> bounds;
        Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
        for (int64_t dim : node->shape) {
          bounds.emplace_back(builder.create<arith::ConstantIndexOp>(loc, dim));
        }

        std::function<void(OpBuilder&, Location, SmallVector<Value>&, size_t)> fillScale;
        fillScale = [&](OpBuilder& b, Location l, SmallVector<Value>& indices, size_t depth) {
          if (depth == node->shape.size()) {
            b.create<memref::StoreOp>(l, scaleConst, scale, indices);
          } else {
            b.create<scf::ForOp>(l, zero, bounds[depth], one, std::nullopt,
              [&](OpBuilder& b2, Location l2, Value iv, ValueRange) {
                indices.emplace_back(iv);
                fillScale(b2, l2, indices, depth + 1);
                indices.pop_back();
                b2.create<scf::YieldOp>(l2);
              });
          }
        };
        SmallVector<Value> indices;
        fillScale(builder, loc, indices, 0);

        builder.create<MulOp>(loc, input, scale, output);
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

  // Collect inputs via topological traversal
  std::vector<std::shared_ptr<GraphNode>> inputs;
  std::unordered_map<GraphNode*, int> visited;

  std::function<void(std::shared_ptr<GraphNode>)> collect;
  collect = [&](std::shared_ptr<GraphNode> node) {
    if (visited.count(node.get())) return;
    visited[node.get()] = 1;

    if (node->type == OpType::Input) {
      inputs.emplace_back(node);
    } else {
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

  // Create function signature: (inputs..., output) -> ()
  std::vector<Type> funcInputTypes;
  for (auto& inp : inputs) {
    auto memrefType = MemRefType::get(inp->shape, builder.getF32Type());
    funcInputTypes.emplace_back(memrefType);
  }
  auto outputType = MemRefType::get(output.node->shape, builder.getF32Type());
  funcInputTypes.emplace_back(outputType);

  auto funcType = builder.getFunctionType(funcInputTypes, {});
  auto func = func::FuncOp::create(builder.getUnknownLoc(), "compute", funcType);
  auto& entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // Map inputs to function arguments
  IRBuilder irBuilder(builder);
  for (size_t i = 0; i < inputs.size(); i++) {
    irBuilder.valueMap[inputs[i].get()] = entryBlock.getArgument(i);
  }

  // Compile computation graph
  Value resultValue = irBuilder.compileNode(output.node);
  Value outputArg = entryBlock.getArgument(inputs.size());

  // Copy result to output
  builder.create<memref::CopyOp>(builder.getUnknownLoc(), resultValue, outputArg);
  builder.create<func::ReturnOp>(builder.getUnknownLoc());

  module.push_back(func);

  // Compile to native code
  void* fnPtr = compiler.compileAndGetFunctionPtr(module, "compute");
  if (!fnPtr) {
    throw std::runtime_error("Failed to compile function");
  }

  // Prepare arguments: inputs, output
  std::vector<void*> args;
  for (auto& inp : inputs) {
    marshal_memref_2d(args, inp->data);
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

PYBIND11_MODULE(ch11, m) {
  m.doc() = "Chapter 11: Attention Mechanism (Pure MLIR JIT)";

  py::class_<Tensor>(m, "Tensor")
    .def(py::init<py::array_t<float>>())
    .def("shape", &Tensor::shape)
    .def("__add__", &Tensor::operator+);

  m.def("matmul", &matmul);
  m.def("add", &add);
  m.def("transpose", &transpose);
  m.def("softmax", &softmax);
  m.def("scale", &scale);
  m.def("attention", &attention, 
        "Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V");
  m.def("forward", &forward, "Execute computation graph via JIT compilation");
}