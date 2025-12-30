//===- bindings.cpp - Python bindings for Chapter 11 (Pure MLIR JIT) ------===//
#include "TransformerDialect.h"
#include "TransformerOps.h"
#include "TransformerToStandard.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
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
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Linalg/Passes.h"

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
    context_.getOrLoadDialect<linalg::LinalgDialect>();
    context_.getOrLoadDialect<memref::MemRefDialect>();
    context_.getOrLoadDialect<scf::SCFDialect>();
    context_.getOrLoadDialect<math::MathDialect>();
    context_.getOrLoadDialect<tensor::TensorDialect>();
    context_.getOrLoadDialect<bufferization::BufferizationDialect>();
    context_.getOrLoadDialect<LLVM::LLVMDialect>();
  }

  MLIRContext& getContext() { return context_; }

  bool lowerToLLVM(ModuleOp module) {
    PassManager pm(&context_);

    // Lower transformer dialect to linalg and standard dialects (tensor-based)
    pm.addNestedPass<func::FuncOp>(createLowerTransformerToStandardPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createCSEPass());

    // Bufferization: tensor → memref
    DialectRegistry registry;
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    context_.appendDialectRegistry(registry);

    bufferization::OneShotBufferizePassOptions bufferizeOptions;
    bufferizeOptions.bufferizeFunctionBoundaries = true;
    pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));
    pm.addPass(bufferization::createBufferResultsToOutParamsPass());
    pm.addPass(createConvertBufferizationToMemRefPass());
    pm.addPass(createCanonicalizerPass());

    // Lower linalg: convert named ops to loops (on memrefs now)
    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

    // Lower to LLVM
    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(createConvertMathToLibmPass());
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

    if (failed(pm.run(module))) {
      llvm::errs() << "Pass manager failed\n";
      return false;
    }
    
    // Debug: print module after lowering
    llvm::errs() << "=== Module after lowering ===\n";
    module.print(llvm::errs());
    llvm::errs() << "\n=== End module ===\n";
    
    return true;
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

  Value createEmptyTensor(const std::vector<int64_t>& shape) {
    auto tensorType = RankedTensorType::get(shape, builder.getF32Type());
    return builder.create<tensor::EmptyOp>(builder.getUnknownLoc(), shape, builder.getF32Type());
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
        auto resultType = RankedTensorType::get(node->shape, builder.getF32Type());
        result = builder.create<MatmulOp>(loc, resultType, lhs, rhs).getResult();
        break;
      }

      case OpType::Add: {
        Value lhs = compileNode(node->inputs[0]);
        Value rhs = compileNode(node->inputs[1]);
        auto resultType = RankedTensorType::get(node->shape, builder.getF32Type());
        result = builder.create<AddOp>(loc, resultType, lhs, rhs).getResult();
        break;
      }

      case OpType::Transpose: {
        Value input = compileNode(node->inputs[0]);
        auto resultType = RankedTensorType::get(node->shape, builder.getF32Type());
        result = builder.create<TransposeOp>(loc, resultType, input).getResult();
        break;
      }

      case OpType::Softmax: {
        Value input = compileNode(node->inputs[0]);
        auto resultType = RankedTensorType::get(node->shape, builder.getF32Type());
        result = builder.create<SoftmaxOp>(loc, resultType, input).getResult();
        break;
      }

      case OpType::Scale: {
        Value input = compileNode(node->inputs[0]);
        
        // Create constant scale tensor (broadcast scalar to shape)
        Value scaleScalar = builder.create<arith::ConstantOp>(
            loc, builder.getFloatAttr(builder.getF32Type(), llvm::APFloat(node->scale_factor)));
        
        // Create empty tensor for scale
        Value emptyScale = createEmptyTensor(node->shape);
        
        // Fill scale tensor with constant using linalg.fill
        Value scaleTensor = builder.create<linalg::FillOp>(loc, scaleScalar, emptyScale).getResult(0);
        
        // Perform element-wise multiplication
        auto resultType = RankedTensorType::get(node->shape, builder.getF32Type());
        result = builder.create<MulOp>(loc, resultType, input, scaleTensor).getResult();
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

  // Create function signature: (inputs...) -> tensor<...>
  std::vector<Type> funcInputTypes;
  for (auto& inp : inputs) {
    auto tensorType = RankedTensorType::get(inp->shape, builder.getF32Type());
    funcInputTypes.emplace_back(tensorType);
  }
  auto outputTensorType = RankedTensorType::get(output.node->shape, builder.getF32Type());

  auto funcType = builder.getFunctionType(funcInputTypes, {outputTensorType});
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

  // Return result tensor
  builder.create<func::ReturnOp>(builder.getUnknownLoc(), resultValue);

  module.push_back(func);

  // Compile to native code
  void* fnPtr = compiler.compileAndGetFunctionPtr(module, "compute");
  if (!fnPtr) {
    throw std::runtime_error("Failed to compile function");
  }

  // After bufferization, function signature changes from:
  //   (tensor, ...) -> tensor
  // to:
  //   (memref*, ..., memref*) -> ()
  // The last argument is the output memref (result of BufferResultsToOutParams pass)

  // Prepare arguments: input memrefs + output memref
  std::vector<void*> args;
  for (auto& inp : inputs) {
    marshal_memref_2d(args, inp->data);
  }

  // Allocate output memref
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