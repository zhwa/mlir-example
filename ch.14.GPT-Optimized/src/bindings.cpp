//===- bindings.cpp - Python bindings for Transformer dialect -----*- C++ -*-===//
#include "TransformerDialect.h"
#include "TransformerOps.h"
#include "TransformerPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"

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
    // Register Transform dialect extensions for Linalg and Vector
    DialectRegistry registry;
    mlir::linalg::registerTransformDialectExtension(registry);
    mlir::vector::registerTransformDialectExtension(registry);
    context_.appendDialectRegistry(registry);
    
    context_.loadDialect<transformer::TransformerDialect,
                         func::FuncDialect, arith::ArithDialect,
                         linalg::LinalgDialect,  // Phase 1: Add Linalg for optimizations
                         vector::VectorDialect,  // Phase 4: Add Vector for SIMD
                         memref::MemRefDialect, scf::SCFDialect, math::MathDialect,
                         LLVM::LLVMDialect,
                         transform::TransformDialect>();  // Phase 6: Transform for vectorization
  }
  
  // Build comprehensive Transform dialect IR for modern optimization
  // This demonstrates production-grade Transform dialect usage
  ModuleOp buildOptimizationTransform(OpBuilder &builder, Location loc) {
    auto transformModule = builder.create<ModuleOp>(loc);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(transformModule.getBody());
    
    // Create transform.sequence as top-level operation
    auto anyOpType = transform::OperationType::get(
        builder.getContext(), builder.getStringAttr("builtin.module"));
    
    auto sequence = builder.create<transform::SequenceOp>(
        loc,
        /*resultTypes=*/TypeRange{},
        /*failure_propagation_mode=*/transform::FailurePropagationMode::Suppress,  // Suppress to continue on failure
        /*root=*/Value(),  // Null value means use top-level
        /*extra_bindings=*/ValueRange{});
    
    Region &region = sequence.getBodyRegion();
    Block *block = builder.createBlock(&region);
    block->addArgument(anyOpType, loc);
    builder.setInsertionPointToStart(block);
    
    Value target = block->getArgument(0);
    
    // Phase 1: Match and tile matmul operations
    // Tiling is essential for:
    // 1. Cache locality (32x32x32 fits in L1 cache)
    // 2. Vectorization enablement (tiled loops are vectorizable)
    // 3. Parallelization opportunities
    
    // Build array of operation names to match
    SmallVector<StringRef> matmulOps = {"linalg.matmul"};
    auto matchMatmul = builder.create<transform::MatchOp>(
        loc,
        /*results=*/transform::AnyOpType::get(builder.getContext()),
        target,
        /*opNames=*/matmulOps);
    
    // Tile matmul: [32, 32, 32] for good cache locality and vectorization
    // These sizes work well for AVX2 (8-wide float SIMD)
    SmallVector<int64_t> tileSizes = {32, 32, 32};
    auto tileMatmul = builder.create<transform::TileUsingForOp>(
        loc,
        /*resultTypes=*/TypeRange{transform::AnyOpType::get(builder.getContext()),
                                   transform::AnyOpType::get(builder.getContext()),
                                   transform::AnyOpType::get(builder.getContext()),
                                   transform::AnyOpType::get(builder.getContext())},
        /*target=*/matchMatmul.getResult(),
        /*dynamic_sizes=*/ValueRange{},
        /*static_sizes=*/builder.getDenseI64ArrayAttr(tileSizes),
        /*interchange=*/DenseI64ArrayAttr(),
        /*scalable_sizes=*/DenseBoolArrayAttr());
    
    Value tiledMatmul = tileMatmul.getResult(0);
    
    // Phase 2: Match and tile generic ops (GELU, Add, etc.)
    // Element-wise ops benefit from tiling for cache and vectorization
    SmallVector<StringRef> genericOps = {"linalg.generic"};
    auto matchGeneric = builder.create<transform::MatchOp>(
        loc,
        /*results=*/transform::AnyOpType::get(builder.getContext()),
        target,
        /*opNames=*/genericOps);
    
    // Tile generic ops: [128] - larger tiles for element-wise (less reuse)
    SmallVector<int64_t> genericTileSizes = {128};
    auto tileGeneric = builder.create<transform::TileUsingForOp>(
        loc,
        /*resultTypes=*/TypeRange{transform::AnyOpType::get(builder.getContext()),
                                   transform::AnyOpType::get(builder.getContext())},
        /*target=*/matchGeneric.getResult(),
        /*dynamic_sizes=*/ValueRange{},
        /*static_sizes=*/builder.getDenseI64ArrayAttr(genericTileSizes),
        /*interchange=*/DenseI64ArrayAttr(),
        /*scalable_sizes=*/DenseBoolArrayAttr());
    
    Value tiledGeneric = tileGeneric.getResult(0);
    
    // Phase 3: Tile-and-fuse producers into consumers
    // This is the MODERN way: tile ops and fuse producers greedily
    // Replaces old createLinalgElementwiseOpFusionPass()
    // Benefits:
    // 1. Reduces memory traffic (fused ops share cache)
    // 2. Enables better vectorization (larger computation kernels)
    // 3. Declarative and composable with other transforms
    
    // Fuse into matmul tiles (32x32x32 tiles with fused producers)
    // Note: FuseOp expects ArrayAttr for tile sizes
    SmallVector<Attribute> matmulTileAttrs;
    for (auto size : tileSizes) {
      matmulTileAttrs.push_back(builder.getI64IntegerAttr(size));
    }
    auto fuseMatmul = builder.create<transform::FuseOp>(
        loc,
        /*resultTypes=*/TypeRange{transform::AnyOpType::get(builder.getContext()),
                                   transform::AnyOpType::get(builder.getContext()),
                                   transform::AnyOpType::get(builder.getContext()),
                                   transform::AnyOpType::get(builder.getContext())},
        /*target=*/tiledMatmul,
        /*tile_sizes=*/builder.getArrayAttr(matmulTileAttrs),
        /*tile_interchange=*/ArrayAttr(),
        /*apply_cleanup=*/false);
    
    Value fusedMatmul = fuseMatmul.getResult(0);
    
    // Fuse into generic op tiles (128 tiles with fused producers)
    SmallVector<Attribute> genericTileAttrs;
    for (auto size : genericTileSizes) {
      genericTileAttrs.push_back(builder.getI64IntegerAttr(size));
    }
    auto fuseGeneric = builder.create<transform::FuseOp>(
        loc,
        /*resultTypes=*/TypeRange{transform::AnyOpType::get(builder.getContext()),
                                   transform::AnyOpType::get(builder.getContext())},
        /*target=*/tiledGeneric,
        /*tile_sizes=*/builder.getArrayAttr(genericTileAttrs),
        /*tile_interchange=*/ArrayAttr(),
        /*apply_cleanup=*/false);
    
    Value fusedGeneric = fuseGeneric.getResult(0);
    
    // Phase 4: Vectorize fused operations
    // Now that ops are tiled AND fused, vectorization creates efficient SIMD kernels
    // The fused ops will be vectorized as single units
    builder.create<transform::VectorizeOp>(
        loc,
        /*resultTypes=*/TypeRange{},
        fusedMatmul,
        /*vector_sizes=*/ValueRange{},
        /*static_vector_sizes=*/DenseI64ArrayAttr(),
        /*vectorize_nd_extract=*/UnitAttr(),
        /*scalable_sizes=*/DenseBoolArrayAttr());
    
    builder.create<transform::VectorizeOp>(
        loc,
        /*resultTypes=*/TypeRange{},
        fusedGeneric,
        /*vector_sizes=*/ValueRange{},
        /*static_vector_sizes=*/DenseI64ArrayAttr(),
        /*vectorize_nd_extract=*/UnitAttr(),
        /*scalable_sizes=*/DenseBoolArrayAttr());
    
    // Phase 5: Apply patterns for cleanup and optimization
    // Use transform.apply_patterns with pattern descriptors
    auto applyPatterns = builder.create<transform::ApplyPatternsOp>(loc, target);
    {
      OpBuilder::InsertionGuard guard(builder);
      Region &patternsRegion = applyPatterns.getRegion();
      Block *patternsBlock = builder.createBlock(&patternsRegion);
      builder.setInsertionPointToStart(patternsBlock);
      
      // Canonicalization patterns to clean up after transformations
      builder.create<transform::ApplyCanonicalizationPatternsOp>(loc);
      
      // Tiling-specific canonicalization for linalg
      builder.create<transform::ApplyTilingCanonicalizationPatternsOp>(loc);
    }
    
    builder.create<transform::YieldOp>(loc);
    
    return transformModule;
  }

  MLIRContext& getContext() { return context_; }

  bool lowerToLLVM(ModuleOp module) {
    PassManager pm(&context_);

    // Lower transformer dialect to standard dialects (now includes linalg ops)
    pm.addNestedPass<func::FuncOp>(createLowerTransformerToStandardPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createCSEPass());

    // Phase 3-6: Apply comprehensive Transform dialect optimizations
    // Modern approach: tile → fuse → vectorize → cleanup (all declarative!)
    // All done declaratively via Transform dialect (not old-style passes)
    OpBuilder builder(&context_);
    auto transformModule = buildOptimizationTransform(builder, module.getLoc());
    
    // Apply the transform sequence to the payload module
    auto sequenceOp = *transformModule.getBody()->getOps<transform::SequenceOp>().begin();
    RaggedArray<transform::MappedValue> emptyMapping;
    
    // Note: Using Suppress mode so we continue even if some ops can't be optimized
    if (failed(transform::applyTransforms(
            module, cast<transform::TransformOpInterface>(sequenceOp.getOperation()),
            emptyMapping,
            transform::TransformOptions()))) {
      // This is informational - not a hard error with Suppress mode
      llvm::errs() << "Note: Some Transform dialect optimizations were not applicable\n";
    }
    
    // Clean up any leftover linalg ops by lowering to loops
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(createCanonicalizerPass());

    // Loop optimizations that help enable vectorization
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    
    // Prepare loops for vectorization
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // Phase 6: Lower vector operations to LLVM SIMD
    // Handle vector operations that couldn't be fully lowered
    pm.addPass(createConvertVectorToSCFPass());  // Vector masking and unrolling
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createConvertVectorToLLVMPass());  // Vector → LLVM SIMD intrinsics
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

void marshal_memref_3d(std::vector<void*>& args, py::array_t<float> arr) {
  auto buf = arr.request();
  float* data = static_cast<float*>(buf.ptr);
  args.emplace_back(data);
  args.emplace_back(data);
  args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(0)));
  args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[0])));
  args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[1])));
  args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[2])));
  // Strides: [dim1*dim2, dim2, 1]
  args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[1] * buf.shape[2])));
  args.emplace_back(reinterpret_cast<void*>(static_cast<intptr_t>(buf.shape[2])));
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

// Marshal int32 arrays (for embedding indices)
void marshal_memref_1d_i32(std::vector<void*>& args, py::array_t<int32_t> arr) {
  auto buf = arr.request();
  int32_t* data = static_cast<int32_t*>(buf.ptr);
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
  Scale,
  Embedding,        // Chapter 13: Token embedding lookup
  CreateCausalMask, // Chapter 13: Generate causal mask
  MaskedSoftmax,    // Chapter 13: Softmax with mask
  RoPE              // Chapter 13: Rotary position embeddings
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
  py::array_t<int32_t> indices;    // Embedding: token IDs
  py::array_t<float> table;        // Embedding: lookup table
  py::array_t<float> mask;         // MaskedSoftmax: attention mask

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
  // Weight is stored as (in_features, out_features) in numpy
  // Output shape is (batch_size, out_features)
  node->shape = {input.node->shape[0], weight_info.shape[1]};
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

// Token embedding lookup (Chapter 13)
// indices: [seq_len] int32, table: [vocab_size, d_model] float
// output: [seq_len, d_model] float
Tensor embedding(py::array_t<int32_t> indices, py::array_t<float> table) {
  auto node = std::make_shared<GraphNode>(OpType::Embedding);
  node->indices = indices;
  node->table = table;

  // Output shape: [seq_len, d_model]
  int64_t seq_len = static_cast<int64_t>(indices.shape(0));
  int64_t d_model = static_cast<int64_t>(table.shape(1));
  node->shape = {seq_len, d_model};

  return Tensor(node);
}

// Create causal attention mask (Chapter 13)
// seq_len: sequence length
// Returns: [seq_len, seq_len] lower triangular mask
Tensor create_causal_mask(int64_t seq_len) {
  auto node = std::make_shared<GraphNode>(OpType::CreateCausalMask);
  node->shape = {seq_len, seq_len};
  return Tensor(node);
}

// Masked softmax (Chapter 13)
// input: [batch, seq_len, seq_len] logits
// mask: [seq_len, seq_len] additive mask
// Returns: [batch, seq_len, seq_len] attention weights
Tensor masked_softmax(const Tensor& input, const Tensor& mask) {
  auto node = std::make_shared<GraphNode>(OpType::MaskedSoftmax);
  node->inputs.emplace_back(input.node);
  node->inputs.emplace_back(mask.node);
  node->shape = input.node->shape;
  return Tensor(node);
}

// Apply Rotary Position Embeddings (Chapter 13)
// input: [seq_len, d_model] query or key matrix
// Returns: [seq_len, d_model] with position information encoded via rotation
Tensor rope(const Tensor& input) {
  auto node = std::make_shared<GraphNode>(OpType::RoPE);
  node->inputs.emplace_back(input.node);
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
// GPT-Specific Compositions (Chapter 13)
//===----------------------------------------------------------------------===//

// GPT Attention with RoPE and causal masking
// input: [seq_len, d_model]
// Returns attention output with position encoding and causal masking
Tensor gpt_attention(const Tensor& input,
                     py::array_t<float> w_q, py::array_t<float> b_q,
                     py::array_t<float> w_k, py::array_t<float> b_k,
                     py::array_t<float> w_v, py::array_t<float> b_v,
                     py::array_t<float> w_o, py::array_t<float> b_o) {
  // Project to Q, K, V
  Tensor Q = linear(input, w_q, b_q);
  Tensor K = linear(input, w_k, b_k);
  Tensor V = linear(input, w_v, b_v);

  // Apply RoPE to Q and K
  Tensor Q_rope = rope(Q);
  Tensor K_rope = rope(K);

  // Compute attention with causal mask
  int64_t seq_len = Q.node->shape[0];
  int64_t d_k = K.node->shape[1];
  float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));

  Tensor K_T = transpose(K_rope);
  Tensor scores = matmul(Q_rope, K_T);
  Tensor scaled_scores = scale(scores, scale_factor);

  // Create and apply causal mask
  Tensor causal_mask = create_causal_mask(seq_len);
  Tensor attn_weights = masked_softmax(scaled_scores, causal_mask);

  Tensor attn_output = matmul(attn_weights, V);
  return linear(attn_output, w_o, b_o);
}

// KV-cached attention for incremental generation
// Only computes attention for new token, using cached K/V from previous tokens
// Pure CPU implementation without MLIR compilation
py::array_t<float> gpt_attention_cached(
    py::array_t<float> new_token_hidden,  // [1, d_model] - single new token
    py::array_t<float> k_cache,           // [max_seq, d_model] - cached keys
    py::array_t<float> v_cache,           // [max_seq, d_model] - cached values
    int cache_pos,                        // Position to write new K/V
    py::array_t<float> w_q, py::array_t<float> b_q,
    py::array_t<float> w_k, py::array_t<float> b_k,
    py::array_t<float> w_v, py::array_t<float> b_v,
    py::array_t<float> w_o, py::array_t<float> b_o) {

  // Get shapes
  int64_t d_model = new_token_hidden.shape(1);
  auto h_ptr = new_token_hidden.data();
  auto wq_ptr = w_q.data();
  auto bq_ptr = b_q.data();
  auto wk_ptr = w_k.data();
  auto bk_ptr = b_k.data();
  auto wv_ptr = w_v.data();
  auto bv_ptr = b_v.data();
  auto wo_ptr = w_o.data();
  auto bo_ptr = b_o.data();

  auto k_cache_ptr = k_cache.mutable_data();
  auto v_cache_ptr = v_cache.mutable_data();

  // Compute Q = h @ Wq.T + bq  [1, d_model]
  py::array_t<float> q(std::vector<ssize_t>{1, d_model});
  auto q_ptr = q.mutable_data();
  for (int64_t j = 0; j < d_model; j++) {
    float sum = bq_ptr[j];
    for (int64_t k = 0; k < d_model; k++) {
      sum += h_ptr[k] * wq_ptr[j * d_model + k];
    }
    q_ptr[j] = sum;
  }

  // Compute K_new = h @ Wk.T + bk and store in cache
  for (int64_t j = 0; j < d_model; j++) {
    float sum = bk_ptr[j];
    for (int64_t k = 0; k < d_model; k++) {
      sum += h_ptr[k] * wk_ptr[j * d_model + k];
    }
    k_cache_ptr[cache_pos * d_model + j] = sum;
  }

  // Compute V_new = h @ Wv.T + bv and store in cache
  for (int64_t j = 0; j < d_model; j++) {
    float sum = bv_ptr[j];
    for (int64_t k = 0; k < d_model; k++) {
      sum += h_ptr[k] * wv_ptr[j * d_model + k];
    }
    v_cache_ptr[cache_pos * d_model + j] = sum;
  }

  // Apply RoPE to Q and K_cache[cache_pos] (simplified - skip for now)
  // TODO: Add RoPE support in cached attention

  // Compute attention scores: Q @ K_cache[:cache_pos+1].T  [1, cache_pos+1]
  int64_t valid_len = cache_pos + 1;
  py::array_t<float> scores(std::vector<ssize_t>{1, valid_len});
  auto scores_ptr = scores.mutable_data();
  float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_model));

  for (int64_t j = 0; j < valid_len; j++) {
    float score = 0.0f;
    for (int64_t k = 0; k < d_model; k++) {
      score += q_ptr[k] * k_cache_ptr[j * d_model + k];
    }
    scores_ptr[j] = score * scale_factor;
  }

  // Softmax
  float max_score = scores_ptr[0];
  for (int64_t j = 1; j < valid_len; j++) {
    max_score = std::max(max_score, scores_ptr[j]);
  }
  float sum_exp = 0.0f;
  for (int64_t j = 0; j < valid_len; j++) {
    scores_ptr[j] = std::exp(scores_ptr[j] - max_score);
    sum_exp += scores_ptr[j];
  }
  for (int64_t j = 0; j < valid_len; j++) {
    scores_ptr[j] /= sum_exp;
  }

  // Attention output: scores @ V_cache[:valid_len]  [1, d_model]
  py::array_t<float> attn(std::vector<ssize_t>{1, d_model});
  auto attn_ptr = attn.mutable_data();
  std::fill(attn_ptr, attn_ptr + d_model, 0.0f);

  for (int64_t j = 0; j < valid_len; j++) {
    float weight = scores_ptr[j];
    for (int64_t k = 0; k < d_model; k++) {
      attn_ptr[k] += weight * v_cache_ptr[j * d_model + k];
    }
  }

  // Output projection: attn @ Wo.T + bo  [1, d_model]
  py::array_t<float> output(std::vector<ssize_t>{1, d_model});
  auto out_ptr = output.mutable_data();
  for (int64_t j = 0; j < d_model; j++) {
    float sum = bo_ptr[j];
    for (int64_t k = 0; k < d_model; k++) {
      sum += attn_ptr[k] * wo_ptr[j * d_model + k];
    }
    out_ptr[j] = sum;
  }

  return output;
}

// GPT Block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
Tensor gpt_block(const Tensor& input,
                 py::array_t<float> w_q, py::array_t<float> b_q,
                 py::array_t<float> w_k, py::array_t<float> b_k,
                 py::array_t<float> w_v, py::array_t<float> b_v,
                 py::array_t<float> w_o, py::array_t<float> b_o,
                 py::array_t<float> w1, py::array_t<float> b1,
                 py::array_t<float> w2, py::array_t<float> b2,
                 py::array_t<float> gamma1, py::array_t<float> beta1,
                 py::array_t<float> gamma2, py::array_t<float> beta2) {
  // Pre-norm attention
  Tensor normed1 = layer_norm(input, gamma1, beta1);
  Tensor attn_out = gpt_attention(normed1, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o);
  Tensor residual1 = input + attn_out;

  // Pre-norm FFN
  Tensor normed2 = layer_norm(residual1, gamma2, beta2);
  Tensor ffn_out = ffn(normed2, w1, b1, w2, b2);
  return residual1 + ffn_out;
}

// Full GPT forward pass
// indices: [seq_len] token IDs
// embedding_table: [vocab_size, d_model]
// all_weights: list of weights for all layers (16 per layer)
// final_gamma, final_beta: final layer norm
// Returns: [seq_len, d_model] hidden states
Tensor gpt_forward(py::array_t<int32_t> indices,
                   py::array_t<float> embedding_table,
                   const std::vector<py::array_t<float>>& all_weights,
                   py::array_t<float> final_gamma,
                   py::array_t<float> final_beta) {
  if (all_weights.size() % 16 != 0) {
    throw std::runtime_error("Expected 16 weights per layer");
  }

  int num_layers = all_weights.size() / 16;

  // Token embeddings
  Tensor hidden = embedding(indices, embedding_table);

  // Apply transformer blocks
  for (int layer = 0; layer < num_layers; layer++) {
    int base = layer * 16;
    hidden = gpt_block(
        hidden,
        all_weights[base + 0],  all_weights[base + 1],   // Q
        all_weights[base + 2],  all_weights[base + 3],   // K
        all_weights[base + 4],  all_weights[base + 5],   // V
        all_weights[base + 6],  all_weights[base + 7],   // O
        all_weights[base + 8],  all_weights[base + 9],   // FFN W1
        all_weights[base + 10], all_weights[base + 11],  // FFN W2
        all_weights[base + 12], all_weights[base + 13],  // LN1
        all_weights[base + 14], all_weights[base + 15]   // LN2
    );
  }

  // Final layer norm
  return layer_norm(hidden, final_gamma, final_beta);
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

        // Create constant scale factor
        Value scale = createAlloc({1});
        Value zeroIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
        Value scaleConst = builder.create<arith::ConstantFloatOp>(
            loc, llvm::APFloat(node->scale_factor), builder.getF32Type());
        builder.create<memref::StoreOp>(loc, scaleConst, scale, ValueRange{zeroIdx});

        builder.create<transformer::ScaleOp>(loc, input, scale, output);
        result = output;
        break;
      }

      case OpType::Embedding: {
        // indices: [seq_len] int32, table: [vocab_size, d_model] float
        // output: [seq_len, d_model] float
        Value indicesValue = getParameter(node->indices.request().ptr);
        Value tableValue = getParameter(node->table.request().ptr);
        Value output = createAlloc(node->shape);
        builder.create<transformer::EmbeddingOp>(loc, indicesValue, tableValue, output);
        result = output;
        break;
      }

      case OpType::CreateCausalMask: {
        // output: [seq_len, seq_len] causal mask
        Value output = createAlloc(node->shape);
        builder.create<transformer::CreateCausalMaskOp>(loc, output);
        result = output;
        break;
      }

      case OpType::MaskedSoftmax: {
        // input: [batch, seq_len, seq_len], mask: [seq_len, seq_len]
        // output: [batch, seq_len, seq_len]
        Value input = compileNode(node->inputs[0]);
        Value mask = compileNode(node->inputs[1]);
        Value output = createAlloc(node->shape);
        builder.create<transformer::MaskedSoftmaxOp>(loc, input, mask, output);
        result = output;
        break;
      }

      case OpType::RoPE: {
        // input: [seq_len, d_model]
        // output: [seq_len, d_model] with rotary position embeddings
        Value input = compileNode(node->inputs[0]);
        Value output = createAlloc(node->shape);
        builder.create<transformer::RoPEOp>(loc, input, output);
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
  std::vector<py::array_t<float>> parameters; // gamma, beta, weight, bias, table
  std::vector<py::array_t<int32_t>> int32_parameters; // indices for embedding
  std::unordered_map<GraphNode*, int> visited;
  std::unordered_map<void*, int> paramIndex; // Map param pointer → arg index (across all param types)

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
      // Chapter 13: Embedding indices (int32) and table (float)
      if (node->indices.size() > 0 && paramIndex.count(node->indices.request().ptr) == 0) {
        paramIndex[node->indices.request().ptr] = int32_parameters.size();
        int32_parameters.emplace_back(node->indices);
      }
      if (node->table.size() > 0 && paramIndex.count(node->table.request().ptr) == 0) {
        paramIndex[node->table.request().ptr] = parameters.size();
        parameters.emplace_back(node->table);
      }

      for (auto& inp : node->inputs) {
        collect(inp);
      }
    }
  };
  collect(output.node);

  // Fix float parameter indices: add int32_parameters.size() offset
  // (During collection, float params were indexed as if starting at 0,
  // but they actually start after int32 params in the function signature)
  int int32_count = static_cast<int>(int32_parameters.size());
  if (int32_count > 0) {
    for (auto& [ptr, idx] : paramIndex) {
      // Check if this parameter is in the int32_parameters list
      bool is_int32 = false;
      for (const auto& int32_param : int32_parameters) {
        if (int32_param.request().ptr == ptr) {
          is_int32 = true;
          break;
        }
      }
      // If it's a float parameter, add the int32 offset
      if (!is_int32) {
        idx += int32_count;
      }
    }
  }

  // Build MLIR module
  OpBuilder builder(&compiler.getContext());
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  // Create function signature: (inputs..., int32_params..., float_params..., output) -> ()
  std::vector<Type> funcInputTypes;
  for (auto& inp : inputs) {
    auto memrefType = MemRefType::get(inp->shape, builder.getF32Type());
    funcInputTypes.emplace_back(memrefType);
  }
  // Add int32 parameters (e.g., embedding indices)
  for (auto& param : int32_parameters) {
    auto buf = param.request();
    std::vector<int64_t> shape;
    for (ssize_t i = 0; i < buf.ndim; i++) {
      shape.emplace_back(buf.shape[i]);
    }
    auto memrefType = MemRefType::get(shape, builder.getI32Type());
    funcInputTypes.emplace_back(memrefType);
  }
  // Add float parameters (gamma, beta, weight, bias, table)
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
  irBuilder.parameterArgOffset = inputs.size(); // Parameters start after inputs
  for (size_t i = 0; i < inputs.size(); i++) {
    irBuilder.valueMap[inputs[i].get()] = entryBlock.getArgument(i);
  }

  // Compile computation graph
  Value resultValue = irBuilder.compileNode(output.node);
  Value outputArg = entryBlock.getArgument(inputs.size() + int32_parameters.size() + parameters.size());

  // Copy result to output
  builder.create<memref::CopyOp>(builder.getUnknownLoc(), resultValue, outputArg);
  builder.create<func::ReturnOp>(builder.getUnknownLoc());

  module.push_back(func);

  // Compile to native code
  void* fnPtr = compiler.compileAndGetFunctionPtr(module, "compute");
  if (!fnPtr) {
    throw std::runtime_error("Failed to compile function");
  }

  // Prepare arguments: inputs, int32_parameters, float_parameters, output
  std::vector<void*> args;
  for (auto& inp : inputs) {
    auto buf = inp->data.request();
    if (buf.ndim == 1) {
      marshal_memref_1d(args, inp->data);
    } else if (buf.ndim == 2) {
      marshal_memref_2d(args, inp->data);
    } else if (buf.ndim == 3) {
      marshal_memref_3d(args, inp->data);
    } else {
      throw std::runtime_error("Only 1D, 2D, and 3D inputs supported");
    }
  }
  // Marshal int32 parameters (e.g., embedding indices)
  for (auto& param : int32_parameters) {
    auto buf = param.request();
    if (buf.ndim == 1) {
      marshal_memref_1d_i32(args, param);
    } else {
      throw std::runtime_error("Only 1D int32 parameters supported");
    }
  }
  // Marshal float parameters
  for (auto& param : parameters) {
    auto buf = param.request();
    if (buf.ndim == 1) {
      marshal_memref_1d(args, param);
    } else if (buf.ndim == 2) {
      marshal_memref_2d(args, param);
    } else {
      throw std::runtime_error("Only 1D and 2D float parameters supported");
    }
  }

  // Allocate output
  py::array_t<float> result(output.node->shape);
  size_t ndim = output.node->shape.size();
  if (ndim == 1) {
    marshal_memref_1d(args, result);
  } else if (ndim == 2) {
    marshal_memref_2d(args, result);
  } else if (ndim == 3) {
    marshal_memref_3d(args, result);
  } else {
    throw std::runtime_error("Only 1D, 2D, and 3D outputs supported");
  }

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

PYBIND11_MODULE(ch14, m) {
  m.doc() = "Chapter 14: Optimized GPT (Linalg + Fusion + Vectorization)";

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

  // Chapter 13: GPT operations
  m.def("embedding", &embedding);
  m.def("create_causal_mask", &create_causal_mask);
  m.def("masked_softmax", &masked_softmax);
  m.def("rope", &rope);

  m.def("ffn", &ffn);
  m.def("scaled_dot_product_attention", &scaled_dot_product_attention);
  m.def("attention", &multi_head_attention);
  m.def("transformer_block", &transformer_block);
  m.def("multi_layer_transformer", &multi_layer_transformer);

  // Chapter 13: GPT compositions
  m.def("gpt_attention", &gpt_attention);
  m.def("gpt_attention_cached", &gpt_attention_cached,
        "Cached attention for incremental generation");
  m.def("gpt_block", &gpt_block);
  m.def("gpt_forward", &gpt_forward);

  m.def("forward", &forward,
        py::arg("output"),
        "JIT compile and execute computation graph");
}