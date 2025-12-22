#include "common.h"

#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

#include "llvm/Support/TargetSelect.h"

using namespace mlir;

// ============================================================================
// GELU Activation: y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
// ============================================================================
//
// GPU Concept: 1D grid of 1D blocks (like Phase 0)
// - Grid: (N+255)/256 blocks
// - Block: 256 threads
// - Each thread: one element
//

void buildGELUKernel(OpBuilder& builder, Location loc,
                     Value input, Value output, Value N) {
    // Constants
    Value c0 = ch15::createIndex(builder, loc, 0);
    Value c1 = ch15::createIndex(builder, loc, 1);
    Value c256 = ch15::createIndex(builder, loc, 256);
    Value c255 = ch15::createIndex(builder, loc, 255);

    // GELU constants
    Value c_half = ch15::createFloat(builder, loc, 0.5f);
    Value c_one = ch15::createFloat(builder, loc, 1.0f);
    Value c_sqrt_2_over_pi = ch15::createFloat(builder, loc, 0.7978845608f); // sqrt(2/pi)
    Value c_0_044715 = ch15::createFloat(builder, loc, 0.044715f);

    // Grid size: ceil(N/256)
    Value N_plus_255 = builder.create<arith::AddIOp>(loc, N, c255);
    Value numBlocks = builder.create<arith::DivUIOp>(loc, N_plus_255, c256);

    // Loop 1: Blocks
    auto blockLoop = builder.create<scf::ForOp>(loc, c0, numBlocks, c1);
    builder.setInsertionPointToStart(blockLoop.getBody());
    Value blockIdx = blockLoop.getInductionVar();

    // Loop 2: Threads within block
    auto threadLoop = builder.create<scf::ForOp>(loc, c0, c256, c1);
    builder.setInsertionPointToStart(threadLoop.getBody());
    Value threadIdx = threadLoop.getInductionVar();

    // Global index: blockIdx * 256 + threadIdx
    Value blockOffset = builder.create<arith::MulIOp>(loc, blockIdx, c256);
    Value globalIdx = builder.create<arith::AddIOp>(loc, blockOffset, threadIdx);

    // Bounds check
    Value inBounds = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalIdx, N
    );

    auto ifOp = builder.create<scf::IfOp>(loc, inBounds, /*withElseRegion=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    // Load input
    Value x = builder.create<memref::LoadOp>(loc, input, globalIdx);

    // Compute GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    // Step 1: x³
    Value x2 = builder.create<arith::MulFOp>(loc, x, x);
    Value x3 = builder.create<arith::MulFOp>(loc, x2, x);

    // Step 2: 0.044715 * x³
    Value term1 = builder.create<arith::MulFOp>(loc, c_0_044715, x3);

    // Step 3: x + 0.044715 * x³
    Value inner = builder.create<arith::AddFOp>(loc, x, term1);

    // Step 4: sqrt(2/π) * (x + 0.044715 * x³)
    Value scaled = builder.create<arith::MulFOp>(loc, c_sqrt_2_over_pi, inner);

    // Step 5: tanh(...)
    Value tanh_val = builder.create<math::TanhOp>(loc, scaled);

    // Step 6: 1 + tanh(...)
    Value one_plus_tanh = builder.create<arith::AddFOp>(loc, c_one, tanh_val);

    // Step 7: x * (1 + tanh(...))
    Value x_times = builder.create<arith::MulFOp>(loc, x, one_plus_tanh);

    // Step 8: 0.5 * x * (1 + tanh(...))
    Value result = builder.create<arith::MulFOp>(loc, c_half, x_times);

    // Store result
    builder.create<memref::StoreOp>(loc, result, output, globalIdx);
}

extern "C" void gelu_kernel(float* input, float* output, int N) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    MLIRContext* context = ch15::createContext();
    OpBuilder builder(context);
    auto loc = builder.getUnknownLoc();

    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // Create function: func @gelu(%input: memref<?xf32>, %output: memref<?xf32>, %N: index)
    auto f32Type = builder.getF32Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, f32Type);
    auto indexType = builder.getIndexType();

    auto funcType = builder.getFunctionType({memrefType, memrefType, indexType}, {});
    auto func = builder.create<func::FuncOp>(loc, "gelu", funcType);
    func.setPublic();

    Block* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    Value input_arg = entryBlock->getArgument(0);
    Value output_arg = entryBlock->getArgument(1);
    Value N_arg = entryBlock->getArgument(2);

    buildGELUKernel(builder, loc, input_arg, output_arg, N_arg);

    builder.setInsertionPointToEnd(entryBlock);
    builder.create<func::ReturnOp>(loc);

    if (failed(verify(module))) {
        llvm::errs() << "Module verification failed\n";
        module->dump();
        return;
    }

    if (failed(ch15::lowerToLLVMDialect(module))) {
        llvm::errs() << "Lowering failed\n";
        return;
    }

    llvm::LLVMContext llvmContext;
    auto llvmModule = ch15::translateToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Translation to LLVM IR failed\n";
        return;
    }

    ExecutionEngineOptions options;
    auto transformer = mlir::makeOptimizingTransformer(2, 0, nullptr);
    options.transformer = transformer;
    options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;

    auto engine = ExecutionEngine::create(module, options);
    if (!engine) {
        llvm::errs() << "Failed to create execution engine\n";
        return;
    }

    // Prepare arguments (1D memrefs: 5 fields each)
    int64_t N_val = static_cast<int64_t>(N);
    int64_t zero = 0;
    int64_t one = 1;

    void* args[] = {
        &input, &input, &zero, &N_val, &one,   // input: 5 args
        &output, &output, &zero, &N_val, &one, // output: 5 args
        &N_val                                   // N: 1 arg
    };

    auto err = (*engine)->invokePacked("gelu", llvm::MutableArrayRef<void*>(args, 11));
    if (err) {
        llvm::errs() << "Execution failed: " << err << "\n";
        return;
    }

    delete context;
}

// ============================================================================
// Element-wise Add: C = A + B
// ============================================================================

void buildAddKernel(OpBuilder& builder, Location loc,
                    Value A, Value B, Value C, Value N) {
    Value c0 = ch15::createIndex(builder, loc, 0);
    Value c1 = ch15::createIndex(builder, loc, 1);
    Value c256 = ch15::createIndex(builder, loc, 256);
    Value c255 = ch15::createIndex(builder, loc, 255);

    Value N_plus_255 = builder.create<arith::AddIOp>(loc, N, c255);
    Value numBlocks = builder.create<arith::DivUIOp>(loc, N_plus_255, c256);

    auto blockLoop = builder.create<scf::ForOp>(loc, c0, numBlocks, c1);
    builder.setInsertionPointToStart(blockLoop.getBody());
    Value blockIdx = blockLoop.getInductionVar();

    auto threadLoop = builder.create<scf::ForOp>(loc, c0, c256, c1);
    builder.setInsertionPointToStart(threadLoop.getBody());
    Value threadIdx = threadLoop.getInductionVar();

    Value blockOffset = builder.create<arith::MulIOp>(loc, blockIdx, c256);
    Value globalIdx = builder.create<arith::AddIOp>(loc, blockOffset, threadIdx);

    Value inBounds = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalIdx, N
    );

    auto ifOp = builder.create<scf::IfOp>(loc, inBounds, /*withElseRegion=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    Value a = builder.create<memref::LoadOp>(loc, A, globalIdx);
    Value b = builder.create<memref::LoadOp>(loc, B, globalIdx);
    Value sum = builder.create<arith::AddFOp>(loc, a, b);
    builder.create<memref::StoreOp>(loc, sum, C, globalIdx);
}

extern "C" void add_kernel(float* A, float* B, float* C, int N) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    MLIRContext* context = ch15::createContext();
    OpBuilder builder(context);
    auto loc = builder.getUnknownLoc();

    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    auto f32Type = builder.getF32Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, f32Type);
    auto indexType = builder.getIndexType();

    auto funcType = builder.getFunctionType(
        {memrefType, memrefType, memrefType, indexType}, {}
    );
    auto func = builder.create<func::FuncOp>(loc, "add", funcType);
    func.setPublic();

    Block* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    buildAddKernel(builder, loc,
                   entryBlock->getArgument(0),
                   entryBlock->getArgument(1),
                   entryBlock->getArgument(2),
                   entryBlock->getArgument(3));

    builder.setInsertionPointToEnd(entryBlock);
    builder.create<func::ReturnOp>(loc);

    if (failed(verify(module))) {
        llvm::errs() << "Module verification failed\n";
        module->dump();
        return;
    }

    if (failed(ch15::lowerToLLVMDialect(module))) {
        llvm::errs() << "Lowering failed\n";
        return;
    }

    llvm::LLVMContext llvmContext;
    auto llvmModule = ch15::translateToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Translation to LLVM IR failed\n";
        return;
    }

    ExecutionEngineOptions options;
    auto transformer = mlir::makeOptimizingTransformer(2, 0, nullptr);
    options.transformer = transformer;
    options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;

    auto engine = ExecutionEngine::create(module, options);
    if (!engine) {
        llvm::errs() << "Failed to create execution engine\n";
        return;
    }

    int64_t N_val = static_cast<int64_t>(N);
    int64_t zero = 0;
    int64_t one = 1;

    void* args[] = {
        &A, &A, &zero, &N_val, &one,
        &B, &B, &zero, &N_val, &one,
        &C, &C, &zero, &N_val, &one,
        &N_val
    };

    auto err = (*engine)->invokePacked("add", llvm::MutableArrayRef<void*>(args, 16));
    if (err) {
        llvm::errs() << "Execution failed: " << err << "\n";
        return;
    }

    delete context;
}

// ============================================================================
// Bias Add: output = input + bias (broadcast bias across all elements)
// ============================================================================

void buildBiasAddKernel(OpBuilder& builder, Location loc,
                        Value input, Value bias_val, Value output, Value N) {
    Value c0 = ch15::createIndex(builder, loc, 0);
    Value c1 = ch15::createIndex(builder, loc, 1);
    Value c256 = ch15::createIndex(builder, loc, 256);
    Value c255 = ch15::createIndex(builder, loc, 255);

    Value N_plus_255 = builder.create<arith::AddIOp>(loc, N, c255);
    Value numBlocks = builder.create<arith::DivUIOp>(loc, N_plus_255, c256);

    auto blockLoop = builder.create<scf::ForOp>(loc, c0, numBlocks, c1);
    builder.setInsertionPointToStart(blockLoop.getBody());
    Value blockIdx = blockLoop.getInductionVar();

    auto threadLoop = builder.create<scf::ForOp>(loc, c0, c256, c1);
    builder.setInsertionPointToStart(threadLoop.getBody());
    Value threadIdx = threadLoop.getInductionVar();

    Value blockOffset = builder.create<arith::MulIOp>(loc, blockIdx, c256);
    Value globalIdx = builder.create<arith::AddIOp>(loc, blockOffset, threadIdx);

    Value inBounds = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalIdx, N
    );

    auto ifOp = builder.create<scf::IfOp>(loc, inBounds, /*withElseRegion=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    Value x = builder.create<memref::LoadOp>(loc, input, globalIdx);
    Value result = builder.create<arith::AddFOp>(loc, x, bias_val);
    builder.create<memref::StoreOp>(loc, result, output, globalIdx);
}

extern "C" void bias_add_kernel(float* input, float bias, float* output, int N) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    MLIRContext* context = ch15::createContext();
    OpBuilder builder(context);
    auto loc = builder.getUnknownLoc();

    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    auto f32Type = builder.getF32Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, f32Type);
    auto indexType = builder.getIndexType();

    auto funcType = builder.getFunctionType(
        {memrefType, f32Type, memrefType, indexType}, {}
    );
    auto func = builder.create<func::FuncOp>(loc, "bias_add", funcType);
    func.setPublic();

    Block* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    buildBiasAddKernel(builder, loc,
                       entryBlock->getArgument(0),
                       entryBlock->getArgument(1),
                       entryBlock->getArgument(2),
                       entryBlock->getArgument(3));

    builder.setInsertionPointToEnd(entryBlock);
    builder.create<func::ReturnOp>(loc);

    if (failed(verify(module))) {
        llvm::errs() << "Module verification failed\n";
        module->dump();
        return;
    }

    if (failed(ch15::lowerToLLVMDialect(module))) {
        llvm::errs() << "Lowering failed\n";
        return;
    }

    llvm::LLVMContext llvmContext;
    auto llvmModule = ch15::translateToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Translation to LLVM IR failed\n";
        return;
    }

    ExecutionEngineOptions options;
    auto transformer = mlir::makeOptimizingTransformer(2, 0, nullptr);
    options.transformer = transformer;
    options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;

    auto engine = ExecutionEngine::create(module, options);
    if (!engine) {
        llvm::errs() << "Failed to create execution engine\n";
        return;
    }

    int64_t N_val = static_cast<int64_t>(N);
    int64_t zero = 0;
    int64_t one = 1;

    void* args[] = {
        &input, &input, &zero, &N_val, &one,  // input: 5 args
        &bias,                                  // bias: 1 arg (scalar f32)
        &output, &output, &zero, &N_val, &one, // output: 5 args
        &N_val                                  // N: 1 arg
    };

    auto err = (*engine)->invokePacked("bias_add", llvm::MutableArrayRef<void*>(args, 12));
    if (err) {
        llvm::errs() << "Execution failed: " << err << "\n";
        return;
    }

    delete context;
}