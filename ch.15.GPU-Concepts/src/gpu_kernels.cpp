//===- gpu_kernels.cpp - GPU Dialect Examples (Simplified) ---------------===//
//
// This file demonstrates GPU programming concepts using MLIR's GPU dialect.
// We generate GPU IR and dump it to show how real GPU programming works.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <string>

using namespace mlir;

// ============================================================================
// Helper: Convert Module to String
// ============================================================================

std::string moduleToString(ModuleOp module) {
    std::string str;
    llvm::raw_string_ostream os(str);
    module.print(os);
    return os.str();
}

// ============================================================================
// Example 1: Vector Addition - Basic 1D GPU Parallelism
// ============================================================================

std::string buildVectorAddGPU() {
    MLIRContext context;
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<gpu::GPUDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();

    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());

    // Types
    auto f32Type = builder.getF32Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, f32Type);
    auto indexType = builder.getIndexType();

    // Create GPU module
    auto gpuModule = builder.create<gpu::GPUModuleOp>(loc, "kernels");
    builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

    // Create GPU kernel function
    auto kernelFuncType = builder.getFunctionType(
        {memrefType, memrefType, memrefType, indexType}, {}
    );
    auto kernelFunc = builder.create<gpu::GPUFuncOp>(
        loc, "vector_add", kernelFuncType
    );
    kernelFunc->setAttr("gpu.kernel", builder.getUnitAttr());

    Block* kernelBlock = kernelFunc.addEntryBlock();
    builder.setInsertionPointToStart(kernelBlock);

    // Get kernel arguments
    Value A = kernelBlock->getArgument(0);
    Value B = kernelBlock->getArgument(1);
    Value C = kernelBlock->getArgument(2);
    Value N = kernelBlock->getArgument(3);

    // Create constants
    auto c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto c256 = builder.create<arith::ConstantIndexOp>(loc, 256);

    // Get thread indices
    auto threadIdx = builder.create<gpu::ThreadIdOp>(loc, indexType, gpu::Dimension::x);
    auto blockIdx = builder.create<gpu::BlockIdOp>(loc, indexType, gpu::Dimension::x);
    auto blockDim = builder.create<gpu::BlockDimOp>(loc, indexType, gpu::Dimension::x);

    // Compute global index: blockIdx * blockDim + threadIdx
    auto blockOffset = builder.create<arith::MulIOp>(loc, blockIdx, blockDim);
    auto globalIdx = builder.create<arith::AddIOp>(loc, blockOffset, threadIdx);

    // Bounds check: if (globalIdx < N)
    auto inBounds = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalIdx, N
    );

    auto ifOp = builder.create<scf::IfOp>(loc, inBounds, /*withElseRegion=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    // Vector addition: C[globalIdx] = A[globalIdx] + B[globalIdx]
    auto aVal = builder.create<memref::LoadOp>(loc, A, ValueRange{globalIdx});
    auto bVal = builder.create<memref::LoadOp>(loc, B, ValueRange{globalIdx});
    auto sum = builder.create<arith::AddFOp>(loc, aVal, bVal);
    builder.create<memref::StoreOp>(loc, sum, C, ValueRange{globalIdx});

    // Return from kernel
    builder.setInsertionPointToEnd(kernelBlock);
    builder.create<gpu::ReturnOp>(loc);

    // Create host function
    builder.setInsertionPointAfter(gpuModule);
    auto hostFuncType = builder.getFunctionType(
        {memrefType, memrefType, memrefType, indexType}, {}
    );
    auto hostFunc = builder.create<func::FuncOp>(loc, "main", hostFuncType);
    hostFunc.setPublic();

    Block* hostBlock = hostFunc.addEntryBlock();
    builder.setInsertionPointToStart(hostBlock);

    // Get host arguments
    Value hA = hostBlock->getArgument(0);
    Value hB = hostBlock->getArgument(1);
    Value hC = hostBlock->getArgument(2);
    Value hN = hostBlock->getArgument(3);

    // Calculate grid size: ceil(N / 256)
    auto c255 = builder.create<arith::ConstantIndexOp>(loc, 255);
    auto nPlus255 = builder.create<arith::AddIOp>(loc, hN, c255);
    auto numBlocks = builder.create<arith::DivUIOp>(loc, nPlus255, c256);

    // Launch kernel
    auto hc1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto hc256 = builder.create<arith::ConstantIndexOp>(loc, 256);

    auto kernelRef = SymbolRefAttr::get(&context, gpuModule.getName(),
                                         {SymbolRefAttr::get(&context, kernelFunc.getName())});

    gpu::KernelDim3 gridSize{numBlocks, hc1, hc1};
    gpu::KernelDim3 blockSize{hc256, hc1, hc1};

    builder.create<gpu::LaunchFuncOp>(
        loc, kernelRef, gridSize, blockSize,
        /*dynamicSharedMemorySize=*/Value(),
        /*kernelOperands=*/ValueRange{hA, hB, hC, hN}
    );

    builder.create<func::ReturnOp>(loc);

    return moduleToString(module);
}

// ============================================================================
// Example 2: Matrix Multiplication - 2D Parallelism + Shared Memory
// ============================================================================

std::string buildMatMulGPU() {
    MLIRContext context;
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<gpu::GPUDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();

    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());

    // Types
    auto f32Type = builder.getF32Type();
    auto matrixType = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32Type);
    auto indexType = builder.getIndexType();

    // Create GPU module
    auto gpuModule = builder.create<gpu::GPUModuleOp>(loc, "kernels");
    builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

    // Create GPU kernel function
    auto kernelFuncType = builder.getFunctionType(
        {matrixType, matrixType, matrixType}, {}
    );
    auto kernelFunc = builder.create<gpu::GPUFuncOp>(
        loc, "matmul", kernelFuncType
    );
    kernelFunc->setAttr("gpu.kernel", builder.getUnitAttr());

    Block* kernelBlock = kernelFunc.addEntryBlock();
    builder.setInsertionPointToStart(kernelBlock);

    // Get kernel arguments
    Value A = kernelBlock->getArgument(0);
    Value B = kernelBlock->getArgument(1);
    Value C = kernelBlock->getArgument(2);

    // Get 2D thread indices
    auto threadX = builder.create<gpu::ThreadIdOp>(loc, indexType, gpu::Dimension::x);
    auto threadY = builder.create<gpu::ThreadIdOp>(loc, indexType, gpu::Dimension::y);
    auto blockX = builder.create<gpu::BlockIdOp>(loc, indexType, gpu::Dimension::x);
    auto blockY = builder.create<gpu::BlockIdOp>(loc, indexType, gpu::Dimension::y);

    // Constants
    auto c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto c16 = builder.create<arith::ConstantIndexOp>(loc, 16);

    // Compute global row/col
    auto blockOffsetY = builder.create<arith::MulIOp>(loc, blockY, c16);
    auto row = builder.create<arith::AddIOp>(loc, blockOffsetY, threadY);

    auto blockOffsetX = builder.create<arith::MulIOp>(loc, blockX, c16);
    auto col = builder.create<arith::AddIOp>(loc, blockOffsetX, threadX);

    // Allocate shared memory (16x16 tiles) - use address space 3 (workgroup)
    auto addrSpace = builder.getI64IntegerAttr(static_cast<int64_t>(gpu::AddressSpace::Workgroup));
    auto tileType = MemRefType::get({16, 16}, f32Type, MemRefLayoutAttrInterface{}, addrSpace);

    auto tileA = builder.create<gpu::AllocOp>(loc, tileType, /*asyncToken=*/nullptr, ValueRange{}, ValueRange{}, ValueRange{}, false).getResult(0);
    auto tileB = builder.create<gpu::AllocOp>(loc, tileType, /*asyncToken=*/nullptr, ValueRange{}, ValueRange{}, ValueRange{}, false).getResult(0);

    // Initialize accumulator
    auto zeroAttr = builder.getFloatAttr(f32Type, 0.0);
    auto zero = builder.create<arith::ConstantOp>(loc, zeroAttr);

    // Simplified: just show structure with barrier
    // Load tile collaboratively
    auto aVal = builder.create<memref::LoadOp>(loc, A, ValueRange{row, col});
    builder.create<memref::StoreOp>(loc, aVal, tileA, ValueRange{threadY, threadX});

    // Barrier: wait for all threads to finish loading
    builder.create<gpu::BarrierOp>(loc);

    // Use shared memory value
    auto tileVal = builder.create<memref::LoadOp>(loc, tileA, ValueRange{threadY, threadX});

    // Store result (simplified)
    builder.create<memref::StoreOp>(loc, tileVal, C, ValueRange{row, col});

    // Another barrier before next iteration
    builder.create<gpu::BarrierOp>(loc);

    // Return from kernel
    builder.create<gpu::ReturnOp>(loc);

    // Host function (simplified)
    builder.setInsertionPointAfter(gpuModule);
    auto hostFuncType = builder.getFunctionType({matrixType, matrixType, matrixType}, {});
    auto hostFunc = builder.create<func::FuncOp>(loc, "main", hostFuncType);
    hostFunc.setPublic();

    Block* hostBlock = hostFunc.addEntryBlock();
    builder.setInsertionPointToStart(hostBlock);

    auto hc1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto hc16 = builder.create<arith::ConstantIndexOp>(loc, 16);
    auto hc64 = builder.create<arith::ConstantIndexOp>(loc, 64);  // Example: 64x64 blocks for 1024x1024 matrix

    auto kernelRef = SymbolRefAttr::get(&context, gpuModule.getName(),
                                         {SymbolRefAttr::get(&context, kernelFunc.getName())});

    gpu::KernelDim3 gridSize{hc64, hc64, hc1};
    gpu::KernelDim3 blockSize{hc16, hc16, hc1};

    builder.create<gpu::LaunchFuncOp>(
        loc, kernelRef, gridSize, blockSize,
        /*dynamicSharedMemorySize=*/Value(),
        /*kernelOperands=*/ValueRange{hostBlock->getArgument(0), hostBlock->getArgument(1), hostBlock->getArgument(2)}
    );

    builder.create<func::ReturnOp>(loc);

    return moduleToString(module);
}

// ============================================================================
// Example 3: Softmax - Reductions + Multiple Barriers
// ============================================================================

std::string buildSoftmaxGPU() {
    MLIRContext context;
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<gpu::GPUDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();
    context.getOrLoadDialect<math::MathDialect>();

    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());

    // Types
    auto f32Type = builder.getF32Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, f32Type);
    auto indexType = builder.getIndexType();

    // Create GPU module
    auto gpuModule = builder.create<gpu::GPUModuleOp>(loc, "kernels");
    builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

    // Create GPU kernel function
    auto kernelFuncType = builder.getFunctionType(
        {memrefType, memrefType, indexType}, {}
    );
    auto kernelFunc = builder.create<gpu::GPUFuncOp>(
        loc, "softmax", kernelFuncType
    );
    kernelFunc->setAttr("gpu.kernel", builder.getUnitAttr());

    Block* kernelBlock = kernelFunc.addEntryBlock();
    builder.setInsertionPointToStart(kernelBlock);

    // Get kernel arguments
    Value input = kernelBlock->getArgument(0);
    Value output = kernelBlock->getArgument(1);
    Value N = kernelBlock->getArgument(2);

    // Get thread index
    auto threadIdx = builder.create<gpu::ThreadIdOp>(loc, indexType, gpu::Dimension::x);

    // Allocate shared memory for reduction
    auto addrSpace = builder.getI64IntegerAttr(static_cast<int64_t>(gpu::AddressSpace::Workgroup));
    auto sharedType = MemRefType::get({256}, f32Type, MemRefLayoutAttrInterface{}, addrSpace);
    auto sharedMax = builder.create<gpu::AllocOp>(loc, sharedType, /*asyncToken=*/nullptr, ValueRange{}, ValueRange{}, ValueRange{}, false).getResult(0);
    auto sharedSum = builder.create<gpu::AllocOp>(loc, sharedType, /*asyncToken=*/nullptr, ValueRange{}, ValueRange{}, ValueRange{}, false).getResult(0);

    // Stage 1: Find local maximum (simplified - each thread just loads one value)
    auto c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto inBounds = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, threadIdx, N
    );

    auto ifOp1 = builder.create<scf::IfOp>(loc, inBounds, false);
    builder.setInsertionPointToStart(&ifOp1.getThenRegion().front());

    auto val = builder.create<memref::LoadOp>(loc, input, ValueRange{threadIdx});
    builder.create<memref::StoreOp>(loc, val, sharedMax, ValueRange{threadIdx});

    // BARRIER 1: Wait for all threads to compute local max
    builder.setInsertionPointToEnd(kernelBlock);
    builder.create<gpu::BarrierOp>(loc);

    // Stage 2: Compute exp(x - max) (simplified)
    auto maxVal = builder.create<memref::LoadOp>(loc, sharedMax, ValueRange{c0});

    auto ifOp2 = builder.create<scf::IfOp>(loc, inBounds, false);
    builder.setInsertionPointToStart(&ifOp2.getThenRegion().front());

    auto val2 = builder.create<memref::LoadOp>(loc, input, ValueRange{threadIdx});
    auto shifted = builder.create<arith::SubFOp>(loc, val2, maxVal);
    auto expVal = builder.create<math::ExpOp>(loc, shifted);
    builder.create<memref::StoreOp>(loc, expVal, sharedSum, ValueRange{threadIdx});

    // BARRIER 2: Wait for all threads to compute exp values
    builder.setInsertionPointToEnd(kernelBlock);
    builder.create<gpu::BarrierOp>(loc);

    // Stage 3: Normalize (simplified)
    auto sumVal = builder.create<memref::LoadOp>(loc, sharedSum, ValueRange{c0});

    auto ifOp3 = builder.create<scf::IfOp>(loc, inBounds, false);
    builder.setInsertionPointToStart(&ifOp3.getThenRegion().front());

    auto expVal3 = builder.create<memref::LoadOp>(loc, sharedSum, ValueRange{threadIdx});
    auto normalized = builder.create<arith::DivFOp>(loc, expVal3, sumVal);
    builder.create<memref::StoreOp>(loc, normalized, output, ValueRange{threadIdx});

    // BARRIER 3: Final synchronization
    builder.setInsertionPointToEnd(kernelBlock);
    builder.create<gpu::BarrierOp>(loc);

    // Return from kernel
    builder.create<gpu::ReturnOp>(loc);

    // Host function (simplified)
    builder.setInsertionPointAfter(gpuModule);
    auto hostFuncType = builder.getFunctionType({memrefType, memrefType, indexType}, {});
    auto hostFunc = builder.create<func::FuncOp>(loc, "main", hostFuncType);
    hostFunc.setPublic();

    Block* hostBlock = hostFunc.addEntryBlock();
    builder.setInsertionPointToStart(hostBlock);

    auto hc1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto hc256 = builder.create<arith::ConstantIndexOp>(loc, 256);

    auto kernelRef = SymbolRefAttr::get(&context, gpuModule.getName(),
                                         {SymbolRefAttr::get(&context, kernelFunc.getName())});

    gpu::KernelDim3 gridSize{hc1, hc1, hc1};
    gpu::KernelDim3 blockSize{hc256, hc1, hc1};

    builder.create<gpu::LaunchFuncOp>(
        loc, kernelRef, gridSize, blockSize,
        /*dynamicSharedMemorySize=*/Value(),
        /*kernelOperands=*/ValueRange{hostBlock->getArgument(0), hostBlock->getArgument(1), hostBlock->getArgument(2)}
    );

    builder.create<func::ReturnOp>(loc);

    return moduleToString(module);
}