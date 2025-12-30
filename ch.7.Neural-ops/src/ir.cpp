#include "graph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include <limits>

using namespace mlir;

// GraphOperation constructor implementation
GraphOperation::GraphOperation(OpType t, std::vector<int> ins, std::vector<int64_t> sh, int i)
    : type(t), inputs(std::move(ins)), shape(std::move(sh)), id(i) {}

// ComputationGraph constructor implementation
ComputationGraph::ComputationGraph(MLIRContext* ctx) : context(ctx), nextId(0) {
    context->getOrLoadDialect<func::FuncDialect>();
    context->getOrLoadDialect<arith::ArithDialect>();
    context->getOrLoadDialect<tensor::TensorDialect>();
    context->getOrLoadDialect<linalg::LinalgDialect>();
    context->getOrLoadDialect<math::MathDialect>();
}

// Add a variable/tensor placeholder
int ComputationGraph::addVariable(const std::vector<int64_t>& shape) {
    int id = nextId++;
    operations.emplace_back(GraphOperation::OpType::Input, std::vector<int>{}, shape, id);
    return id;
}

// Add element-wise addition
int ComputationGraph::add(int lhs, int rhs) {
    const auto& lhsShape = operations[lhs].shape;
    int id = nextId++;
    operations.emplace_back(GraphOperation::OpType::Add, std::vector<int>{lhs, rhs}, lhsShape, id);
    return id;
}

// Add element-wise multiplication
int ComputationGraph::mul(int lhs, int rhs) {
    const auto& lhsShape = operations[lhs].shape;
    int id = nextId++;
    operations.emplace_back(GraphOperation::OpType::Mul, std::vector<int>{lhs, rhs}, lhsShape, id);
    return id;
}

// Add matrix multiplication
int ComputationGraph::matmul(int lhs, int rhs) {
    const auto& lhsShape = operations[lhs].shape;
    const auto& rhsShape = operations[rhs].shape;
    std::vector<int64_t> resultShape = {lhsShape[0], rhsShape[1]};
    int id = nextId++;
    operations.emplace_back(GraphOperation::OpType::MatMul, std::vector<int>{lhs, rhs}, resultShape, id);
    return id;
}

// Add ReLU activation
int ComputationGraph::relu(int input) {
    const auto& inputShape = operations[input].shape;
    int id = nextId++;
    operations.emplace_back(GraphOperation::OpType::ReLU, std::vector<int>{input}, inputShape, id);
    return id;
}

// Add Softmax activation
int ComputationGraph::softmax(int input) {
    const auto& inputShape = operations[input].shape;
    int id = nextId++;
    operations.emplace_back(GraphOperation::OpType::Softmax, std::vector<int>{input}, inputShape, id);
    return id;
}

// Generate MLIR module from the computation graph
ModuleOp ComputationGraph::generateMLIR(int outputId, const std::string& funcName) {
    OpBuilder builder(context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module.getBody());

    // Collect all input operations
    std::vector<int> inputIds;
    for (const auto& op : operations) {
        if (op.type == GraphOperation::OpType::Input) {
            inputIds.push_back(op.id);
        }
    }

    // Build function signature with tensors
    std::vector<Type> inputTypes;
    for (int inputId : inputIds) {
        const auto& shape = operations[inputId].shape;
        auto elemType = builder.getF32Type();
        SmallVector<int64_t> mlirShape(shape.begin(), shape.end());
        auto tensorType = RankedTensorType::get(mlirShape, elemType);
        inputTypes.push_back(tensorType);
    }

    // Output type
    const auto& outputShape = operations[outputId].shape;
    SmallVector<int64_t> mlirOutputShape(outputShape.begin(), outputShape.end());
    auto outputType = RankedTensorType::get(mlirOutputShape, builder.getF32Type());

    auto funcType = builder.getFunctionType(inputTypes, {outputType});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), funcName, funcType);
    auto& entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    // Map input IDs to function arguments
    std::unordered_map<int, Value> valueMap;
    for (size_t i = 0; i < inputIds.size(); ++i) {
        valueMap[inputIds[i]] = entryBlock.getArgument(i);
    }

    // Build the computation graph
    std::vector<Value> funcArgs(entryBlock.args_begin(), entryBlock.args_end());
    Value result = buildOperation(builder, builder.getUnknownLoc(), outputId, valueMap, funcArgs);

    // Return result tensor
    builder.create<func::ReturnOp>(builder.getUnknownLoc(), result);
    return module;
}

// Recursively build operations
Value ComputationGraph::buildOperation(OpBuilder& builder, Location loc, int opId,
                                      std::unordered_map<int, Value>& valueMap,
                                      const std::vector<Value>& funcArgs) {
    // Check if already computed
    if (valueMap.count(opId)) {
        return valueMap[opId];
    }

    const auto& op = operations[opId];
    Value result;

    switch (op.type) {
        case GraphOperation::OpType::Input:
            // Should already be in valueMap
            return valueMap[opId];

        case GraphOperation::OpType::Add:
        case GraphOperation::OpType::Mul: {
            Value lhs = buildOperation(builder, loc, op.inputs[0], valueMap, funcArgs);
            Value rhs = buildOperation(builder, loc, op.inputs[1], valueMap, funcArgs);
            result = buildElementWiseOp(builder, loc, lhs, rhs, op.type);
            break;
        }

        case GraphOperation::OpType::MatMul: {
            Value lhs = buildOperation(builder, loc, op.inputs[0], valueMap, funcArgs);
            Value rhs = buildOperation(builder, loc, op.inputs[1], valueMap, funcArgs);
            result = buildMatMul(builder, loc, lhs, rhs);
            break;
        }

        case GraphOperation::OpType::ReLU: {
            Value input = buildOperation(builder, loc, op.inputs[0], valueMap, funcArgs);
            result = buildReLU(builder, loc, input);
            break;
        }

        case GraphOperation::OpType::Softmax: {
            Value input = buildOperation(builder, loc, op.inputs[0], valueMap, funcArgs);
            result = buildSoftmax(builder, loc, input);
            break;
        }
    }

    valueMap[opId] = result;
    return result;
}

// Build element-wise operations using linalg.generic
Value ComputationGraph::buildElementWiseOp(OpBuilder& builder, Location loc,
                                          Value lhs, Value rhs,
                                          GraphOperation::OpType opType) {
    auto lhsType = mlir::cast<RankedTensorType>(lhs.getType());
    auto rank = lhsType.getRank();

    // Create empty tensor for result
    SmallVector<OpFoldResult> dynSizes;
    for (int i = 0; i < rank; ++i) {
        dynSizes.push_back(builder.create<tensor::DimOp>(loc, lhs, i).getResult());
    }
    Value empty = builder.create<tensor::EmptyOp>(loc, dynSizes, lhsType.getElementType());

    // Create affine maps for indexing
    auto identityMap = builder.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap> indexingMaps(3, identityMap); // lhs, rhs, out
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    // Build linalg.generic operation
    auto genericOp = builder.create<linalg::GenericOp>(
        loc, lhsType, ValueRange{lhs, rhs}, ValueRange{empty},
        indexingMaps, iteratorTypes,
        [&](OpBuilder& b, Location l, ValueRange args) {
            Value result;
            if (opType == GraphOperation::OpType::Add) {
                result = b.create<arith::AddFOp>(l, args[0], args[1]);
            } else { // Mul
                result = b.create<arith::MulFOp>(l, args[0], args[1]);
            }
            b.create<linalg::YieldOp>(l, result);
        }
    );

    return genericOp.getResult(0);
}

// Build matrix multiplication using linalg.matmul
Value ComputationGraph::buildMatMul(OpBuilder& builder, Location loc, Value lhs, Value rhs) {
    auto lhsType = mlir::cast<RankedTensorType>(lhs.getType());
    auto rhsType = mlir::cast<RankedTensorType>(rhs.getType());

    // Get dimensions
    Value m = builder.create<tensor::DimOp>(loc, lhs, 0);
    Value n = builder.create<tensor::DimOp>(loc, rhs, 1);

    // Determine result type shape - use static if both inputs are static
    SmallVector<int64_t> resultShape;
    if (lhsType.isDynamicDim(0)) {
        resultShape.push_back(ShapedType::kDynamic);
    } else {
        resultShape.push_back(lhsType.getDimSize(0));
    }
    if (rhsType.isDynamicDim(1)) {
        resultShape.push_back(ShapedType::kDynamic);
    } else {
        resultShape.push_back(rhsType.getDimSize(1));
    }
    
    auto resultType = RankedTensorType::get(resultShape, builder.getF32Type());

    // Create empty output tensor with potentially static sizes
    SmallVector<OpFoldResult> outputSizes;
    outputSizes.push_back(lhsType.isDynamicDim(0) ? 
        OpFoldResult(m.getDefiningOp()->getResult(0)) :
        builder.getIndexAttr(lhsType.getDimSize(0)));
    outputSizes.push_back(rhsType.isDynamicDim(1) ? 
        OpFoldResult(n.getDefiningOp()->getResult(0)) :
        builder.getIndexAttr(rhsType.getDimSize(1)));
    
    Value empty = builder.create<tensor::EmptyOp>(loc, outputSizes, builder.getF32Type());

    // Initialize to zero
    Value zero = builder.create<arith::ConstantOp>(loc, builder.getF32Type(),
                                                   builder.getF32FloatAttr(0.0));
    Value init = builder.create<linalg::FillOp>(loc, zero, empty).getResult(0);

    // Perform matrix multiplication using linalg.matmul
    auto matmulOp = builder.create<linalg::MatmulOp>(
        loc, resultType,
        ValueRange{lhs, rhs},
        ValueRange{init}
    );

    return matmulOp.getResult(0);
}

// Build ReLU activation using linalg.generic
Value ComputationGraph::buildReLU(OpBuilder& builder, Location loc, Value input) {
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto rank = inputType.getRank();

    // Create empty tensor for result
    SmallVector<OpFoldResult> dynSizes;
    for (int i = 0; i < rank; ++i) {
        dynSizes.push_back(builder.create<tensor::DimOp>(loc, input, i).getResult());
    }
    Value empty = builder.create<tensor::EmptyOp>(loc, dynSizes, inputType.getElementType());

    // Create constant zero
    Value zero = builder.create<arith::ConstantOp>(loc, builder.getF32Type(),
                                                   builder.getF32FloatAttr(0.0));

    // Create affine maps
    auto identityMap = builder.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    // Build linalg.generic for ReLU: max(x, 0)
    auto genericOp = builder.create<linalg::GenericOp>(
        loc, inputType, ValueRange{input}, ValueRange{empty},
        indexingMaps, iteratorTypes,
        [&](OpBuilder& b, Location l, ValueRange args) {
            Value maxVal = b.create<arith::MaximumFOp>(l, args[0], zero);
            b.create<linalg::YieldOp>(l, maxVal);
        }
    );

    return genericOp.getResult(0);
}

// Build Softmax using linalg.reduce and linalg.generic (from Chapter 6)
Value ComputationGraph::buildSoftmax(OpBuilder& builder, Location loc, Value input) {
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto f32Type = builder.getF32Type();
    auto scalarTensorType = RankedTensorType::get({}, f32Type);

    // Pass 1: Find maximum using linalg.reduce
    Value negInf = builder.create<arith::ConstantOp>(
        loc, builder.getFloatAttr(f32Type,
            APFloat::getInf(f32Type.getFloatSemantics(), /*Negative=*/true)));
    Value initMaxTensor = builder.create<tensor::FromElementsOp>(
        loc, scalarTensorType, ValueRange{negInf});

    auto reduceMaxOp = builder.create<linalg::ReduceOp>(
        loc, input, initMaxTensor,
        SmallVector<int64_t>{0},
        [&](OpBuilder& b, Location l, ValueRange args) {
            Value newMax = b.create<arith::MaximumFOp>(l, args[0], args[1]);
            b.create<linalg::YieldOp>(l, newMax);
        }
    );
    Value maxTensor = reduceMaxOp.getResult(0);
    Value maxVal = builder.create<tensor::ExtractOp>(loc, maxTensor, ValueRange{});

    // Pass 2: Compute exp(x - max) using linalg.generic
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<OpFoldResult> dynamicSizes = {
        builder.create<tensor::DimOp>(loc, input, c0).getResult()
    };
    Value emptyTensor = builder.create<tensor::EmptyOp>(loc, dynamicSizes, f32Type);

    auto identityMap = builder.getMultiDimIdentityMap(1);
    SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
    SmallVector<utils::IteratorType> iteratorTypes = {utils::IteratorType::parallel};

    auto expOp = builder.create<linalg::GenericOp>(
        loc, inputType, input, emptyTensor,
        indexingMaps, iteratorTypes,
        [&](OpBuilder& b, Location l, ValueRange args) {
            Value shifted = b.create<arith::SubFOp>(l, args[0], maxVal);
            Value expVal = b.create<math::ExpOp>(l, shifted);
            b.create<linalg::YieldOp>(l, expVal);
        }
    );
    Value expTensor = expOp.getResult(0);

    // Pass 3: Sum exp values using linalg.reduce
    Value zeroFloat = builder.create<arith::ConstantOp>(
        loc, builder.getFloatAttr(f32Type, APFloat(0.0f)));
    Value initSumTensor = builder.create<tensor::FromElementsOp>(
        loc, scalarTensorType, ValueRange{zeroFloat});

    auto reduceSumOp = builder.create<linalg::ReduceOp>(
        loc, expTensor, initSumTensor,
        SmallVector<int64_t>{0},
        [&](OpBuilder& b, Location l, ValueRange args) {
            Value newSum = b.create<arith::AddFOp>(l, args[0], args[1]);
            b.create<linalg::YieldOp>(l, newSum);
        }
    );
    Value sumTensor = reduceSumOp.getResult(0);
    Value sumVal = builder.create<tensor::ExtractOp>(loc, sumTensor, ValueRange{});

    // Pass 4: Normalize using linalg.generic
    Value outputEmpty = builder.create<tensor::EmptyOp>(loc, dynamicSizes, f32Type);

    auto normalizeOp = builder.create<linalg::GenericOp>(
        loc, inputType, expTensor, outputEmpty,
        indexingMaps, iteratorTypes,
        [&](OpBuilder& b, Location l, ValueRange args) {
            Value normalized = b.create<arith::DivFOp>(l, args[0], sumVal);
            b.create<linalg::YieldOp>(l, normalized);
        }
    );

    return normalizeOp.getResult(0);
}