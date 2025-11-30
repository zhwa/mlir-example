#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

using namespace mlir;

// Forward declaration
class ComputationGraph;

// Represents a symbolic operation result in the computation graph
struct GraphOperation {
    enum class OpType {
        Input,      // Input placeholder
        Add,        // Element-wise addition
        Mul,        // Element-wise multiplication
        MatMul,     // Matrix multiplication
        ReLU,       // Rectified Linear Unit
        Softmax     // Softmax activation
    };

    OpType type;
    std::vector<int> inputs;  // Indices of input operations
    std::vector<int64_t> shape;  // Shape of the result
    int id;  // Unique identifier

    GraphOperation(OpType t, std::vector<int> ins, std::vector<int64_t> sh, int i)
        : type(t), inputs(std::move(ins)), shape(std::move(sh)), id(i) {}
};

// Computation graph that tracks operations and generates MLIR
class ComputationGraph {
public:
    ComputationGraph(MLIRContext* ctx) : context(ctx), nextId(0) {
        context->getOrLoadDialect<func::FuncDialect>();
        context->getOrLoadDialect<arith::ArithDialect>();
        context->getOrLoadDialect<memref::MemRefDialect>();
        context->getOrLoadDialect<scf::SCFDialect>();
        context->getOrLoadDialect<math::MathDialect>();
        context->getOrLoadDialect<linalg::LinalgDialect>();
    }

    // Add an input placeholder
    int addInput(std::vector<int64_t> shape) {
        int id = nextId++;
        operations.emplace_back(GraphOperation::OpType::Input, std::vector<int>{}, std::move(shape), id);
        return id;
    }

    // Add element-wise addition
    int add(int lhs, int rhs) {
        const auto& lhsShape = operations[lhs].shape;
        int id = nextId++;
        operations.emplace_back(GraphOperation::OpType::Add, std::vector<int>{lhs, rhs}, lhsShape, id);
        return id;
    }

    // Add element-wise multiplication
    int mul(int lhs, int rhs) {
        const auto& lhsShape = operations[lhs].shape;
        int id = nextId++;
        operations.emplace_back(GraphOperation::OpType::Mul, std::vector<int>{lhs, rhs}, lhsShape, id);
        return id;
    }

    // Add matrix multiplication
    int matmul(int lhs, int rhs) {
        const auto& lhsShape = operations[lhs].shape;
        const auto& rhsShape = operations[rhs].shape;
        std::vector<int64_t> resultShape = {lhsShape[0], rhsShape[1]};
        int id = nextId++;
        operations.emplace_back(GraphOperation::OpType::MatMul, std::vector<int>{lhs, rhs}, resultShape, id);
        return id;
    }

    // Add ReLU activation
    int relu(int input) {
        const auto& inputShape = operations[input].shape;
        int id = nextId++;
        operations.emplace_back(GraphOperation::OpType::ReLU, std::vector<int>{input}, inputShape, id);
        return id;
    }

    // Add Softmax activation
    int softmax(int input) {
        const auto& inputShape = operations[input].shape;
        int id = nextId++;
        operations.emplace_back(GraphOperation::OpType::Softmax, std::vector<int>{input}, inputShape, id);
        return id;
    }

    // Generate MLIR module from the computation graph
    ModuleOp generateMLIR(int outputId, const std::string& funcName) {
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

        // Build function signature
        std::vector<Type> inputTypes;
        for (int inputId : inputIds) {
            const auto& shape = operations[inputId].shape;
            auto elemType = builder.getF32Type();
            SmallVector<int64_t> mlirShape(shape.begin(), shape.end());
            auto memrefType = MemRefType::get(mlirShape, elemType);
            inputTypes.push_back(memrefType);
        }

        // Output type
        const auto& outputShape = operations[outputId].shape;
        SmallVector<int64_t> mlirOutputShape(outputShape.begin(), outputShape.end());
        auto outputType = MemRefType::get(mlirOutputShape, builder.getF32Type());
        inputTypes.push_back(outputType);

        auto funcType = builder.getFunctionType(inputTypes, {});
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

        // Copy result to output buffer
        Value outputBuffer = entryBlock.getArgument(inputIds.size());

        // Create nested loops to copy data
        const auto& shape = operations[outputId].shape;
        if (shape.size() == 1) {
            // 1D case
            auto dim0 = builder.create<memref::DimOp>(builder.getUnknownLoc(), result, 0);
            auto zero = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 0);
            auto one = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);

            builder.create<scf::ForOp>(builder.getUnknownLoc(), zero, dim0, one,
                ValueRange{}, [&](OpBuilder& b, Location loc, Value i, ValueRange) {
                    auto val = b.create<memref::LoadOp>(loc, result, ValueRange{i});
                    b.create<memref::StoreOp>(loc, val, outputBuffer, ValueRange{i});
                    b.create<scf::YieldOp>(loc);
                });
        } else if (shape.size() == 2) {
            // 2D case
            auto dim0 = builder.create<memref::DimOp>(builder.getUnknownLoc(), result, 0);
            auto dim1 = builder.create<memref::DimOp>(builder.getUnknownLoc(), result, 1);
            auto zero = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 0);
            auto one = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);

            builder.create<scf::ForOp>(builder.getUnknownLoc(), zero, dim0, one,
                ValueRange{}, [&](OpBuilder& b, Location loc, Value i, ValueRange) {
                    b.create<scf::ForOp>(loc, zero, dim1, one,
                        ValueRange{}, [&](OpBuilder& b2, Location loc2, Value j, ValueRange) {
                            auto val = b2.create<memref::LoadOp>(loc2, result, ValueRange{i, j});
                            b2.create<memref::StoreOp>(loc2, val, outputBuffer, ValueRange{i, j});
                            b2.create<scf::YieldOp>(loc2);
                        });
                    b.create<scf::YieldOp>(loc);
                });
        }

        builder.create<func::ReturnOp>(builder.getUnknownLoc());
        return module;
    }

private:
    MLIRContext* context;
    std::vector<GraphOperation> operations;
    int nextId;

    // Helper to build operations recursively
    Value buildOperation(OpBuilder& builder, Location loc, int opId,
                        std::unordered_map<int, Value>& valueMap,
                        const std::vector<Value>& funcArgs);

    // Helper to build element-wise operations
    Value buildElementWiseOp(OpBuilder& builder, Location loc,
                            Value lhs, Value rhs,
                            GraphOperation::OpType opType);

    // Helper to build matrix multiplication
    Value buildMatMul(OpBuilder& builder, Location loc, Value lhs, Value rhs);

    // Helper to build ReLU
    Value buildReLU(OpBuilder& builder, Location loc, Value input);

    // Helper to build Softmax (reusing Chapter 6 implementation)
    Value buildSoftmax(OpBuilder& builder, Location loc, Value input);
};

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

// Build element-wise operations
Value ComputationGraph::buildElementWiseOp(OpBuilder& builder, Location loc,
                                          Value lhs, Value rhs,
                                          GraphOperation::OpType opType) {
    auto lhsType = lhs.getType().cast<MemRefType>();
    auto resultType = lhsType;

    // Allocate result buffer
    SmallVector<Value> dynSizes;
    for (int i = 0; i < lhsType.getRank(); ++i) {
        if (lhsType.isDynamicDim(i)) {
            dynSizes.push_back(builder.create<memref::DimOp>(loc, lhs, i));
        }
    }
    Value result = builder.create<memref::AllocOp>(loc, resultType, dynSizes);

    // Build nested loops
    if (lhsType.getRank() == 1) {
        auto dim0 = builder.create<memref::DimOp>(loc, lhs, 0);
        auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        auto one = builder.create<arith::ConstantIndexOp>(loc, 1);

        builder.create<scf::ForOp>(loc, zero, dim0, one,
            ValueRange{}, [&](OpBuilder& b, Location l, Value i, ValueRange) {
                auto lVal = b.create<memref::LoadOp>(l, lhs, ValueRange{i});
                auto rVal = b.create<memref::LoadOp>(l, rhs, ValueRange{i});
                Value resVal;
                if (opType == GraphOperation::OpType::Add) {
                    resVal = b.create<arith::AddFOp>(l, lVal, rVal);
                } else {
                    resVal = b.create<arith::MulFOp>(l, lVal, rVal);
                }
                b.create<memref::StoreOp>(l, resVal, result, ValueRange{i});
                b.create<scf::YieldOp>(l);
            });
    } else if (lhsType.getRank() == 2) {
        auto dim0 = builder.create<memref::DimOp>(loc, lhs, 0);
        auto dim1 = builder.create<memref::DimOp>(loc, lhs, 1);
        auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        auto one = builder.create<arith::ConstantIndexOp>(loc, 1);

        builder.create<scf::ForOp>(loc, zero, dim0, one,
            ValueRange{}, [&](OpBuilder& b, Location l, Value i, ValueRange) {
                b.create<scf::ForOp>(l, zero, dim1, one,
                    ValueRange{}, [&](OpBuilder& b2, Location l2, Value j, ValueRange) {
                        auto lVal = b2.create<memref::LoadOp>(l2, lhs, ValueRange{i, j});
                        auto rVal = b2.create<memref::LoadOp>(l2, rhs, ValueRange{i, j});
                        Value resVal;
                        if (opType == GraphOperation::OpType::Add) {
                            resVal = b2.create<arith::AddFOp>(l2, lVal, rVal);
                        } else {
                            resVal = b2.create<arith::MulFOp>(l2, lVal, rVal);
                        }
                        b2.create<memref::StoreOp>(l2, resVal, result, ValueRange{i, j});
                        b2.create<scf::YieldOp>(l2);
                    });
                b.create<scf::YieldOp>(l);
            });
    }

    return result;
}

// Build matrix multiplication
Value ComputationGraph::buildMatMul(OpBuilder& builder, Location loc, Value lhs, Value rhs) {
    auto lhsType = lhs.getType().cast<MemRefType>();
    auto rhsType = rhs.getType().cast<MemRefType>();

    auto m = builder.create<memref::DimOp>(loc, lhs, 0);
    auto n = builder.create<memref::DimOp>(loc, rhs, 1);

    SmallVector<int64_t> resultShape = {lhsType.getShape()[0], rhsType.getShape()[1]};
    auto resultType = MemRefType::get(resultShape, builder.getF32Type());

    SmallVector<Value> dynSizes;
    if (lhsType.isDynamicDim(0)) dynSizes.push_back(m);
    if (rhsType.isDynamicDim(1)) dynSizes.push_back(n);

    Value result = builder.create<memref::AllocOp>(loc, resultType, dynSizes);

    // Initialize to zero
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto zeroF = builder.create<arith::ConstantOp>(loc, builder.getF32Type(),
                                                   builder.getF32FloatAttr(0.0));

    builder.create<scf::ForOp>(loc, zero, m, one,
        ValueRange{}, [&](OpBuilder& b, Location l, Value i, ValueRange) {
            b.create<scf::ForOp>(l, zero, n, one,
                ValueRange{}, [&](OpBuilder& b2, Location l2, Value j, ValueRange) {
                    b2.create<memref::StoreOp>(l2, zeroF, result, ValueRange{i, j});
                    b2.create<scf::YieldOp>(l2);
                });
            b.create<scf::YieldOp>(l);
        });

    // Perform matrix multiplication
    auto k = builder.create<memref::DimOp>(loc, lhs, 1);

    builder.create<scf::ForOp>(loc, zero, m, one,
        ValueRange{}, [&](OpBuilder& b, Location l, Value i, ValueRange) {
            b.create<scf::ForOp>(l, zero, n, one,
                ValueRange{}, [&](OpBuilder& b2, Location l2, Value j, ValueRange) {
                    b2.create<scf::ForOp>(l2, zero, k, one,
                        ValueRange{}, [&](OpBuilder& b3, Location l3, Value kk, ValueRange) {
                            auto lVal = b3.create<memref::LoadOp>(l3, lhs, ValueRange{i, kk});
                            auto rVal = b3.create<memref::LoadOp>(l3, rhs, ValueRange{kk, j});
                            auto prod = b3.create<arith::MulFOp>(l3, lVal, rVal);
                            auto acc = b3.create<memref::LoadOp>(l3, result, ValueRange{i, j});
                            auto sum = b3.create<arith::AddFOp>(l3, acc, prod);
                            b3.create<memref::StoreOp>(l3, sum, result, ValueRange{i, j});
                            b3.create<scf::YieldOp>(l3);
                        });
                    b2.create<scf::YieldOp>(l2);
                });
            b.create<scf::YieldOp>(l);
        });

    return result;
}

// Build ReLU activation
Value ComputationGraph::buildReLU(OpBuilder& builder, Location loc, Value input) {
    auto inputType = input.getType().cast<MemRefType>();

    SmallVector<Value> dynSizes;
    for (int i = 0; i < inputType.getRank(); ++i) {
        if (inputType.isDynamicDim(i)) {
            dynSizes.push_back(builder.create<memref::DimOp>(loc, input, i));
        }
    }
    Value result = builder.create<memref::AllocOp>(loc, inputType, dynSizes);

    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto zeroF = builder.create<arith::ConstantOp>(loc, builder.getF32Type(),
                                                   builder.getF32FloatAttr(0.0));

    if (inputType.getRank() == 1) {
        auto dim0 = builder.create<memref::DimOp>(loc, input, 0);
        builder.create<scf::ForOp>(loc, zero, dim0, one,
            ValueRange{}, [&](OpBuilder& b, Location l, Value i, ValueRange) {
                auto val = b.create<memref::LoadOp>(l, input, ValueRange{i});
                auto maxVal = b.create<arith::MaximumFOp>(l, val, zeroF);
                b.create<memref::StoreOp>(l, maxVal, result, ValueRange{i});
                b.create<scf::YieldOp>(l);
            });
    } else if (inputType.getRank() == 2) {
        auto dim0 = builder.create<memref::DimOp>(loc, input, 0);
        auto dim1 = builder.create<memref::DimOp>(loc, input, 1);
        builder.create<scf::ForOp>(loc, zero, dim0, one,
            ValueRange{}, [&](OpBuilder& b, Location l, Value i, ValueRange) {
                b.create<scf::ForOp>(l, zero, dim1, one,
                    ValueRange{}, [&](OpBuilder& b2, Location l2, Value j, ValueRange) {
                        auto val = b2.create<memref::LoadOp>(l2, input, ValueRange{i, j});
                        auto maxVal = b2.create<arith::MaximumFOp>(l2, val, zeroF);
                        b2.create<memref::StoreOp>(l2, maxVal, result, ValueRange{i, j});
                        b2.create<scf::YieldOp>(l2);
                    });
                b.create<scf::YieldOp>(l);
            });
    }

    return result;
}

// Build Softmax (from Chapter 6)
Value ComputationGraph::buildSoftmax(OpBuilder& builder, Location loc, Value input) {
    auto inputType = input.getType().cast<MemRefType>();
    auto dim0 = builder.create<memref::DimOp>(loc, input, 0);

    SmallVector<Value> dynSizes;
    if (inputType.isDynamicDim(0)) {
        dynSizes.push_back(dim0);
    }
    Value result = builder.create<memref::AllocOp>(loc, inputType, dynSizes);

    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto negInf = builder.create<arith::ConstantOp>(loc, builder.getF32Type(),
        builder.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    auto zeroF = builder.create<arith::ConstantOp>(loc, builder.getF32Type(),
        builder.getF32FloatAttr(0.0));

    // Pass 1: Find max
    auto maxVal = builder.create<scf::ForOp>(loc, zero, dim0, one,
        ValueRange{negInf}, [&](OpBuilder& b, Location l, Value i, ValueRange iterArgs) {
            auto val = b.create<memref::LoadOp>(l, input, ValueRange{i});
            auto currentMax = b.create<arith::MaximumFOp>(l, iterArgs[0], val);
            b.create<scf::YieldOp>(l, ValueRange{currentMax});
        }).getResult(0);

    // Pass 2: Compute exp and sum
    auto sum = builder.create<scf::ForOp>(loc, zero, dim0, one,
        ValueRange{zeroF}, [&](OpBuilder& b, Location l, Value i, ValueRange iterArgs) {
            auto val = b.create<memref::LoadOp>(l, input, ValueRange{i});
            auto diff = b.create<arith::SubFOp>(l, val, maxVal);
            auto expVal = b.create<math::ExpOp>(l, diff);
            b.create<memref::StoreOp>(l, expVal, result, ValueRange{i});
            auto newSum = b.create<arith::AddFOp>(l, iterArgs[0], expVal);
            b.create<scf::YieldOp>(l, ValueRange{newSum});
        }).getResult(0);

    // Pass 3: Normalize
    builder.create<scf::ForOp>(loc, zero, dim0, one,
        ValueRange{}, [&](OpBuilder& b, Location l, Value i, ValueRange) {
            auto val = b.create<memref::LoadOp>(l, result, ValueRange{i});
            auto normalized = b.create<arith::DivFOp>(l, val, sum);
            b.create<memref::StoreOp>(l, normalized, result, ValueRange{i});
            b.create<scf::YieldOp>(l);
        });

    return result;
}

// Export the ComputationGraph class
// Note: We use C++ wrapper functions instead of raw C exports to handle C++ types
ComputationGraph* createGraph(MLIRContext* ctx) {
    return new ComputationGraph(ctx);
}

void deleteGraph(ComputationGraph* graph) {
    delete graph;
}

int addInput(ComputationGraph* graph, int64_t* shape, int rank) {
    std::vector<int64_t> shapeVec(shape, shape + rank);
    return graph->addInput(shapeVec);
}

int addAddOp(ComputationGraph* graph, int lhs, int rhs) {
    return graph->add(lhs, rhs);
}

int addMulOp(ComputationGraph* graph, int lhs, int rhs) {
    return graph->mul(lhs, rhs);
}

int addMatMulOp(ComputationGraph* graph, int lhs, int rhs) {
    return graph->matmul(lhs, rhs);
}

int addReLUOp(ComputationGraph* graph, int input) {
    return graph->relu(input);
}

int addSoftmaxOp(ComputationGraph* graph, int input) {
    return graph->softmax(input);
}

ModuleOp generateMLIR(ComputationGraph* graph, int outputId, const char* funcName) {
    return graph->generateMLIR(outputId, funcName);
}