#pragma once
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

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

    GraphOperation(OpType t, std::vector<int> ins, std::vector<int64_t> sh, int i);
};

// Computation graph that tracks operations and generates MLIR
class ComputationGraph {
public:
    explicit ComputationGraph(mlir::MLIRContext* ctx);

    // Add an input placeholder
    int addInput(const std::vector<int64_t>& shape);

    // Add element-wise addition
    int add(int lhs, int rhs);

    // Add element-wise multiplication
    int mul(int lhs, int rhs);

    // Add matrix multiplication
    int matmul(int lhs, int rhs);

    // Add ReLU activation
    int relu(int input);

    // Add Softmax activation
    int softmax(int input);

    // Generate MLIR module from the computation graph
    mlir::ModuleOp generateMLIR(int outputId, const std::string& funcName);

private:
    mlir::MLIRContext* context;
    std::vector<GraphOperation> operations;
    int nextId;

    // Helper to recursively build operations
    mlir::Value buildOperation(mlir::OpBuilder& builder, mlir::Location loc, int opId,
                               std::unordered_map<int, mlir::Value>& valueMap,
                               const std::vector<mlir::Value>& funcArgs);

    // Helper to build element-wise operations
    mlir::Value buildElementWiseOp(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Value lhs, mlir::Value rhs,
                                   GraphOperation::OpType opType);

    // Helper to build matrix multiplication
    mlir::Value buildMatMul(mlir::OpBuilder& builder, mlir::Location loc, 
                           mlir::Value lhs, mlir::Value rhs);

    // Helper to build ReLU
    mlir::Value buildReLU(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input);

    // Helper to build Softmax
    mlir::Value buildSoftmax(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input);
};

// JIT Compiler class (using pimpl to hide LLVM types)
class JITCompiler {
public:
    JITCompiler();
    ~JITCompiler();
    void* compile(mlir::ModuleOp module, const std::string& funcName);

private:
    struct Impl;  // Forward declaration of implementation
    Impl* pImpl;  // Pointer to implementation
};