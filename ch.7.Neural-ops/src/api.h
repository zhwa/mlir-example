#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

// Forward declaration
class ComputationGraph;
class JITCompiler;

// Functions from ir.cpp
ComputationGraph* createGraph(mlir::MLIRContext* ctx);
void deleteGraph(ComputationGraph* graph);
int addInput(ComputationGraph* graph, int64_t* shape, int rank);
int addAddOp(ComputationGraph* graph, int lhs, int rhs);
int addMulOp(ComputationGraph* graph, int lhs, int rhs);
int addMatMulOp(ComputationGraph* graph, int lhs, int rhs);
int addReLUOp(ComputationGraph* graph, int input);
int addSoftmaxOp(ComputationGraph* graph, int input);
mlir::ModuleOp generateMLIR(ComputationGraph* graph, int outputId, const char* funcName);

// Functions from jit.cpp
JITCompiler* createJIT();
void deleteJIT(JITCompiler* jit);
void* compileModule(JITCompiler* jit, mlir::ModuleOp module, const char* funcName);