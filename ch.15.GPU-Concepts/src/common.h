#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"

#include <memory>
#include <string>

namespace ch15 {

// ============================================================================
// MLIR Context Management
// ============================================================================

// Create and initialize MLIR context with all required dialects
// This registers: Func, Arith, MemRef, SCF, Math, ControlFlow, LLVM
mlir::MLIRContext* createContext();

// ============================================================================
// Lowering Pipeline
// ============================================================================

// Lower MLIR module to LLVM dialect
// Pipeline: SCF → CF → (Func+MemRef+Math+Arith) → LLVM → ReconcileUnrealizedCasts
// This runs on CPU (no GPU lowering needed - we already use SCF loops)
mlir::LogicalResult lowerToLLVMDialect(mlir::ModuleOp module);

// ============================================================================
// LLVM IR Translation
// ============================================================================

// Translate MLIR LLVM dialect to LLVM IR
// Returns nullptr on failure
std::unique_ptr<llvm::Module> translateToLLVMIR(
    mlir::ModuleOp module,
    llvm::LLVMContext& llvmContext
);

// ============================================================================
// Helper Builders
// ============================================================================

// Create index constant
inline mlir::Value createIndex(mlir::OpBuilder& builder, mlir::Location loc, int64_t value) {
    return builder.create<mlir::arith::ConstantIndexOp>(loc, value);
}

// Create float constant
inline mlir::Value createFloat(mlir::OpBuilder& builder, mlir::Location loc, float value) {
    return builder.create<mlir::arith::ConstantOp>(
        loc, builder.getF32Type(), builder.getF32FloatAttr(value)
    );
}

} // namespace ch15