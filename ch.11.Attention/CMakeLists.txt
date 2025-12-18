# Chapter 11: Attention Mechanism with Transformer Dialect
cmake_minimum_required(VERSION 3.20)

# Find libffi
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFI REQUIRED libffi)

# Generate dialect files using TableGen
set(LLVM_TARGET_DEFINITIONS inc/TransformerDialect.td)
mlir_tablegen(inc/TransformerDialect.h.inc -gen-dialect-decls -I ${CMAKE_CURRENT_SOURCE_DIR}/inc)
mlir_tablegen(inc/TransformerDialect.cpp.inc -gen-dialect-defs -I ${CMAKE_CURRENT_SOURCE_DIR}/inc)
add_public_tablegen_target(TransformerDialectIncGen)

set(LLVM_TARGET_DEFINITIONS inc/TransformerOps.td)
mlir_tablegen(inc/TransformerOps.h.inc -gen-op-decls -I ${CMAKE_CURRENT_SOURCE_DIR}/inc)
mlir_tablegen(inc/TransformerOps.cpp.inc -gen-op-defs -I ${CMAKE_CURRENT_SOURCE_DIR}/inc)
add_public_tablegen_target(TransformerOpsIncGen)

# Build the Transformer dialect as a library
add_library(TransformerDialect STATIC
    src/TransformerDialect.cpp
    src/TransformerOps.cpp
    src/TransformerToStandard.cpp
)

target_include_directories(TransformerDialect PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_BINARY_DIR}/inc
)

add_dependencies(TransformerDialect
    TransformerDialectIncGen
    TransformerOpsIncGen
)

target_link_libraries(TransformerDialect PUBLIC
    MLIRIR
    MLIRSupport
    MLIRSideEffectInterfaces
)

# Python bindings for attention
pybind11_add_module(ch11 src/bindings.cpp)

target_include_directories(ch11 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_BINARY_DIR}/inc
    ${FFI_INCLUDE_DIRS}
)

target_link_libraries(ch11 PRIVATE
    TransformerDialect  # Our Transformer dialect
    MLIRExecutionEngine
    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRSupport
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRMathDialect
    MLIRTensorDialect
    MLIRVectorDialect
    MLIRLLVMDialect
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRMathToLLVM
    MLIRMathToLibm
    MLIRMemRefToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRVectorToLLVM
    MLIRVectorToSCF
    MLIRLinalgTransforms
    LLVM
    ${FFI_LIBRARIES}
)
