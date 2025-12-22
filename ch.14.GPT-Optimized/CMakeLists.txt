# Chapter 14: Optimized GPT (Linalg + Fusion + Vectorization)
project(ch14-gpt-optimized)

# MLIR is already found by root CMakeLists.txt

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/inc)
include_directories(${PROJECT_BINARY_DIR}/inc)

# TableGen for Dialect (use unique target names)
set(LLVM_TARGET_DEFINITIONS inc/TransformerDialect.td)
mlir_tablegen(inc/TransformerDialect.h.inc -gen-dialect-decls)
mlir_tablegen(inc/TransformerDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(Ch14TransformerDialectIncGen)

# TableGen for Operations (use unique target names)
set(LLVM_TARGET_DEFINITIONS inc/TransformerOps.td)
mlir_tablegen(inc/TransformerOps.h.inc -gen-op-decls)
mlir_tablegen(inc/TransformerOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(Ch14TransformerOpsIncGen)

# Transformer Dialect Library (use unique name)
add_llvm_library(MLIRCh14TransformerDialect
  src/TransformerDialect.cpp
  src/TransformerOps.cpp
  src/TransformerPasses.cpp

  DEPENDS
  Ch14TransformerDialectIncGen
  Ch14TransformerOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRTransforms
)

# Find libffi
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFI REQUIRED libffi)

# Python bindings (pybind11 is already fetched at root level)

pybind11_add_module(ch14 src/bindings.cpp)

target_link_libraries(ch14 PRIVATE
  MLIRCh14TransformerDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRExecutionEngine
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgTransformOps
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRVectorTransforms
  MLIRVectorTransformOps
  MLIRTransformDialect
  MLIRTransforms
  MLIRTargetLLVMIRExport
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRLinalgToStandard
  MLIRMathToLLVM
  MLIRMathToLibm
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRVectorToLLVMPass
  MLIRVectorToSCF
  LLVM
  ${FFI_LIBRARIES}
)

target_include_directories(ch14 PRIVATE
  ${PROJECT_SOURCE_DIR}/inc
  ${PROJECT_BINARY_DIR}/inc
  ${FFI_INCLUDE_DIRS}
)
