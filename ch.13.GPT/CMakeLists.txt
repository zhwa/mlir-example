# Chapter 13: Minimal GPT with RoPE
project(ch13-gpt)

# Find MLIR
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/inc)
include_directories(${PROJECT_BINARY_DIR}/inc)

# TableGen for Dialect (use unique target names)
set(LLVM_TARGET_DEFINITIONS inc/TransformerDialect.td)
mlir_tablegen(inc/TransformerDialect.h.inc -gen-dialect-decls)
mlir_tablegen(inc/TransformerDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(Ch13TransformerDialectIncGen)

# TableGen for Operations (use unique target names)
set(LLVM_TARGET_DEFINITIONS inc/TransformerOps.td)
mlir_tablegen(inc/TransformerOps.h.inc -gen-op-decls)
mlir_tablegen(inc/TransformerOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(Ch13TransformerOpsIncGen)

# Transformer Dialect Library (use unique name)
add_llvm_library(MLIRCh13TransformerDialect
  src/TransformerDialect.cpp
  src/TransformerOps.cpp
  src/TransformerPasses.cpp

  DEPENDS
  Ch13TransformerDialectIncGen
  Ch13TransformerOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
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

pybind11_add_module(ch13 src/bindings.cpp)

target_link_libraries(ch13 PRIVATE
  MLIRCh13TransformerDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRExecutionEngine
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTransforms
  MLIRTargetLLVMIRExport
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMathToLLVM
  MLIRMathToLibm
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  LLVM
  ${FFI_LIBRARIES}
)

target_include_directories(ch13 PRIVATE
  ${PROJECT_SOURCE_DIR}/inc
  ${PROJECT_BINARY_DIR}/inc
  ${FFI_INCLUDE_DIRS}
)
