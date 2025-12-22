cmake_minimum_required(VERSION 3.20)
project(mlir-gemm-tutorial LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find LLVM/MLIR (version specified by LLVM_DIR/MLIR_DIR in preset)
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Add LLVM/MLIR to module path for TableGen
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# Only include TableGen for ODS/TableGen tasks
include(TableGen)

# Standard include directories
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Fetch pybind11 once at root level
set(DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/deps)
if (NOT EXISTS ${DEPS_DIR})
    file(MAKE_DIRECTORY ${DEPS_DIR})
endif()

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
    SOURCE_DIR ${DEPS_DIR}/pybind11
)
FetchContent_MakeAvailable(pybind11)

# Common MLIR libraries for all chapters
# Explicitly list targets instead of using get_property to avoid 
# "targets already defined" errors with system-installed MLIR
set(MLIR_COMMON_LIBS
  # Dialects
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLinalgDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRVectorDialect
  
  # Conversions
  MLIRAffineToStandard
  MLIRArithToLLVM
  MLIRBufferizationToMemRef
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRLinalgToStandard
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  MLIRTensorToLinalg
  MLIRVectorToLLVM
  
  # Core infrastructure
  MLIRAnalysis
  MLIRExecutionEngine
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRTransforms
  
  # LLVM
  LLVM
)

# Add all chapter subdirectories
add_subdirectory(ch.1.Fixed-size)
add_subdirectory(ch.2.Dynamic-size)
add_subdirectory(ch.3.JIT-caching)
add_subdirectory(ch.4.Tensor-bufferization)
add_subdirectory(ch.5.Vector-ops)
add_subdirectory(ch.6.Softmax)
add_subdirectory(ch.7.Neural-ops)
add_subdirectory(ch.8.Custom-dialect)
add_subdirectory(ch.9.TableGen-dialect)
add_subdirectory(ch.10.Optimizations)
add_subdirectory(ch.11.Attention)
add_subdirectory(ch.12.Transformer)
add_subdirectory(ch.13.GPT)
add_subdirectory(ch.14.GPT-Optimized)
add_subdirectory(ch.15.GPU-Concepts)