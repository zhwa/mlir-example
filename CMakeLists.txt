cmake_minimum_required(VERSION 3.20)
project(mlir-gemm-tutorial LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find system LLVM/MLIR (version 20 or compatible)
find_package(LLVM 20.0 CONFIG)
if(NOT LLVM_FOUND)
    find_package(LLVM REQUIRED CONFIG)
endif()

# Force use of shared LLVM library to avoid CommandLine duplicate registration
set(LLVM_LINK_LLVM_DYLIB ON)
link_libraries(LLVM)

find_package(MLIR REQUIRED CONFIG)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

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

# LLVM components - use shared library
set(llvm_libs LLVM)

# Get MLIR dialect/conversion/extension libraries
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

# Common MLIR libraries for all subprojects
set(MLIR_COMMON_LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRAnalysis
  MLIRExecutionEngine
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRTransforms
  ${llvm_libs}
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