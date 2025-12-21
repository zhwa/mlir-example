cmake_minimum_required(VERSION 3.20)
project(ch15_gpu_concepts)

# Find MLIR
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Python and pybind11 are already configured by root CMakeLists.txt
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})

# Definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Create Python module
pybind11_add_module(ch15 
  src/bindings.cpp
)

# Link MLIR libraries
target_link_libraries(ch15 PRIVATE
  # Core MLIR
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRExecutionEngine
  
  # Dialects
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRGPUDialect                    # GPU dialect!
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRVectorDialect
  MLIRAffineDialect
  MLIRControlFlowDialect
  
  # Transformations
  MLIRGPUTransforms                 # GPU passes
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIRSCFTransforms
  MLIRVectorTransforms
  
  # Conversions - GPU emulation is KEY!
  MLIRGPUToGPURuntimeTransforms     # GPU runtime setup
  
  # Standard conversions
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  MLIRArithToLLVM
  MLIRVectorToLLVM
  MLIRAffineToStandard
  MLIRReconcileUnrealizedCasts
  
  # LLVM support
  ${llvm_libs}
  LLVMSupport
  LLVMCore
  LLVMOrcJIT
)

# Compiler flags
target_compile_features(ch15 PRIVATE cxx_std_17)
set_target_properties(ch15 PROPERTIES
  CXX_VISIBILITY_PRESET "hidden"
  CUDA_VISIBILITY_PRESET "hidden"
)