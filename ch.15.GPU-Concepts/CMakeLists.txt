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

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# ============================================================================
# AOT Compilation Architecture (No JIT, No Python)
# ============================================================================
# Each kernel is compiled separately as an object file
# Main test executable links all kernels together
# Runs on CPU via SCF loop emulation (no actual GPU needed)

# Common utilities
add_library(ch15_common OBJECT
  src/common.cpp
)

target_link_libraries(ch15_common PUBLIC
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation

  # Dialects
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRMathDialect
  MLIRControlFlowDialect
  MLIRLLVMDialect

  # Conversions
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRArithToLLVM
  MLIRMathToLLVM
  MLIRMathToLibm
  MLIRReconcileUnrealizedCasts
)

# Phase 0: Vector Add Kernel
add_library(vector_add_kernel OBJECT
  src/vector_add.cpp
)

target_link_libraries(vector_add_kernel PUBLIC
  ch15_common
  MLIRExecutionEngine
  LLVMOrcJIT
)

# Phase 1: MatMul Kernel (2D GPU)
add_library(matmul_kernel OBJECT
  src/matmul.cpp
)

target_link_libraries(matmul_kernel PUBLIC
  ch15_common
  MLIRExecutionEngine
  LLVMOrcJIT
)

# Phase 2: Element-wise Operations
add_library(elementwise_kernel OBJECT
  src/elementwise.cpp
)

target_link_libraries(elementwise_kernel PUBLIC
  ch15_common
  MLIRExecutionEngine
  LLVMOrcJIT
)

# Phase 3: Softmax (Reductions)
add_library(softmax_kernel OBJECT
  src/softmax.cpp
)

target_link_libraries(softmax_kernel PUBLIC
  ch15_common
  MLIRExecutionEngine
  LLVMOrcJIT
)

# Phase 4: LayerNorm (The JIT Bug Survivor!)
add_library(layernorm_kernel OBJECT
  src/layernorm.cpp
)

target_link_libraries(layernorm_kernel PUBLIC
  ch15_common
  MLIRExecutionEngine
  LLVMOrcJIT
)

# Phase 5: Transpose (Memory Access Patterns)
add_library(transpose_kernel OBJECT
  src/transpose.cpp
)

target_link_libraries(transpose_kernel PUBLIC
  ch15_common
  MLIRExecutionEngine
  LLVMOrcJIT
)

# Phase 6: Attention Mechanism (Composing Operations)
add_library(attention_kernel OBJECT
  src/attention.cpp
)

target_link_libraries(attention_kernel PUBLIC
  ch15_common
  MLIRExecutionEngine
  LLVMOrcJIT
)

# Phase 7: Transformer & Nano-GPT
add_library(transformer_kernel OBJECT
  src/transformer.cpp
)

target_link_libraries(transformer_kernel PUBLIC
  ch15_common
  MLIRExecutionEngine
  LLVMOrcJIT
)

# Test executable
add_executable(ch15_test
  src/main.cpp
)

target_link_libraries(ch15_test PRIVATE
  vector_add_kernel
  matmul_kernel
  elementwise_kernel
  softmax_kernel
  layernorm_kernel
  transpose_kernel
  attention_kernel
  transformer_kernel
  ch15_common
)

# Compiler flags
target_compile_features(ch15_test PRIVATE cxx_std_17)