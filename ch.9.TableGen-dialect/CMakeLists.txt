cmake_minimum_required(VERSION 3.20)
project(ch9-tablegen-dialect)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find MLIR
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/inc)
include_directories(${PROJECT_BINARY_DIR}/inc)  # For generated files
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# pybind11
add_subdirectory(${CMAKE_SOURCE_DIR}/build/deps/pybind11 pybind11_build EXCLUDE_FROM_ALL)

# Find libffi
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFI REQUIRED libffi)

#===------------------------------------------------------------------------===#
# TableGen for dialect code generation
#===------------------------------------------------------------------------===#

set(LLVM_TARGET_DEFINITIONS inc/NNOps.td)
mlir_tablegen(inc/NNOps.h.inc -gen-op-decls)
mlir_tablegen(inc/NNOps.cpp.inc -gen-op-defs)
mlir_tablegen(inc/NNOpsDialect.h.inc -gen-dialect-decls -dialect=nn)
mlir_tablegen(inc/NNOpsDialect.cpp.inc -gen-dialect-defs -dialect=nn)
add_public_tablegen_target(NNOpsIncGen)

# Make sure TableGen runs before compiling
add_dependencies(NNOpsIncGen MLIRTableGen)

#===------------------------------------------------------------------------===#
# NN Dialect Library
#===------------------------------------------------------------------------===#

add_library(NNDialect STATIC
    src/NNDialect.cpp
    src/NNOps.cpp
    src/NNToStandard.cpp
)

target_include_directories(NNDialect PUBLIC
    ${PROJECT_SOURCE_DIR}/inc
    ${PROJECT_BINARY_DIR}/inc
)

# Get MLIR libraries
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(NNDialect PUBLIC
    ${dialect_libs}
    ${conversion_libs}
    MLIRAnalysis
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRSupport
)

add_dependencies(NNDialect NNOpsIncGen)

#===------------------------------------------------------------------------===#
# Python bindings
#===------------------------------------------------------------------------===#

pybind11_add_module(ch9
    src/bindings.cpp
)

target_link_libraries(ch9 PRIVATE
    NNDialect
    ${dialect_libs}
    ${conversion_libs}
    MLIRExecutionEngine
    MLIRJitRunner
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRSupport
    MLIROptLib
    LLVM
    ${FFI_LIBRARIES}
    LLVMX86Info
    LLVMX86AsmParser
    LLVMX86CodeGen
    LLVMX86Desc
)

target_include_directories(ch9 PRIVATE
    ${PROJECT_SOURCE_DIR}/inc
    ${PROJECT_BINARY_DIR}/inc
    ${FFI_INCLUDE_DIRS}
)

add_dependencies(ch9 NNDialect)

set_target_properties(ch9 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ch.9.TableGen-dialect"
)

message(STATUS "Chapter 9 build configured:")
message(STATUS "  - Custom NN dialect with TableGen")
message(STATUS "  - TableGen generates: NNOps.h.inc, NNOps.cpp.inc, NNOpsDialect.h.inc")
message(STATUS "  - Python module: ch9.so")
