cmake_minimum_required(VERSION 3.20)
project(ch10-optimizations)

# MLIR is already found by root CMakeLists.txt
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
# Reuse Chapter 9's NN Dialect (avoid duplicate TableGen targets)
#===------------------------------------------------------------------------===#

# Include Chapter 9's generated headers
include_directories(${CMAKE_SOURCE_DIR}/ch.9.TableGen-dialect/inc)
include_directories(${CMAKE_BINARY_DIR}/ch.9.TableGen-dialect/inc)

# Link against Chapter 9's NNDialect library
set(NN_DIALECT_LIB "${CMAKE_BINARY_DIR}/ch.9.TableGen-dialect/libNNDialect.a")

#===------------------------------------------------------------------------===#
# Python bindings (single file - all logic in bindings.cpp)
#===------------------------------------------------------------------------===#
pybind11_add_module(ch10
    src/bindings.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(ch10 PRIVATE
    ${NN_DIALECT_LIB}
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

target_include_directories(ch10 PRIVATE
    ${CMAKE_SOURCE_DIR}/ch.9.TableGen-dialect/inc
    ${CMAKE_BINARY_DIR}/ch.9.TableGen-dialect/inc
    ${FFI_INCLUDE_DIRS}
)

add_dependencies(ch10 NNDialect)

set_target_properties(ch10 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ch.10.Optimizations"
)

message(STATUS "Chapter 10 build configured:")
message(STATUS "  - NN dialect from Chapter 9 (reused)")
message(STATUS "  - NEW: Baseline vs optimized lowering pipelines")
message(STATUS "  - NEW: Performance comparison tests")
message(STATUS "  - Python module: ch10.so")