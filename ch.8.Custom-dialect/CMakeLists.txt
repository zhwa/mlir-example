cmake_minimum_required(VERSION 3.20)
project(ch8-custom-dialect)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# MLIR is already found by root CMakeLists.txt

include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# pybind11
add_subdirectory(${CMAKE_SOURCE_DIR}/build/deps/pybind11 pybind11_build EXCLUDE_FROM_ALL)

# Find libffi for variadic function calls
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFI REQUIRED libffi)

# Get MLIR libraries
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(MLIR_LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIRExecutionEngine
    MLIRJitRunner
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRSupport
    MLIROptLib
)

# Python bindings library
pybind11_add_module(ch8
    src/bindings.cpp
)

target_link_libraries(ch8 PRIVATE
    ${MLIR_LIBS}
    LLVM
    ${FFI_LIBRARIES}
    LLVMX86Info
    LLVMX86AsmParser
    LLVMX86CodeGen
    LLVMX86Desc
)

target_include_directories(ch8 PRIVATE
    ${FFI_INCLUDE_DIRS}
)

# Install the Python module to the build directory
set_target_properties(ch8 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ch.8.Custom-dialect"
)

# Copy Python source files to build directory
# This makes them available alongside the compiled module
add_custom_command(
    TARGET ch8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/python
        ${CMAKE_BINARY_DIR}/ch.8.Custom-dialect/nn_dialect
    COMMENT "Copying Python source files to build directory"
)

# Create __init__.py that exports the high-level API
add_custom_command(
    TARGET ch8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo
        "# Chapter 8: Custom Dialect API"
        > ${CMAKE_BINARY_DIR}/ch.8.Custom-dialect/__init__.py
    COMMAND ${CMAKE_COMMAND} -E echo
        "from .nn_dialect.graph_builder import Graph"
        >> ${CMAKE_BINARY_DIR}/ch.8.Custom-dialect/__init__.py
    COMMAND ${CMAKE_COMMAND} -E echo
        "from .nn_dialect.lowering import MLIRLowering"
        >> ${CMAKE_BINARY_DIR}/ch.8.Custom-dialect/__init__.py
    COMMAND ${CMAKE_COMMAND} -E echo
        "from .ch8 import execute, execute_with_shape"
        >> ${CMAKE_BINARY_DIR}/ch.8.Custom-dialect/__init__.py
    COMMENT "Creating __init__.py"
)

message(STATUS "Chapter 8 build configured:")
message(STATUS "  - Python module: ch8.so")
message(STATUS "  - Python sources: copied to build directory")
message(STATUS "  - Usage: PYTHONPATH=build/x64-release/ch.8.Custom-dialect python3 test.py")
