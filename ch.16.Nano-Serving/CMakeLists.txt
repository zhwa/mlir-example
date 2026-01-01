project(ch16-nano-serving)

# MLIR is already found by root CMakeLists.txt

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Fetch pybind11 (already fetched by root CMakeLists.txt, just use it)
include_directories(${pybind11_INCLUDE_DIRS})

# C++ library
add_library(ch16_lib STATIC
    src/radix_cache.cpp
)

target_include_directories(ch16_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Pybind11 module
pybind11_add_module(ch16 src/bindings.cpp)
target_link_libraries(ch16 PRIVATE ch16_lib)

# Set output directory
set_target_properties(ch16 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
