project(ch5-vector-ops)

# Python module
file(GLOB CPP_SOURCE_FILES "src/*.cpp")
pybind11_add_module(${PROJECT_NAME} ${CPP_SOURCE_FILES})

# Set output name to use underscores
set_target_properties(${PROJECT_NAME} PROPERTIES 
    OUTPUT_NAME "ch5_vector_ops"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE 
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

# Link against common MLIR libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${MLIR_COMMON_LIBS})