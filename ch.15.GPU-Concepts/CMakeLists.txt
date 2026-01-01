project(ch15-gpu-concepts)

# Python module for GPU dialect IR generation
file(GLOB CPP_SOURCE_FILES "src/*.cpp")
pybind11_add_module(${PROJECT_NAME} ${CPP_SOURCE_FILES})

# Set output name to use underscores
set_target_properties(${PROJECT_NAME} PROPERTIES 
    OUTPUT_NAME "ch15"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE 
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

# Link against MLIR libraries needed for GPU dialect IR generation
target_link_libraries(${PROJECT_NAME} PRIVATE
    # Core MLIR
    MLIRIR
    MLIRParser
    MLIRSupport
    
    # Dialects needed for GPU IR
    MLIRArithDialect
    MLIRFuncDialect
    MLIRGPUDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRMathDialect
)