//===- bindings.cpp - Python Bindings for GPU IR Generation --------------===//
//
// This file exposes GPU dialect IR generation functions to Python.
// Unlike other chapters, this doesn't execute code - it only generates
// and returns IR as strings for educational inspection.
//
// Functions exposed to Python:
//   - vector_add_ir() → str    # GPU IR for vector addition
//   - matmul_ir() → str        # GPU IR for matrix multiplication
//   - softmax_ir() → str       # GPU IR for softmax with reductions
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

// Forward declarations from gpu_kernels.cpp
std::string buildVectorAddGPU();
std::string buildMatMulGPU();
std::string buildSoftmaxGPU();

/// Python module definition
/// Module name: ch15
PYBIND11_MODULE(ch15, m) {
    m.doc() = "GPU Dialect IR Generation for Educational Purposes";

    m.def("vector_add_ir", &buildVectorAddGPU,
          "Generate GPU dialect IR for vector addition (1D parallelism)\n\n"
          "Returns:\n"
          "    str: MLIR GPU dialect IR demonstrating:\n"
          "         - gpu.thread_id, gpu.block_id for thread indexing\n"
          "         - gpu.launch_func for kernel launch\n"
          "         - Bounds checking pattern\n"
          "         - 1D parallel computation");

    m.def("matmul_ir", &buildMatMulGPU,
          "Generate GPU dialect IR for matrix multiplication (2D parallelism + shared memory)\n\n"
          "Returns:\n"
          "    str: MLIR GPU dialect IR demonstrating:\n"
          "         - 2D thread organization (blocks and threads)\n"
          "         - gpu.alloc for shared memory allocation\n"
          "         - gpu.barrier for synchronization\n"
          "         - Tiled matrix multiplication algorithm");

    m.def("softmax_ir", &buildSoftmaxGPU,
          "Generate GPU dialect IR for softmax (reductions + block cooperation)\n\n"
          "Returns:\n"
          "    str: MLIR GPU dialect IR demonstrating:\n"
          "         - Multi-stage reduction algorithm\n"
          "         - Multiple gpu.barrier synchronization points\n"
          "         - Shared memory for cooperative reductions\n"
          "         - Block cooperation pattern");
}
