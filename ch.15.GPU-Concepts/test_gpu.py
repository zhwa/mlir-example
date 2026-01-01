"""
Chapter 15: GPU Programming Concepts using MLIR

This module demonstrates GPU programming patterns by generating and displaying
GPU dialect IR. No GPU hardware required - we learn by examining the IR structure!
"""

import sys
import os

# Add build directory to path
build_paths = [
    '../build/x64-release/ch.15.GPU-Concepts',
    '../build/x64-debug/ch.15.GPU-Concepts',
    'build/x64-release/ch.15.GPU-Concepts',
    'build/x64-debug/ch.15.GPU-Concepts'
]

build_dir = None
for path in build_paths:
    if os.path.exists(path):
        build_dir = path
        break

if build_dir:
    print(f"Using build directory: {build_dir}")
    sys.path.insert(0, build_dir)
else:
    print("Warning: Build directory not found, attempting to import anyway")

try:
    import ch15
except ImportError as e:
    print(f"Error: Could not import ch15 module: {e}")
    print("Please build Chapter 15 first:")
    print("  cmake --build build/x64-release --target ch15")
    sys.exit(1)

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def main():
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "Chapter 15: GPU Programming with MLIR" + " " * 21 + "║")
    print("║" + " " * 18 + "Learning GPU Concepts through IR Generation" + " " * 17 + "║")
    print("╚" + "═" * 78 + "╝")

    # Example 1: Vector Addition - Basic 1D parallelism
    print_section("Example 1: Vector Addition - Basic 1D GPU Parallelism")
    print("Concepts: gpu.launch_func, gpu.thread_id, gpu.block_id, gpu.block_dim")
    print()
    ir = ch15.vector_add_ir()
    print(ir)

    print("\n📝 Key observations:")
    print("  • gpu.launch_func launches kernel with grid=(numBlocks,1,1) blocks=(256,1,1)")
    print("  • Inside kernel: globalIdx = blockIdx.x * blockDim.x + threadIdx.x")
    print("  • Bounds check ensures we don't access beyond array size")
    print("  • This pattern scales to any array size automatically")

    # Example 2: Matrix Multiplication - 2D parallelism + shared memory
    print_section("Example 2: Matrix Multiplication - 2D Parallelism + Shared Memory")
    print("Concepts: 2D indexing, gpu.alloc (shared memory), gpu.barrier")
    print()
    ir = ch15.matmul_ir()
    print(ir)

    print("\n📝 Key observations:")
    print("  • 2D grid: each thread computes one output element")
    print("  • gpu.alloc with workgroup address space = shared memory")
    print("  • gpu.barrier synchronizes threads before/after shared memory access")
    print("  • Tiled computation reuses data in fast shared memory")

    # Example 3: Softmax - Reductions and synchronization
    print_section("Example 3: Softmax - Reductions and Multi-Stage Algorithms")
    print("Concepts: Reductions, multiple gpu.barrier calls, block cooperation")
    print()
    ir = ch15.softmax_ir()
    print(ir)

if __name__ == "__main__":
    main()