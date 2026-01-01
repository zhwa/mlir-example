import sys
import os

def import_cpp_module(module_name: str, chapter_path: str):
    """
    Import C++ module from build directories

    Args:
        module_name: Name of the module (e.g., 'ch14', 'ch16')
        chapter_path: Path suffix (e.g., 'ch.14.GPT-Optimized', 'ch.16.Nano-Serving')

    Returns:
        Imported module

    Raises:
        ImportError: If module cannot be found
    """
    build_paths = [
        f'../build/x64-release/{chapter_path}',
        f'../build/x64-debug/{chapter_path}',
        f'build/x64-release/{chapter_path}',
        f'build/x64-debug/{chapter_path}',
        f'../../build/x64-release/{chapter_path}',
        f'../../build/x64-debug/{chapter_path}',
    ]

    for path in build_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            try:
                return __import__(module_name)
            except ImportError:
                sys.path.pop(0)

    # Final attempt without path (if already in sys.path)
    return __import__(module_name)

# Pre-import common modules
ch14 = import_cpp_module('ch14', 'ch.14.GPT-Optimized')
ch16 = import_cpp_module('ch16', 'ch.16.Nano-Serving')

# Export commonly used classes
KVCachePool = ch14.KVCachePool
RadixCache = ch16.RadixCache
# Note: RadixNode no longer exposed - nodes are accessed via cache.get_node(node_id)
INVALID_NODE = ch16.INVALID_NODE