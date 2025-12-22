/*
 * Python bindings for Nano Serving Engine
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "kv_cache.h"

namespace py = pybind11;
using namespace nano_serving;

PYBIND11_MODULE(ch16, m) {
    m.doc() = "Nano LLM Serving Engine - Chapter 16";

    // KVCachePool class
    py::class_<KVCachePool>(m, "KVCachePool")
        .def(py::init<int, int, int, int, int>(),
             py::arg("num_pages"),
             py::arg("page_size"),
             py::arg("num_layers"),
             py::arg("num_heads"),
             py::arg("head_dim"),
             "Initialize KV cache pool")

        .def("allocate", &KVCachePool::allocate,
             py::arg("num_tokens"),
             "Allocate pages for num_tokens")

        .def("free", &KVCachePool::free,
             py::arg("pages"),
             "Free pages back to pool")

        .def("store_kv", [](KVCachePool& pool,
                           py::array_t<float> k,
                           py::array_t<float> v,
                           const std::vector<int>& page_indices,
                           int layer_id) {
            // Validate input shapes
            auto k_buf = k.request();
            auto v_buf = v.request();

            if (k_buf.ndim != 3 || v_buf.ndim != 3) {
                throw std::runtime_error("K/V must be 3D tensors [num_tokens, num_heads, head_dim]");
            }

            int num_tokens = k_buf.shape[0];

            // Call C++ implementation
            pool.storeKV(static_cast<const float*>(k_buf.ptr),
                        static_cast<const float*>(v_buf.ptr),
                        page_indices,
                        layer_id,
                        num_tokens);
        },
        py::arg("k"),
        py::arg("v"),
        py::arg("page_indices"),
        py::arg("layer_id"),
        "Store K/V tensors at specified pages")

        .def("get_layer_cache", [](KVCachePool& pool, int layer_id) {
            auto [k_ptr, v_ptr] = pool.getLayerCache(layer_id);

            // Return as numpy arrays (view, not copy)
            int cache_size = pool.getNumPages() * pool.getPageSize();

            // Shape: [num_pages * page_size, num_heads, head_dim]
            // Note: This is a simplified view for testing
            py::array_t<float> k_array({cache_size}, {sizeof(float)}, k_ptr);
            py::array_t<float> v_array({cache_size}, {sizeof(float)}, v_ptr);

            return py::make_tuple(k_array, v_array);
        },
        py::arg("layer_id"),
        "Get K/V cache arrays for a layer")

        .def("get_num_free_pages", &KVCachePool::getNumFreePages,
             "Get number of free pages")

        .def("get_num_pages", &KVCachePool::getNumPages,
             "Get total number of pages")

        .def("get_page_size", &KVCachePool::getPageSize,
             "Get page size");
}