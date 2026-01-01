/*
 * Python bindings for Nano Serving Engine (using shared_ptr for RAII)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "radix_cache.h"

namespace py = pybind11;
using namespace nano_serving;

PYBIND11_MODULE(ch16, m) {
    m.doc() = "Nano LLM Serving Engine - Chapter 16: Radix Cache for Prefix Sharing (RAII)";

    // RadixNode class - using shared_ptr holder for automatic lifetime management
    py::class_<RadixNode, std::shared_ptr<RadixNode>>(m, "RadixNode")
        .def(py::init<int>(),
             py::arg("token") = -1,
             "Initialize radix node (managed by shared_ptr)")

        .def("get_token", &RadixNode::get_token,
             "Get token ID")

        .def("is_leaf", &RadixNode::is_leaf,
             "Check if this is a leaf node")

        .def("is_root", &RadixNode::is_root,
             "Check if this is the root node")

        .def("use_count", &RadixNode::use_count,
             "Get shared_ptr reference count (automatic RAII management)")

        .def("get_last_access_time", &RadixNode::get_last_access_time,
             "Get last access timestamp")

        .def("get_kv_pages", &RadixNode::get_kv_pages,
             py::return_value_policy::reference_internal,
             "Get KV page indices")

        .def("get_child", &RadixNode::get_child,
             py::arg("token"),
             "Get child node for token (returns shared_ptr)")

        .def("add_child", &RadixNode::add_child,
             py::arg("token"),
             "Add child node for token (returns shared_ptr)")

        .def("remove_child", &RadixNode::remove_child,
             py::arg("token"),
             "Remove child node")

        .def("update_access_time", &RadixNode::update_access_time,
             "Update last access time")

        .def("num_descendants", &RadixNode::num_descendants,
             "Count total descendants");

    // RadixCache class
    py::class_<RadixCache>(m, "RadixCache")
        .def(py::init<>(),
             "Initialize radix cache")

        .def("match_prefix", &RadixCache::match_prefix,
             py::arg("tokens"),
             "Find longest matching prefix, returns (match_length, last_node)")

        .def("insert", &RadixCache::insert,
             py::arg("tokens"),
             py::arg("kv_pages"),
             "Insert token sequence into tree with KV pages")

        .def("get_pages_for_prefix", &RadixCache::get_pages_for_prefix,
             py::arg("tokens"),
             "Get KV page indices for a prefix")

        .def("find_lru_leaf", &RadixCache::find_lru_leaf,
             "Find least-recently-used leaf node, returns (leaf, path)")

        .def("evict_leaf", &RadixCache::evict_leaf,
             py::arg("leaf"),
             py::arg("path"),
             "Evict a leaf node, returns freed pages")

        .def("evict_until_available", &RadixCache::evict_until_available,
             py::arg("num_pages"),
             "Evict LRU leaves until at least num_pages are freed")

        .def("clear", &RadixCache::clear,
             "Clear entire cache")

        .def("get_num_nodes", &RadixCache::get_num_nodes,
             "Get total number of nodes")

        .def("get_total_tokens_cached", &RadixCache::get_total_tokens_cached,
             "Get total tokens cached")

        .def("get_root", &RadixCache::get_root,
             "Get root node (returns shared_ptr)");
}