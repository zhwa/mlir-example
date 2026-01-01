// Python bindings for Nano Serving Engine (pointer-free arena-based design)
#include "radix_cache.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace nano_serving;

PYBIND11_MODULE(ch16, m) {
    m.doc() = "Nano LLM Serving Engine - Chapter 16: Radix Cache with Arena Allocation";

    // Expose NodeID as a type alias for clarity
    m.attr("INVALID_NODE") = INVALID_NODE;

    // RadixCache class - nodes are accessed via cache.get_node(id)
    py::class_<RadixCache>(m, "RadixCache")
        .def(py::init<>(),
             "Initialize radix cache")

        .def("match_prefix", 
             [](RadixCache& self, const std::vector<int>& tokens) {
                 auto result = self.match_prefix(tokens);
                 return py::make_tuple(result.first, result.second);
             },
             py::arg("tokens"),
             "Find longest matching prefix, returns (match_length, node_id)")

        .def("insert", &RadixCache::insert,
             py::arg("tokens"),
             py::arg("kv_pages"),
             "Insert token sequence into tree with KV pages, returns node_id")

        .def("get_pages_for_prefix", &RadixCache::get_pages_for_prefix,
             py::arg("tokens"),
             "Get KV page indices for a prefix")

        .def("find_lru_leaf", 
             [](RadixCache& self) {
                 auto result = self.find_lru_leaf();
                 return py::make_tuple(result.first, result.second);
             },
             "Find least-recently-used leaf node, returns (leaf_id, path_ids)")

        .def("evict_leaf", &RadixCache::evict_leaf,
             py::arg("leaf_id"),
             py::arg("path"),
             "Evict a leaf node, returns freed pages")

        .def("evict_until_available", &RadixCache::evict_until_available,
             py::arg("num_pages"),
             "Evict LRU leaves until at least num_pages are freed")

        .def("clear", &RadixCache::clear,
             "Clear entire cache")

        .def("get_num_nodes", &RadixCache::get_num_nodes,
             "Get total number of active nodes")

        .def("get_total_tokens_cached", &RadixCache::get_total_tokens_cached,
             "Get total tokens cached")

        .def("get_root_id", &RadixCache::get_root_id,
             "Get root node ID")

        // Node accessor - returns node properties as a dict
        .def("get_node", 
             [](RadixCache& self, NodeID id) {
                 const auto& node = self.get_node(id);
                 py::dict d;
                 d["token"] = node.get_token();
                 d["is_leaf"] = node.is_leaf();
                 d["is_root"] = node.is_root();
                 d["last_access_time"] = node.get_last_access_time();
                 d["kv_pages"] = node.get_kv_pages();
                 
                 // Convert children map to dict
                 py::dict children;
                 for (const auto& [token, child_id] : node.get_children()) {
                     children[py::int_(token)] = child_id;
                 }
                 d["children"] = children;
                 
                 return d;
             },
             py::arg("node_id"),
             "Get node properties as dictionary (no pointers!)");
}