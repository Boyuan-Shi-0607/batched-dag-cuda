#pragma once

#include "bdg/graph.hpp"
#include <vector>

namespace bdg {

struct PackedEdge { std::uint16_t parent, child, weight, coefficient; };
static_assert(sizeof(PackedEdge) == 8, "PackedEdge must occupy eight bytes");

struct Partition {
    std::uint64_t edge_offset, parent_offset, child_offset, weight_offset, coefficient_offset;
    std::uint32_t edge_count, parent_count, child_count, weight_count, coefficient_count;
    std::size_t shared_bytes(std::size_t scalar_bytes) const;
};

struct LayerPlan {
    std::vector<std::uint64_t> row_offsets;
    std::vector<std::uint32_t> parents, children, weights;
    std::vector<double> coefficients;
    std::vector<PackedEdge> packed_edges;
    std::vector<std::uint32_t> unique_parents, unique_children, unique_weights;
    std::vector<double> unique_coefficients;
    std::vector<Partition> partitions;
    std::size_t max_shared_bytes = 0;
};

struct Readout { std::uint32_t node, column; };
struct Plan {
    std::vector<LayerPlan> layers;
    std::vector<std::vector<Readout>> readouts;
};

Plan compile_plan(const Graph& graph, std::size_t max_edges = 1024,
                  std::size_t shared_budget = 32768, std::size_t scalar_bytes = 4);

}  // namespace bdg
