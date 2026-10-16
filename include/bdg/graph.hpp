#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bdg {

struct Edge {
    std::uint32_t parent;
    std::uint32_t child;
    std::uint32_t weight;
    double coefficient = 1.0;
};

struct Output {
    std::uint32_t layer;
    std::uint32_t node;
};

// Every edge in layers[l] goes from layer l to l+1. Node IDs are layer-local.
struct Graph {
    std::uint32_t weight_count = 0;
    std::vector<std::uint32_t> widths;
    std::vector<std::vector<Edge>> layers;
    std::vector<Output> outputs;

    void validate() const;
    std::size_t edge_count() const;
    std::uint32_t max_width() const;
};

std::size_t checked_product(std::size_t a, std::size_t b);
std::size_t ceil_div(std::size_t n, std::size_t d);

}  // namespace bdg
