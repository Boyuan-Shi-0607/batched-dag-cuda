#pragma once

#include "bdg/graph.hpp"

#include <istream>
#include <ostream>
#include <vector>

namespace bdg {

struct ParseLimits {
    std::size_t layers = 4096;
    std::size_t edges = 20000000;
    std::size_t outputs = 1000000;
    std::size_t batch_values = 50000000;
    std::size_t nodes = 10000000;
    std::size_t weights = 10000000;
};

struct InputBatch {
    std::size_t batch = 0;
    std::uint32_t weight_count = 0;
    std::uint32_t source_width = 0;
    std::vector<double> weights;
    std::vector<double> sources;
};

Graph read_graph(std::istream& input, const ParseLimits& limits = {});
void write_graph(std::ostream& output, const Graph& graph);
InputBatch read_batch(std::istream& input, const ParseLimits& limits = {});
void write_batch(std::ostream& output, const InputBatch& batch);

}  // namespace bdg
