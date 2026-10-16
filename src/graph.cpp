#include "bdg/graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace bdg {

std::size_t checked_product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("Size product overflows size_t");
    return a * b;
}

std::size_t ceil_div(std::size_t n, std::size_t d) {
    if (!d) throw std::invalid_argument("Zero divisor in ceil_div");
    return n / d + (n % d != 0);
}

void Graph::validate() const {
    if (widths.empty() || widths.size() != layers.size() + 1)
        throw std::invalid_argument("Graph needs one more width than edge layers");
    if (!weight_count) throw std::invalid_argument("weight_count must be positive");
    if (outputs.empty()) throw std::invalid_argument("Graph needs at least one output");
    if (outputs.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Too many outputs");
    for (auto width : widths)
        if (!width) throw std::invalid_argument("Layer widths must be positive");
    for (std::size_t l = 0; l < layers.size(); ++l) {
        for (const auto& e : layers[l]) {
            if (e.parent >= widths[l] || e.child >= widths[l + 1] || e.weight >= weight_count)
                throw std::invalid_argument("Edge index outside graph bounds");
            if (!std::isfinite(e.coefficient))
                throw std::invalid_argument("Edge coefficient must be finite");
        }
    }
    for (const auto& out : outputs)
        if (out.layer >= widths.size() || out.node >= widths[out.layer])
            throw std::invalid_argument("Output outside graph bounds");
}

std::size_t Graph::edge_count() const {
    std::size_t count = 0;
    for (const auto& layer : layers) {
        if (layer.size() > std::numeric_limits<std::size_t>::max() - count)
            throw std::overflow_error("Edge count overflow");
        count += layer.size();
    }
    return count;
}

std::uint32_t Graph::max_width() const {
    if (widths.empty()) throw std::invalid_argument("Empty graph");
    return *std::max_element(widths.begin(), widths.end());
}

}  // namespace bdg
