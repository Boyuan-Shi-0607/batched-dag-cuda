#pragma once

#include "bdg/graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace bdg {

template <class T>
void validate_inputs(const Graph& graph, std::size_t batch, const std::vector<T>& weights,
                     const std::vector<T>& sources) {
    static_assert(std::is_floating_point<T>::value, "Floating-point scalar required");
    graph.validate();
    if (!batch) throw std::invalid_argument("Batch must be positive");
    if (weights.size() != checked_product(batch, graph.weight_count))
        throw std::invalid_argument("Weight buffer must have batch * weight_count entries");
    if (!sources.empty() && sources.size() != checked_product(batch, graph.widths.front()))
        throw std::invalid_argument("Source buffer must have batch * source_width entries");
    for (T x : weights)
        if (!std::isfinite(x)) throw std::invalid_argument("Non-finite input weight");
    for (T x : sources)
        if (!std::isfinite(x)) throw std::invalid_argument("Non-finite source value");
    for (const auto& layer : graph.layers)
        for (const auto& edge : layer)
            if (!std::isfinite(static_cast<T>(edge.coefficient)))
                throw std::invalid_argument("Coefficient overflows selected precision");
}

// Batch-major inputs and outputs. Empty sources means all source nodes equal 1.
template <class T>
std::vector<T> evaluate_cpu(const Graph& graph, std::size_t batch,
                            const std::vector<T>& weights,
                            const std::vector<T>& sources = {}) {
    validate_inputs(graph, batch, weights, sources);
    std::vector<T> result(checked_product(batch, graph.outputs.size()), T(0));
    std::vector<T> current(checked_product(batch, graph.widths.front()), T(1));
    if (!sources.empty()) current = sources;
    for (std::size_t layer = 0; layer < graph.widths.size(); ++layer) {
        for (std::size_t o = 0; o < graph.outputs.size(); ++o) {
            if (graph.outputs[o].layer != layer) continue;
            for (std::size_t b = 0; b < batch; ++b)
                result[b * graph.outputs.size() + o] =
                    current[b * graph.widths[layer] + graph.outputs[o].node];
        }
        if (layer == graph.layers.size()) break;
        std::vector<T> next(checked_product(batch, graph.widths[layer + 1]), T(0));
        for (std::size_t b = 0; b < batch; ++b)
            for (const auto& e : graph.layers[layer])
                next[b * graph.widths[layer + 1] + e.child] +=
                    (current[b * graph.widths[layer] + e.parent] *
                     weights[b * graph.weight_count + e.weight]) * static_cast<T>(e.coefficient);
        current.swap(next);
    }
    return result;
}

struct Comparison {
    bool passed = true;
    double max_absolute_error = 0;
    double max_relative_error = 0;
    std::size_t worst_absolute_index = 0;
};

template <class A, class B>
Comparison compare(const std::vector<A>& actual, const std::vector<B>& expected,
                   double atol, double rtol) {
    if (actual.size() != expected.size()) throw std::invalid_argument("Output size mismatch");
    if (!(atol >= 0) || !(rtol >= 0) || !std::isfinite(atol) || !std::isfinite(rtol))
        throw std::invalid_argument("Invalid comparison tolerance");
    Comparison c;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const long double a = actual[i], b = expected[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            c.passed = false;
            c.max_absolute_error = c.max_relative_error = std::numeric_limits<double>::infinity();
            c.worst_absolute_index = i;
            continue;
        }
        const auto error = std::abs(a - b);
        if (error > c.max_absolute_error) {
            c.max_absolute_error = static_cast<double>(error);
            c.worst_absolute_index = i;
        }
        const auto relative = std::abs(b) > 0 ? error / std::abs(b) : error;
        c.max_relative_error = std::max(c.max_relative_error, static_cast<double>(relative));
        if (error > atol + rtol * std::abs(b)) c.passed = false;
    }
    return c;
}

}  // namespace bdg
