#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bdg {

// Reproducible synthetic inputs, independent of standard-library distributions.
template <class T>
std::vector<T> sample_weights(std::size_t count, std::uint32_t seed) {
    std::vector<T> result(count);
    for (auto& value : result) {
        seed = seed * 1664525u + 1013904223u;
        value = static_cast<T>(0.01 + 0.19 * static_cast<double>(seed >> 8) / 16777216.0);
    }
    return result;
}

}  // namespace bdg
