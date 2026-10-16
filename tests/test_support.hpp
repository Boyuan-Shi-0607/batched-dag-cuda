#pragma once

#include "bdg/graph.hpp"
#include <stdexcept>
#include <string>

inline void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Function> void require_throws(Function function, const std::string& message) {
    bool threw = false;
    try { function(); } catch (const std::exception&) { threw = true; }
    require(threw, message);
}

inline bdg::Graph fan_in_graph() {
    bdg::Graph graph{17, {193, 8, 3}, {}, {{0, 5}, {1, 0}, {1, 7}, {2, 0}, {2, 2}, {1, 0}}};
    graph.layers.resize(2);
    for (unsigned child = 0; child < 7; ++child)
        for (unsigned parent = 0; parent < 193; ++parent)
            graph.layers[0].push_back({parent, child, (parent + child) % 17, parent % 3 ? 1.0 : -0.5});
    graph.layers[1] = {{0, 0, 1, 1}, {7, 0, 2, -1}, {1, 2, 3, 2}, {1, 2, 3, -1}};
    return graph;
}
