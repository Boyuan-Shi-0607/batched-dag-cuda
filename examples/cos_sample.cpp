#include "cos_sample.hpp"

#include <map>
#include <stdexcept>
#include <tuple>

namespace bdg::examples {
namespace {
struct State {
    unsigned head, head_site, end, end_site;
    std::vector<unsigned char> mask;  // 0 absent, 1/2 open at A/B, 3 complete.
    bool operator<(const State& other) const {
        return std::tie(head, head_site, end, end_site, mask) <
               std::tie(other.head, other.head_site, other.end, other.end_site, other.mask);
    }
};

std::uint32_t weight_index(unsigned n, unsigned i, unsigned si, unsigned j, unsigned sj) {
    return (2 * i + si) * (4 * n) + 2 * j + sj;
}
}  // namespace

Graph diamond() {
    return {4, {2, 2, 1}, {{{0, 0, 0, 1}, {1, 0, 1, -1}, {0, 1, 2, 1}, {1, 1, 3, 1}},
                           {{0, 0, 2, 2}, {1, 0, 0, -1}}}, {{1, 1}, {2, 0}}};
}

Graph cos_su2_sublattice(unsigned n) {
    if (!n || n > 8) throw std::invalid_argument("CoS sample order must be in [1,8]");
    Graph graph;
    graph.weight_count = 16 * n * n;
    graph.widths.push_back(2);
    std::vector<State> current;
    for (unsigned site = 0; site < 2; ++site) {
        State state{0, site, 0, site, std::vector<unsigned char>(n, 0)};
        state.mask[0] = static_cast<unsigned char>(site + 1);
        current.push_back(state);
    }
    std::size_t total_nodes = 2, total_edges = 0;
    for (unsigned visited = 1; visited < 2 * n; ++visited) {
        std::map<State, std::uint32_t> ids;
        std::vector<State> next;
        std::vector<Edge> edges;
        auto append = [&](std::uint32_t source, State state, std::uint32_t weight, double coefficient) {
            auto it = ids.find(state);
            std::uint32_t id;
            if (it == ids.end()) {
                if (++total_nodes > 2000000) throw std::runtime_error("CoS sample node limit exceeded");
                id = static_cast<std::uint32_t>(next.size());
                ids.emplace(state, id); next.push_back(std::move(state));
            } else id = it->second;
            if (++total_edges > 20000000) throw std::runtime_error("CoS sample edge limit exceeded");
            edges.push_back({source, id, weight, coefficient});
        };
        for (std::size_t source = 0; source < current.size(); ++source) {
            const State& s = current[source];
            for (unsigned w = 0; w < 2 * n; ++w) {
                if ((s.end < n) != (w < n) || w == s.head || s.mask[w % n] == 3) continue;
                const auto status = s.mask[w % n];
                for (unsigned site = 0; site < 2; ++site) {
                    if (status && site != static_cast<unsigned>(status - 1)) continue;
                    State child = s;
                    child.end = w; child.end_site = site;
                    child.mask[w % n] = static_cast<unsigned char>(status ? 3 : site + 1);
                    append(static_cast<std::uint32_t>(source), std::move(child),
                           weight_index(n, s.end, s.end_site, w, site), 1);
                }
            }
            unsigned open = 0;
            while (open < n && (s.mask[open] == 0 || s.mask[open] == 3)) ++open;
            if (open == n) continue;
            for (unsigned head : {open, open + n}) {
                if (head == s.head) continue;
                State child = s;
                child.head = child.end = head;
                child.head_site = child.end_site = s.mask[open] - 1;
                child.mask[open] = 3;
                append(static_cast<std::uint32_t>(source), std::move(child),
                       weight_index(n, s.end, s.end_site, s.head, s.head_site), -1);
            }
        }
        graph.layers.push_back(std::move(edges));
        graph.widths.push_back(static_cast<std::uint32_t>(next.size()));
        current.swap(next);
    }
    std::vector<Edge> closing;
    for (std::size_t i = 0; i < current.size(); ++i) {
        const auto& s = current[i];
        closing.push_back({static_cast<std::uint32_t>(i), 0,
                           weight_index(n, s.end, s.end_site, s.head, s.head_site), -1});
    }
    graph.layers.push_back(std::move(closing)); graph.widths.push_back(1);
    graph.outputs.push_back({2 * n, 0});
    graph.validate(); return graph;
}

}  // namespace bdg::examples
