#include "bdg/plan.hpp"

#include <algorithm>
#include <deque>
#include <map>
#include <numeric>
#include <stdexcept>

namespace bdg {

std::size_t Partition::shared_bytes(std::size_t scalar_bytes) const {
    return checked_product(static_cast<std::size_t>(parent_count) + child_count +
                           weight_count + coefficient_count, scalar_bytes);
}

namespace {
template <class T>
std::uint16_t intern(T value, std::map<T, std::uint16_t>& map, std::vector<T>& values) {
    auto it = map.find(value);
    if (it != map.end()) return it->second;
    if (values.size() >= 65536) throw std::overflow_error("Partition-local index exceeds uint16");
    auto index = static_cast<std::uint16_t>(values.size());
    values.push_back(value); map.emplace(value, index);
    return index;
}

std::vector<std::uint16_t> sort_ids(std::vector<std::uint32_t>& values) {
    std::vector<std::size_t> permutation(values.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin(), permutation.end(), [&](auto a, auto b) { return values[a] < values[b]; });
    std::vector<std::uint16_t> inverse(values.size());
    std::vector<std::uint32_t> sorted(values.size());
    for (std::size_t i = 0; i < permutation.size(); ++i) {
        inverse[permutation[i]] = static_cast<std::uint16_t>(i);
        sorted[i] = values[permutation[i]];
    }
    values.swap(sorted);
    return inverse;
}
}  // namespace

Plan compile_plan(const Graph& g, std::size_t max_edges, std::size_t shared_budget,
                  std::size_t scalar_bytes) {
    g.validate();
    if (!max_edges || max_edges > 65535 || (scalar_bytes != 4 && scalar_bytes != 8) || shared_budget < 4 * scalar_bytes)
        throw std::invalid_argument("Invalid partition budget, edge cap, or scalar size");
    Plan plan;
    plan.readouts.resize(g.widths.size());
    for (std::size_t i = 0; i < g.outputs.size(); ++i)
        plan.readouts[g.outputs[i].layer].push_back({g.outputs[i].node, static_cast<std::uint32_t>(i)});
    for (std::size_t l = 0; l < g.layers.size(); ++l) {
        LayerPlan lp;
        auto edges = g.layers[l];
        std::stable_sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) { return a.child < b.child; });
        lp.row_offsets.assign(static_cast<std::size_t>(g.widths[l + 1]) + 1, 0);
        for (auto e : edges) {
            ++lp.row_offsets[e.child + 1];
            lp.parents.push_back(e.parent); lp.children.push_back(e.child);
            lp.weights.push_back(e.weight); lp.coefficients.push_back(e.coefficient);
        }
        for (std::size_t i = 1; i < lp.row_offsets.size(); ++i) lp.row_offsets[i] += lp.row_offsets[i - 1];
        std::size_t offset = 0;
        while (offset < edges.size()) {
            std::map<std::uint32_t, std::uint16_t> pm, cm, wm;
            std::map<double, std::uint16_t> km;
            std::vector<std::uint32_t> pv, cv, wv;
            std::vector<double> kv;
            std::vector<PackedEdge> pe;
            while (offset < edges.size() && pe.size() < max_edges) {
                const auto e = edges[offset];
                const auto count = pv.size() + cv.size() + wv.size() + kv.size() +
                    !pm.count(e.parent) + !cm.count(e.child) + !wm.count(e.weight) + !km.count(e.coefficient);
                if (checked_product(count, scalar_bytes) > shared_budget) break;
                pe.push_back({intern(e.parent, pm, pv), intern(e.child, cm, cv),
                              intern(e.weight, wm, wv), intern(e.coefficient, km, kv)});
                ++offset;
            }
            if (pe.empty()) throw std::invalid_argument("Partition budget cannot fit one edge");
            const auto pi = sort_ids(pv), ci = sort_ids(cv), wi = sort_ids(wv);
            for (auto& e : pe) { e.parent = pi[e.parent]; e.child = ci[e.child]; e.weight = wi[e.weight]; }
            // Round-robin child groups spreads same-address shared atomics.
            std::vector<std::deque<PackedEdge>> groups(cv.size());
            for (auto e : pe) groups[e.child].push_back(e);
            std::deque<std::size_t> active;
            for (std::size_t i = 0; i < groups.size(); ++i) if (!groups[i].empty()) active.push_back(i);
            Partition part{lp.packed_edges.size(), lp.unique_parents.size(), lp.unique_children.size(),
                           lp.unique_weights.size(), lp.unique_coefficients.size(),
                           static_cast<std::uint32_t>(pe.size()), static_cast<std::uint32_t>(pv.size()),
                           static_cast<std::uint32_t>(cv.size()), static_cast<std::uint32_t>(wv.size()),
                           static_cast<std::uint32_t>(kv.size())};
            while (!active.empty()) {
                const auto i = active.front(); active.pop_front();
                lp.packed_edges.push_back(groups[i].front()); groups[i].pop_front();
                if (!groups[i].empty()) active.push_back(i);
            }
            lp.unique_parents.insert(lp.unique_parents.end(), pv.begin(), pv.end());
            lp.unique_children.insert(lp.unique_children.end(), cv.begin(), cv.end());
            lp.unique_weights.insert(lp.unique_weights.end(), wv.begin(), wv.end());
            lp.unique_coefficients.insert(lp.unique_coefficients.end(), kv.begin(), kv.end());
            lp.partitions.push_back(part);
            lp.max_shared_bytes = std::max(lp.max_shared_bytes, part.shared_bytes(scalar_bytes));
        }
        plan.layers.push_back(std::move(lp));
    }
    return plan;
}

}  // namespace bdg
