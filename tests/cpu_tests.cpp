#include "bdg/cpu.hpp"
#include "bdg/io.hpp"
#include "bdg/plan.hpp"
#include "bdg/sample_data.hpp"
#include "cos_sample.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <tuple>

namespace {
std::vector<double> interpret_partitions(const bdg::Graph& g, const bdg::Plan& plan,
                                       std::size_t batch, const std::vector<double>& weights) {
    std::vector<double> current(batch * g.widths.front(), 1), output(batch * g.outputs.size(), 0);
    for (std::size_t l = 0; l < g.widths.size(); ++l) {
        for (auto readout : plan.readouts[l])
            for (std::size_t b = 0; b < batch; ++b)
                output[b * g.outputs.size() + readout.column] = current[b * g.widths[l] + readout.node];
        if (l == g.layers.size()) break;
        const auto& lp = plan.layers[l];
        std::vector<double> next(batch * g.widths[l + 1], 0);
        for (const auto& p : lp.partitions) {
            for (std::size_t b = 0; b < batch; ++b) {
                std::vector<double> local(p.child_count, 0);
                for (std::size_t j = 0; j < p.edge_count; ++j) {
                    const auto e = lp.packed_edges[p.edge_offset + j];
                    require(e.parent < p.parent_count && e.child < p.child_count && e.weight < p.weight_count &&
                            e.coefficient < p.coefficient_count, "Packed local index out of range");
                    const auto parent = lp.unique_parents[p.parent_offset + e.parent];
                    const auto weight = lp.unique_weights[p.weight_offset + e.weight];
                    local[e.child] += current[b * g.widths[l] + parent] * weights[b * g.weight_count + weight] *
                                      lp.unique_coefficients[p.coefficient_offset + e.coefficient];
                }
                for (std::size_t j = 0; j < p.child_count; ++j)
                    next[b * g.widths[l + 1] + lp.unique_children[p.child_offset + j]] += local[j];
            }
        }
        current.swap(next);
    }
    return output;
}

long long permutation_oracle(unsigned n, const std::vector<double>& weights) {
    std::vector<std::vector<unsigned>> perms;
    std::vector<unsigned> p(n); std::iota(p.begin(), p.end(), 0);
    do { perms.push_back(p); } while (std::next_permutation(p.begin(), p.end()));
    auto sign = [](const std::vector<unsigned>& p) {
        int parity = 0;
        for (unsigned i = 0; i < p.size(); ++i)
            for (unsigned j = i + 1; j < p.size(); ++j) parity ^= p[i] > p[j];
        return parity ? -1 : 1;
    };
    long long total = 0;
    for (const auto& up : perms) for (const auto& down : perms) {
        std::vector<bool> seen(n, false); std::vector<unsigned> pending{0}; seen[0] = true;
        while (!pending.empty()) {
            auto i = pending.back(); pending.pop_back();
            for (auto j : {up[i], down[i]}) if (!seen[j]) { seen[j] = true; pending.push_back(j); }
        }
        if (std::find(seen.begin(), seen.end(), false) != seen.end()) continue;
        for (unsigned sites = 0; sites < (1u << n); ++sites) {
            long long term = sign(up) * sign(down);
            for (unsigned spin = 0; spin < 2; ++spin) {
                const auto& permutation = spin ? down : up;
                for (unsigned i = 0; i < n; ++i) {
                    const unsigned j = permutation[i];
                    const auto row = 2 * (spin * n + i) + ((sites >> i) & 1);
                    const auto col = 2 * (spin * n + j) + ((sites >> j) & 1);
                    term *= static_cast<long long>(weights[row * (4 * n) + col]);
                }
            }
            total += term;
        }
    }
    return total;
}

void test_math_and_plan() {
    auto graph = bdg::examples::diamond();
    const auto actual = bdg::evaluate_cpu<double>(graph, 1, {3, 5, 7, 11}, {1, 2});
    require(actual == std::vector<double>({29, -185}), "Diamond hand calculation");
    std::ifstream batch_file(std::string(BDG_DATA_DIR) + "/diamond.batch");
    auto batch = bdg::read_batch(batch_file);
    require(bdg::evaluate_cpu(graph, batch.batch, batch.weights, batch.sources) ==
            std::vector<double>({29, -185, 1, -19}), "Batch fixture hand calculation");
    std::ifstream fixture_file(std::string(BDG_DATA_DIR) + "/cos_su2_n2.dag");
    std::ostringstream frozen, generated;
    bdg::write_graph(frozen, bdg::read_graph(fixture_file));
    bdg::write_graph(generated, bdg::examples::cos_su2_sublattice(2));
    require(frozen.str() == generated.str(), "Frozen CoS fixture differs from generator");
    for (unsigned n = 1; n <= 4; ++n) {
        auto cos = bdg::examples::cos_su2_sublattice(n);
        for (unsigned trial = 0; trial < 3; ++trial) {
            std::vector<double> w(cos.weight_count);
            std::uint32_t state = 1921 + n * 71 + trial;
            for (auto& x : w) { state = 1664525u * state + 1013904223u; x = static_cast<int>(state % 5) - 2; }
            require(bdg::evaluate_cpu(cos, 1, w)[0] == permutation_oracle(n, w), "CoS vs connected permutation/site oracle");
        }
    }
    for (auto g : {graph, fan_in_graph(), bdg::examples::cos_su2_sublattice(3)}) {
        auto weights = bdg::sample_weights<double>(13 * g.weight_count, 33);
        auto expected = bdg::evaluate_cpu(g, 13, weights);
        for (auto bytes : {4u, 8u}) for (auto cap : {1u, 7u, 385u}) {
            const auto budget = 512u;
            auto plan = bdg::compile_plan(g, cap, budget, bytes);
            for (std::size_t l = 0; l < plan.layers.size(); ++l) {
                const auto& layer = plan.layers[l];
                require(layer.row_offsets.back() == g.layers[l].size(), "CSR lost edges");
                std::map<std::tuple<unsigned, unsigned, unsigned, double>, unsigned> original, packed;
                for (auto e : g.layers[l]) ++original[{e.parent, e.child, e.weight, e.coefficient}];
                for (auto p : layer.partitions) {
                    require(p.edge_count <= cap && p.shared_bytes(bytes) <= budget, "Partition exceeds budget");
                    for (std::size_t j = 0; j < p.edge_count; ++j) {
                        auto e = layer.packed_edges[p.edge_offset + j];
                        ++packed[{layer.unique_parents[p.parent_offset + e.parent], layer.unique_children[p.child_offset + e.child],
                                  layer.unique_weights[p.weight_offset + e.weight], layer.unique_coefficients[p.coefficient_offset + e.coefficient]}];
                    }
                }
                require(original == packed, "Partition changed edge multiset");
            }
            require(bdg::compare(interpret_partitions(g, plan, 13, weights), expected, 1e-12, 1e-11).passed,
                    "CPU packed-plan interpreter differs");
        }
    }
    for (std::size_t batch_size : {1, 7, 12, 13, 31, 32, 33, 101})
        for (std::size_t group : {1, 4, 12, 32}) {
            std::vector<unsigned> visits(batch_size, 0);
            for (std::size_t g = 0; g < bdg::ceil_div(batch_size, group); ++g)
                for (std::size_t s = 0; s < group && g * group + s < batch_size; ++s) ++visits[g * group + s];
            require(std::all_of(visits.begin(), visits.end(), [](auto n) { return n == 1; }), "Batch tail coverage");
        }
}

void test_format_and_failures() {
    auto g = bdg::examples::diamond();
    std::ostringstream output; bdg::write_graph(output, g);
    std::istringstream input(output.str()); const auto parsed = bdg::read_graph(input);
    std::ostringstream again; bdg::write_graph(again, parsed);
    require(output.str() == again.str(), "Graph round trip");
    bdg::InputBatch batch{2, 4, 2, {1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 2, 3}};
    std::ostringstream batch_text; bdg::write_batch(batch_text, batch);
    std::istringstream batch_input(batch_text.str()); auto copy = bdg::read_batch(batch_input);
    require(copy.weights == batch.weights && copy.sources == batch.sources, "Batch round trip");
    std::istringstream trailing(output.str() + "garbage");
    require_throws([&] { bdg::read_graph(trailing); }, "Trailing data accepted");
    std::istringstream negative("BDAG 1 weights -1 layers 1 widths 1 outputs 1 0 0 edges 0 end");
    require_throws([&] { bdg::read_graph(negative); }, "Negative unsigned field accepted");
    std::istringstream truncated("BDAG 1 weights 4 layers 3 widths 2");
    require_throws([&] { bdg::read_graph(truncated); }, "Truncated graph accepted");
    auto limits = bdg::ParseLimits{}; limits.nodes = 4;
    std::istringstream too_wide(output.str());
    require_throws([&] { bdg::read_graph(too_wide, limits); }, "Excessive node count accepted");
    limits = {}; limits.weights = 3;
    std::istringstream too_many_weights(output.str());
    require_throws([&] { bdg::read_graph(too_many_weights, limits); }, "Excessive weight count accepted");
    auto invalid = g; invalid.layers[0][0].weight = 4;
    require_throws([&] { invalid.validate(); }, "Invalid weight index accepted");
    invalid = g; invalid.layers[0][0].coefficient = std::numeric_limits<double>::quiet_NaN();
    require_throws([&] { invalid.validate(); }, "NaN coefficient accepted");
    invalid = g; invalid.outputs[0].layer = 3;
    require_throws([&] { invalid.validate(); }, "Invalid output accepted");
    require_throws([&] { bdg::evaluate_cpu<double>(g, 0, {}); }, "Zero batch accepted");
    require_throws([&] { bdg::evaluate_cpu<double>(g, 1, {1, 2}); }, "Wrong weight size accepted");
    require_throws([&] { bdg::compile_plan(g, 0); }, "Zero partition cap accepted");
    require_throws([&] { bdg::compile_plan(g, 65536); }, "Oversized local indices accepted");
    require_throws([&] { bdg::checked_product(std::numeric_limits<std::size_t>::max(), 2); }, "Size overflow accepted");
    require(!bdg::compare(std::vector<double>{std::numeric_limits<double>::quiet_NaN()}, std::vector<double>{0}, 1, 1).passed,
            "NaN passed comparison");
    const bdg::Graph roots_only{1, {2}, {}, {{0, 1}, {0, 0}}};
    require(bdg::evaluate_cpu<double>(roots_only, 1, {4}, {7, 9}) == std::vector<double>({9, 7}), "Source readouts");
    const bdg::Graph empty_layer{1, {1, 2, 1}, {{}, {{1, 0, 0, 1}}}, {{1, 0}, {2, 0}}};
    require(bdg::evaluate_cpu<double>(empty_layer, 1, {4}) == std::vector<double>({0, 0}), "Empty-layer zeros");
}
}  // namespace

int main() {
    try {
        test_math_and_plan(); test_format_and_failures();
        std::cout << "PASS: CPU, packed-plan, format, boundary, and independent CoS order 1..4 checks\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
