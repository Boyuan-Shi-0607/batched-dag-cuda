#include "bdg/io.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>

namespace bdg {
namespace {

class Tokens {
public:
    explicit Tokens(std::istream& input) : input_(input) {}
    std::string next() {
        std::string value;
        while (input_ >> value) {
            if (value[0] == '#') { std::getline(input_, value); continue; }
            return value;
        }
        throw std::runtime_error("Unexpected end of input");
    }
    void expect(const std::string& value) {
        if (next() != value) throw std::runtime_error("Expected token: " + value);
    }
    std::size_t count(std::size_t maximum) {
        const auto token = next();
        std::uint64_t value = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || value > maximum)
            throw std::runtime_error("Invalid or excessive integer: " + token);
        return static_cast<std::size_t>(value);
    }
    double real() {
        std::istringstream number(next());
        number.imbue(std::locale::classic());
        double value = 0;
        if (!(number >> value) || number.peek() != std::char_traits<char>::eof() || !std::isfinite(value))
            throw std::runtime_error("Expected a finite floating-point value");
        return value;
    }
    void finish() {
        expect("end");
        std::string token;
        while (input_ >> token) {
            if (token[0] != '#') throw std::runtime_error("Unexpected data after end");
            std::getline(input_, token);
        }
        if (input_.bad()) throw std::runtime_error("Input read failure");
    }
private:
    std::istream& input_;
};

constexpr auto u32max = std::numeric_limits<std::uint32_t>::max();
}  // namespace

Graph read_graph(std::istream& input, const ParseLimits& limits) {
    Tokens tokens(input);
    tokens.expect("BDAG"); tokens.expect("1");
    Graph g;
    tokens.expect("weights"); g.weight_count = static_cast<std::uint32_t>(tokens.count(std::min<std::size_t>(u32max, limits.weights)));
    tokens.expect("layers"); const auto count = tokens.count(limits.layers);
    if (!count) throw std::runtime_error("A graph needs a source layer");
    tokens.expect("widths");
    std::size_t remaining_nodes = limits.nodes;
    for (std::size_t i = 0; i < count; ++i) {
        const auto width = tokens.count(std::min<std::size_t>(u32max, remaining_nodes));
        g.widths.push_back(static_cast<std::uint32_t>(width));
        remaining_nodes -= width;
    }
    g.layers.resize(count - 1);
    tokens.expect("outputs"); const auto outputs = tokens.count(limits.outputs);
    for (std::size_t i = 0; i < outputs; ++i)
        g.outputs.push_back({static_cast<std::uint32_t>(tokens.count(u32max)),
                             static_cast<std::uint32_t>(tokens.count(u32max))});
    tokens.expect("edges"); const auto edges = tokens.count(limits.edges);
    for (std::size_t i = 0; i < edges; ++i) {
        const auto layer = tokens.count(u32max);
        if (layer >= g.layers.size()) throw std::runtime_error("Edge layer outside graph");
        Edge e;
        e.parent = static_cast<std::uint32_t>(tokens.count(u32max));
        e.child = static_cast<std::uint32_t>(tokens.count(u32max));
        e.weight = static_cast<std::uint32_t>(tokens.count(u32max));
        e.coefficient = tokens.real();
        g.layers[layer].push_back(e);
    }
    tokens.finish(); g.validate();
    return g;
}

void write_graph(std::ostream& out, const Graph& g) {
    g.validate(); out.imbue(std::locale::classic());
    out << "BDAG 1\nweights " << g.weight_count << "\nlayers " << g.widths.size() << "\nwidths";
    for (auto width : g.widths) out << ' ' << width;
    out << "\noutputs " << g.outputs.size() << '\n';
    for (auto output : g.outputs) out << output.layer << ' ' << output.node << '\n';
    out << "edges " << g.edge_count() << '\n' << std::setprecision(17);
    for (std::size_t l = 0; l < g.layers.size(); ++l)
        for (auto e : g.layers[l])
            out << l << ' ' << e.parent << ' ' << e.child << ' ' << e.weight << ' ' << e.coefficient << '\n';
    out << "end\n";
    if (!out) throw std::runtime_error("Graph write failure");
}

InputBatch read_batch(std::istream& in, const ParseLimits& limits) {
    Tokens tokens(in);
    tokens.expect("BDAG_BATCH"); tokens.expect("1");
    InputBatch b;
    tokens.expect("batch"); b.batch = tokens.count(limits.batch_values);
    tokens.expect("weights"); b.weight_count = static_cast<std::uint32_t>(tokens.count(u32max));
    tokens.expect("sources"); b.source_width = static_cast<std::uint32_t>(tokens.count(u32max));
    if (!b.batch || !b.weight_count || !b.source_width) throw std::runtime_error("Zero batch dimension");
    const auto nw = checked_product(b.batch, b.weight_count), ns = checked_product(b.batch, b.source_width);
    if (nw > limits.batch_values || ns > limits.batch_values - nw)
        throw std::runtime_error("Batch exceeds parser value limit");
    tokens.expect("values"); b.weights.reserve(nw);
    for (std::size_t i = 0; i < nw; ++i) b.weights.push_back(tokens.real());
    tokens.expect("roots"); b.sources.reserve(ns);
    for (std::size_t i = 0; i < ns; ++i) b.sources.push_back(tokens.real());
    tokens.finish(); return b;
}

void write_batch(std::ostream& out, const InputBatch& b) {
    if (!b.batch || !b.weight_count || !b.source_width ||
        b.weights.size() != checked_product(b.batch, b.weight_count) ||
        b.sources.size() != checked_product(b.batch, b.source_width))
        throw std::invalid_argument("Invalid batch dimensions");
    for (double v : b.weights) if (!std::isfinite(v)) throw std::invalid_argument("Non-finite batch value");
    for (double v : b.sources) if (!std::isfinite(v)) throw std::invalid_argument("Non-finite batch value");
    out.imbue(std::locale::classic());
    out << "BDAG_BATCH 1\nbatch " << b.batch << "\nweights " << b.weight_count
        << "\nsources " << b.source_width << "\nvalues\n" << std::setprecision(17);
    for (std::size_t i = 0; i < b.weights.size(); ++i)
        out << b.weights[i] << ((i + 1) % b.weight_count ? ' ' : '\n');
    out << "roots\n";
    for (std::size_t i = 0; i < b.sources.size(); ++i)
        out << b.sources[i] << ((i + 1) % b.source_width ? ' ' : '\n');
    out << "end\n";
    if (!out) throw std::runtime_error("Batch write failure");
}

}  // namespace bdg
