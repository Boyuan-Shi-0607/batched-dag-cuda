#include "bdg/io.hpp"
#include "cos_sample.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        unsigned order = 2;
        std::string output;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::cout << "bdg_make_cos_fixture [--order 1..8] [--output file.dag]\n"
                             "Without --output, the graph is written to stdout.\n";
                return 0;
            }
            if (++i == argc) throw std::invalid_argument("Missing option value");
            if (key == "--order") {
                const std::string value = argv[i]; std::size_t consumed = 0;
                const auto parsed = std::stoul(value, &consumed);
                if (consumed != value.size() || parsed < 1 || parsed > 8) throw std::invalid_argument("Invalid order");
                order = static_cast<unsigned>(parsed);
            } else if (key == "--output") output = argv[i];
            else throw std::invalid_argument("Unknown option: " + key);
        }
        const auto graph = bdg::examples::cos_su2_sublattice(order);
        if (output.empty()) bdg::write_graph(std::cout, graph);
        else {
            std::ofstream file(output);
            if (!file) throw std::runtime_error("Cannot open output: " + output);
            bdg::write_graph(file, graph);
        }
        std::cerr << "order=" << order << " layers=" << graph.widths.size()
                  << " edges=" << graph.edge_count() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'; return 1;
    }
}
