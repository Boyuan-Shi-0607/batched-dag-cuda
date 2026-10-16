#include "bdg/cpu.hpp"
#include "bdg/cuda.hpp"
#include "bdg/io.hpp"
#include "bdg/sample_data.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Usage: bdg_evaluate graph.dag (requires CUDA device)");
        std::ifstream file(argv[1]);
        if (!file) throw std::runtime_error("Cannot open graph");
        const auto graph = bdg::read_graph(file);
        constexpr std::size_t batch = 13;
        bdg::CudaExecutor<double> executor(graph, batch);
        for (auto seed : {1729u, 1730u}) {
            auto weights = bdg::sample_weights<double>(bdg::checked_product(batch, graph.weight_count), seed);
            executor.set_inputs(weights);
            executor.launch();
            const auto result = executor.download_outputs();
            const auto expected = bdg::evaluate_cpu(graph, batch, weights);
            if (!bdg::compare(result, expected, 1e-11, 1e-9).passed)
                throw std::runtime_error("Output differs from CPU reference");
            std::cout << "seed=" << seed << " first_output=" << result.front() << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
