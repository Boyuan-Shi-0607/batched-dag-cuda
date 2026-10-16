#include "bdg/cpu.hpp"
#include "bdg/cuda.hpp"
#include "bdg/sample_data.hpp"
#include "cos_sample.hpp"
#include "test_support.hpp"

#include <iostream>
#include <string>
#include <utility>

namespace {
template <class T> std::size_t test(bool quick) {
    std::size_t cases = 0;
    std::vector<bdg::Graph> graphs = {bdg::examples::diamond(), bdg::examples::cos_su2_sublattice(2), fan_in_graph()};
    if (!quick) {
        graphs.push_back(bdg::examples::cos_su2_sublattice(3));
        graphs.push_back({1, {2}, {}, {{0, 1}, {0, 0}}});
        graphs.push_back({1, {1, 2, 1}, {{}, {{1, 0, 0, 1}}}, {{1, 0}, {2, 0}}});
    }
    for (const auto& graph : graphs)
        for (auto backend : {bdg::CudaBackend::edge_atomic, bdg::CudaBackend::shared_partition, bdg::CudaBackend::gather})
            for (bool capture : {false, true})
                for (std::size_t batch : (quick ? std::vector<std::size_t>{13} : std::vector<std::size_t>{1, 7, 13, 33})) {
                    bdg::CudaOptions options;
                    options.backend = backend; options.cuda_graph = capture; options.threads = 128;
                    options.edges_per_partition = 385; options.samples_per_group = 4; options.shared_budget_bytes = 8192;
                    bdg::CudaExecutor<T> executor(graph, batch, options);
                    require_throws([&] { executor.launch(); }, "Launch without inputs accepted");
                    require_throws([&] { executor.download_outputs(); }, "Download without output accepted");
                    for (unsigned trial = 0; trial < 3; ++trial) {
                        auto weights = bdg::sample_weights<T>(batch * graph.weight_count, 19 + trial);
                        auto roots = bdg::sample_weights<T>(batch * graph.widths[0], 31 + trial);
                        if (trial) for (std::size_t i = 0; i < weights.size(); i += 3) weights[i] = -weights[i];
                        if (trial == 2) {
                            for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = T(int(i % 5) - 2);
                            for (std::size_t i = 0; i < roots.size(); ++i) roots[i] = T(1 + i % 2);
                        }
                        const double atol = trial == 2 ? 0 : sizeof(T) == 4 ? 1e-5 : 1e-11;
                        const double rtol = trial == 2 ? 0 : sizeof(T) == 4 ? 1e-4 : 1e-10;
                        const auto reference = bdg::evaluate_cpu(graph, batch, weights, roots);
                        executor.set_inputs(weights, roots);
                        for (int replay = 0; replay < 3; ++replay) executor.launch();
                        auto result = executor.download_outputs();
                        require(bdg::compare(result, reference, atol, rtol).passed,
                                "CUDA replay mismatch: " + std::string(bdg::backend_name(backend)));
                        const auto timing = executor.benchmark(2, 1);
                        require(timing.device_ms_per_iteration > 0, "Invalid CUDA event time");
                        require(bdg::compare(executor.download_outputs(), reference, atol, rtol).passed,
                                "Benchmark changed outputs");
                    }
                    ++cases;
                }
    return cases;
}
}  // namespace

int main(int argc, char** argv) {
    bool quick = false, required = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--quick") quick = true;
        else if (std::string(argv[i]) == "--require-device") required = true;
        else { std::cerr << "Unknown test option\n"; return 1; }
    }
    std::string reason;
    if (!bdg::cuda_device_available(0, &reason)) {
        std::cout << "SKIP: " << reason << '\n'; return required ? 1 : 77;
    }
    try {
        const auto cases = test<float>(quick) + test<double>(quick);
        std::cout << "PASS: " << cases << " CUDA cases; both precisions, three kernels, graph/stream replay\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
