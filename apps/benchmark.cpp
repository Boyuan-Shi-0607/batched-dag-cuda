#include "bdg/cpu.hpp"
#include "bdg/cuda.hpp"
#include "bdg/io.hpp"
#include "bdg/sample_data.hpp"

#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
struct Options {
    std::string graph, input, json, outputs, backend = "cpu", precision = "double";
    std::size_t batch = 13, iterations = 20, warmup = 3;
    std::uint32_t seed = 1729;
    double atol = -1, rtol = -1;
    bool batch_explicit = false;
    bdg::CudaOptions cuda;
};

std::size_t integer(const std::string& text) {
    std::size_t value = 0;
    auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
        throw std::invalid_argument("Invalid integer: " + text);
    return value;
}
unsigned small_integer(const std::string& text) {
    auto value = integer(text);
    if (value > std::numeric_limits<unsigned>::max()) throw std::invalid_argument("Integer too large");
    return static_cast<unsigned>(value);
}
double real(const std::string& text) {
    std::size_t consumed = 0;
    double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value) || value < 0)
        throw std::invalid_argument("Invalid nonnegative tolerance");
    return value;
}

void help() {
    std::cout <<
        "bdg_benchmark --graph FILE [options]\n\n"
        "  --backend cpu|cuda-edge|cuda-shared|cuda-gather   (default: cpu)\n"
        "  --precision float|double                       (default: double)\n"
        "  --input FILE        BDAG_BATCH input; otherwise generate synthetic weights\n"
        "  --batch N           number of samples (default: 13)\n"
        "  --seed N            synthetic input seed (default: 1729)\n"
        "  --iterations N      timed independent evaluations (default: 20)\n"
        "  --warmup N          warmup evaluations (default: 3)\n"
        "  --threads N         CUDA block threads, multiple of 32 (default: 256)\n"
        "  --partition-edges N shared backend edge cap (default: 1024)\n"
        "  --shared-bytes N    shared backend resource budget (default: 32768)\n"
        "  --group-size N      samples per shared block group (default: 4)\n"
        "  --no-cuda-graph     use ordinary stream launches\n"
        "  --device N          CUDA device index (default: 0)\n"
        "  --atol X --rtol X   output validation tolerances\n"
        "  --json FILE         write machine-readable verified metrics\n"
        "  --outputs FILE      write output values as CSV\n\n"
        "GPU timing includes root reset, child initialization, kernels and readouts.\n"
        "Compilation, input upload and output download are outside GPU event timing.\n"
        "All outputs are checked against a long-double CPU reference before and after timing.\n";
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--no-cuda-graph") { o.cuda.cuda_graph = false; continue; }
        if (++i >= argc) throw std::invalid_argument("Missing value for " + key);
        const std::string value = argv[i];
        if (key == "--graph") o.graph = value;
        else if (key == "--input") o.input = value;
        else if (key == "--json") o.json = value;
        else if (key == "--outputs") o.outputs = value;
        else if (key == "--backend") o.backend = value;
        else if (key == "--precision") o.precision = value;
        else if (key == "--batch") { o.batch = integer(value); o.batch_explicit = true; }
        else if (key == "--iterations") o.iterations = integer(value);
        else if (key == "--warmup") o.warmup = integer(value);
        else if (key == "--seed") o.seed = small_integer(value);
        else if (key == "--threads") o.cuda.threads = small_integer(value);
        else if (key == "--partition-edges") o.cuda.edges_per_partition = small_integer(value);
        else if (key == "--shared-bytes") o.cuda.shared_budget_bytes = integer(value);
        else if (key == "--group-size") o.cuda.samples_per_group = small_integer(value);
        else if (key == "--device") {
            auto device = integer(value);
            if (device > std::numeric_limits<int>::max()) throw std::invalid_argument("Device index too large");
            o.cuda.device = static_cast<int>(device);
        } else if (key == "--atol") o.atol = real(value);
        else if (key == "--rtol") o.rtol = real(value);
        else throw std::invalid_argument("Unknown option: " + key);
    }
    if (o.graph.empty() || !o.batch || !o.iterations)
        throw std::invalid_argument("--graph, a positive batch, and positive iterations are required");
    if (o.precision != "float" && o.precision != "double") throw std::invalid_argument("Unknown precision");
    if (o.backend == "cuda-edge") o.cuda.backend = bdg::CudaBackend::edge_atomic;
    else if (o.backend == "cuda-shared") o.cuda.backend = bdg::CudaBackend::shared_partition;
    else if (o.backend == "cuda-gather") o.cuda.backend = bdg::CudaBackend::gather;
    else if (o.backend != "cpu") throw std::invalid_argument("Unknown backend");
    return o;
}

std::string quoted(const std::string& text) {
    std::ostringstream out; out << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else out << c;
    }
    out << '"'; return out.str();
}

std::string json_real(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream out;
    out << std::setprecision(12) << value;
    return out.str();
}

template <class T> int run(Options o) {
    std::ifstream input(o.graph);
    if (!input) throw std::runtime_error("Cannot open graph: " + o.graph);
    const auto graph = bdg::read_graph(input);
    std::vector<T> weights, sources;
    if (!o.input.empty()) {
        std::ifstream file(o.input);
        if (!file) throw std::runtime_error("Cannot open batch: " + o.input);
        const auto batch = bdg::read_batch(file);
        if (batch.weight_count != graph.weight_count || batch.source_width != graph.widths.front() ||
            (o.batch_explicit && batch.batch != o.batch)) throw std::invalid_argument("Batch file dimensions do not match");
        o.batch = batch.batch;
        weights.assign(batch.weights.begin(), batch.weights.end());
        sources.assign(batch.sources.begin(), batch.sources.end());
    } else weights = bdg::sample_weights<T>(bdg::checked_product(o.batch, graph.weight_count), o.seed);
    bdg::validate_inputs(graph, o.batch, weights, sources);
    if (o.atol < 0) o.atol = sizeof(T) == 4 ? 1e-5 : 1e-11;
    if (o.rtol < 0) o.rtol = sizeof(T) == 4 ? 1e-4 : 1e-9;
    auto reference_graph = graph;
    for (auto& layer : reference_graph.layers)
        for (auto& edge : layer) edge.coefficient = static_cast<T>(edge.coefficient);
    const auto reference = bdg::evaluate_cpu<long double>(reference_graph, o.batch,
        std::vector<long double>(weights.begin(), weights.end()),
        std::vector<long double>(sources.begin(), sources.end()));
    bdg::DeviceInfo info;
    bdg::Timing timing;
    double setup_ms = 0, upload_ms = 0;
    std::vector<T> actual;
    std::unique_ptr<bdg::CudaExecutor<T>> gpu;
    if (o.backend != "cpu") {
        const auto begin = std::chrono::steady_clock::now();
        gpu = std::make_unique<bdg::CudaExecutor<T>>(graph, o.batch, o.cuda);
        const auto built = std::chrono::steady_clock::now();
        gpu->set_inputs(weights, sources);
        const auto uploaded = std::chrono::steady_clock::now();
        setup_ms = std::chrono::duration<double, std::milli>(built - begin).count();
        upload_ms = std::chrono::duration<double, std::milli>(uploaded - built).count();
        gpu->launch(); actual = gpu->download_outputs(); info = gpu->device_info();
    } else actual = bdg::evaluate_cpu(graph, o.batch, weights, sources);
    auto validation = bdg::compare(actual, reference, o.atol, o.rtol);
    if (!validation.passed) throw std::runtime_error("Pre-benchmark output validation failed; throughput not reported");
    if (gpu) {
        timing = gpu->benchmark(o.iterations, o.warmup);
        actual = gpu->download_outputs();
    } else {
        for (std::size_t i = 0; i < o.warmup; ++i) actual = bdg::evaluate_cpu(graph, o.batch, weights, sources);
        volatile double guard = 0;
        const auto begin = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < o.iterations; ++i) {
            actual = bdg::evaluate_cpu(graph, o.batch, weights, sources);
            guard += static_cast<double>(actual.front());
        }
        (void)guard;
        timing.host_ms_per_iteration = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() / o.iterations;
    }
    validation = bdg::compare(actual, reference, o.atol, o.rtol);
    if (!validation.passed) throw std::runtime_error("Post-replay output validation failed; throughput not reported");
    const double milliseconds = gpu ? timing.device_ms_per_iteration : timing.host_ms_per_iteration;
    if (!(milliseconds > 0)) throw std::runtime_error("Timer resolution insufficient; increase iterations");
    std::ostringstream metrics;
    metrics << std::setprecision(12)
        << "{\n  \"version\": " << quoted(BDG_VERSION)
        << ",\n  \"backend\": " << quoted(o.backend)
        << ",\n  \"precision\": " << quoted(o.precision)
        << ",\n  \"verified\": true,\n  \"batch\": " << o.batch
        << ",\n  \"edge_count\": " << graph.edge_count()
        << ",\n  \"outputs_per_sample\": " << graph.outputs.size()
        << ",\n  \"iterations\": " << o.iterations << ",\n  \"warmup\": " << o.warmup
        << ",\n  \"seed\": " << o.seed << ",\n  \"graph_file\": " << quoted(o.graph)
        << ",\n  \"input_file\": " << quoted(o.input)
        << ",\n  \"cuda_graph\": " << (gpu && o.cuda.cuda_graph ? "true" : "false")
        << ",\n  \"host_ms_per_iteration\": " << timing.host_ms_per_iteration
        << ",\n  \"device_ms_per_iteration\": ";
    if (gpu) metrics << timing.device_ms_per_iteration; else metrics << "null";
    metrics << ",\n  \"samples_per_second\": " << o.batch * 1000.0 / milliseconds
        << ",\n  \"logical_gedges_per_second\": " << static_cast<double>(graph.edge_count()) * o.batch / (milliseconds * 1e6)
        << ",\n  \"setup_ms\": " << setup_ms << ",\n  \"upload_ms\": " << upload_ms
        << ",\n  \"includes_reset_and_readouts\": true,\n  \"includes_h2d_d2h\": false"
        << ",\n  \"atol\": " << o.atol << ",\n  \"rtol\": " << o.rtol
        << ",\n  \"max_absolute_error\": " << json_real(validation.max_absolute_error)
        << ",\n  \"max_relative_error\": " << json_real(validation.max_relative_error)
        << ",\n  \"device_name\": " << quoted(info.name)
        << ",\n  \"compute_capability\": " << quoted(std::to_string(info.major) + "." + std::to_string(info.minor))
        << ",\n  \"cuda_runtime\": " << info.runtime_version << ",\n  \"cuda_driver\": " << info.driver_version
        << ",\n  \"threads\": " << o.cuda.threads << ",\n  \"group_size\": " << o.cuda.samples_per_group
        << ",\n  \"partition_edge_cap_requested\": " << o.cuda.edges_per_partition
        << ",\n  \"shared_budget_requested\": " << o.cuda.shared_budget_bytes
        << ",\n  \"partitions\": " << info.partitions << ",\n  \"max_dynamic_shared_bytes\": " << info.max_dynamic_shared_bytes
        << ",\n  \"numerical_workspace_bytes\": " << info.workspace_bytes << "\n}\n";
    std::cout << metrics.str();
    if (!o.json.empty()) {
        std::ofstream file(o.json);
        if (!(file << metrics.str())) throw std::runtime_error("Cannot write metrics: " + o.json);
    }
    if (!o.outputs.empty()) {
        std::ofstream file(o.outputs);
        file << "sample,output,layer,node,value\n" << std::setprecision(std::numeric_limits<T>::max_digits10);
        for (std::size_t b = 0; b < o.batch; ++b)
            for (std::size_t j = 0; j < graph.outputs.size(); ++j)
                file << b << ',' << j << ',' << graph.outputs[j].layer << ',' << graph.outputs[j].node
                     << ',' << actual[b * graph.outputs.size() + j] << '\n';
        if (!file) throw std::runtime_error("Cannot write outputs: " + o.outputs);
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--help") { help(); return 0; }
        const auto options = parse(argc, argv);
        return options.precision == "float" ? run<float>(options) : run<double>(options);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'; return 1;
    }
}
