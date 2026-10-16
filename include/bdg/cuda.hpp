#pragma once

#include "bdg/graph.hpp"
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace bdg {

enum class CudaBackend { edge_atomic, shared_partition, gather };
const char* backend_name(CudaBackend backend);
bool cuda_device_available(int device = 0, std::string* reason = nullptr) noexcept;

struct CudaOptions {
    CudaBackend backend = CudaBackend::gather;
    unsigned threads = 256;
    unsigned edges_per_partition = 1024;
    unsigned samples_per_group = 4;
    std::size_t shared_budget_bytes = 32768;
    bool cuda_graph = true;
    int device = 0;
};

struct DeviceInfo {
    std::string name;
    int major = 0, minor = 0, multiprocessors = 0;
    int runtime_version = 0, driver_version = 0;
    std::size_t workspace_bytes = 0;
    std::size_t partitions = 0;
    std::size_t max_dynamic_shared_bytes = 0;
};

struct Timing {
    double device_ms_per_iteration = 0;
    double host_ms_per_iteration = 0;
};

// Owns its stream and buffers. Not thread-safe; distinct instances are independent.
template <class T>
class CudaExecutor {
    static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                  "CudaExecutor supports float and double");
public:
    CudaExecutor(const Graph& graph, std::size_t batch, CudaOptions options = {});
    ~CudaExecutor();
    CudaExecutor(CudaExecutor&&) noexcept;
    CudaExecutor& operator=(CudaExecutor&&) noexcept;
    CudaExecutor(const CudaExecutor&) = delete;
    CudaExecutor& operator=(const CudaExecutor&) = delete;

    void set_inputs(const std::vector<T>& weights, const std::vector<T>& sources = {});
    void launch();
    void synchronize();
    std::vector<T> download_outputs();
    Timing benchmark(std::size_t iterations, std::size_t warmup = 5);
    const DeviceInfo& device_info() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

extern template class CudaExecutor<float>;
extern template class CudaExecutor<double>;

}  // namespace bdg
