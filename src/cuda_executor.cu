#include "bdg/cuda.hpp"
#include "bdg/cpu.hpp"
#include "bdg/plan.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace bdg {
namespace {

void checked(cudaError_t status, const char* operation, int line) {
    if (status == cudaSuccess) return;
    std::ostringstream message;
    message << operation << " failed at cuda_executor.cu:" << line << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}
#define BDG_CUDA(call) checked((call), #call, __LINE__)

template <class T> class Buffer {
public:
    T* ptr = nullptr;
    std::size_t size = 0;
    Buffer() = default;
    ~Buffer() { if (ptr) cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept : ptr(other.ptr), size(other.size) { other.ptr = nullptr; other.size = 0; }
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            if (ptr) cudaFree(ptr);
            ptr = other.ptr; size = other.size; other.ptr = nullptr; other.size = 0;
        }
        return *this;
    }
    void allocate(std::size_t count) {
        if (ptr) throw std::logic_error("Buffer already allocated");
        if (count) BDG_CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr), checked_product(count, sizeof(T))));
        size = count;
    }
    void upload(const std::vector<T>& values) {
        allocate(values.size());
        if (size) BDG_CUDA(cudaMemcpy(ptr, values.data(), checked_product(size, sizeof(T)), cudaMemcpyHostToDevice));
    }
};

struct Stream {
    cudaStream_t handle = nullptr;
    ~Stream() { if (handle) cudaStreamDestroy(handle); }
};
struct GraphHandle {
    cudaGraph_t handle = nullptr;
    ~GraphHandle() { if (handle) cudaGraphDestroy(handle); }
};
struct Executable {
    cudaGraphExec_t handle = nullptr;
    ~Executable() { if (handle) cudaGraphExecDestroy(handle); }
};
struct Event {
    cudaEvent_t handle = nullptr;
    Event() { BDG_CUDA(cudaEventCreate(&handle)); }
    ~Event() { if (handle) cudaEventDestroy(handle); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

template <class T>
__global__ void edge_kernel(const std::uint32_t* parents, const std::uint32_t* children,
                            const std::uint32_t* indices, const T* coefficients,
                            std::uint64_t edges, std::uint64_t batch, const T* weights,
                            std::uint32_t weight_count, const T* input, T* output,
                            std::uint32_t in_width, std::uint32_t out_width) {
    const auto stride = std::uint64_t(gridDim.x) * blockDim.x;
    for (auto item = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         item < batch * edges; item += stride) {
        const auto b = item / edges, e = item % edges;
        const T value = (input[b * in_width + parents[e]] * weights[b * weight_count + indices[e]]) * coefficients[e];
        atomicAdd(output + b * out_width + children[e], value);
    }
}

template <class T>
__global__ void gather_kernel(const std::uint64_t* offsets, const std::uint32_t* parents,
                              const std::uint32_t* indices, const T* coefficients,
                              std::uint64_t batch, const T* weights, std::uint32_t weight_count,
                              const T* input, T* output, std::uint32_t in_width, std::uint32_t out_width) {
    const unsigned lane = threadIdx.x & 31;
    const auto warp_stride = std::uint64_t(gridDim.x) * (blockDim.x / 32);
    for (auto item = (std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
         item < batch * out_width; item += warp_stride) {
        const auto b = item / out_width;
        const auto child = item % out_width;
        T sum = 0;
        for (auto e = offsets[child] + lane; e < offsets[child + 1]; e += 32)
            sum += (input[b * in_width + parents[e]] * weights[b * weight_count + indices[e]]) * coefficients[e];
        for (unsigned delta = 16; delta; delta >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, delta);
        if (!lane) output[b * out_width + child] = sum;
    }
}

constexpr unsigned edges_per_thread = 4;

template <class T>
__global__ void shared_kernel(const PackedEdge* edges, const Partition* partitions,
                              std::uint64_t partition_count, const std::uint32_t* parents,
                              const std::uint32_t* children, const std::uint32_t* indices,
                              const T* coefficients, const T* weights, std::uint32_t weight_count,
                              const T* input, T* output, std::uint32_t in_width, std::uint32_t out_width,
                              std::uint64_t batch, unsigned samples_per_group, std::uint64_t groups) {
    extern __shared__ __align__(16) unsigned char storage[];
    for (auto task = std::uint64_t(blockIdx.x); task < groups * partition_count; task += gridDim.x) {
        const auto group = task / partition_count;
        const Partition part = partitions[task % partition_count];
        T* p = reinterpret_cast<T*>(storage);
        T* c = p + part.parent_count;
        T* w = c + part.child_count;
        T* k = w + part.weight_count;
        PackedEdge cached[edges_per_thread];
        #pragma unroll
        for (unsigned slot = 0; slot < edges_per_thread; ++slot) {
            const auto i = threadIdx.x + slot * blockDim.x;
            if (i < part.edge_count) cached[slot] = edges[part.edge_offset + i];
        }
        for (auto i = threadIdx.x; i < part.coefficient_count; i += blockDim.x)
            k[i] = coefficients[part.coefficient_offset + i];
        for (unsigned sample = 0; sample < samples_per_group; ++sample) {
            const auto b = group * samples_per_group + sample;
            if (b >= batch) break;  // Uniform across the entire block.
            for (auto i = threadIdx.x; i < part.parent_count; i += blockDim.x)
                p[i] = input[b * in_width + parents[part.parent_offset + i]];
            for (auto i = threadIdx.x; i < part.child_count; i += blockDim.x) c[i] = T(0);
            for (auto i = threadIdx.x; i < part.weight_count; i += blockDim.x)
                w[i] = weights[b * weight_count + indices[part.weight_offset + i]];
            __syncthreads();
            #pragma unroll
            for (unsigned slot = 0; slot < edges_per_thread; ++slot) {
                if (threadIdx.x + slot * blockDim.x < part.edge_count) {
                    const PackedEdge e = cached[slot];
                    atomicAdd(c + e.child, (p[e.parent] * w[e.weight]) * k[e.coefficient]);
                }
            }
            __syncthreads();
            for (auto i = threadIdx.x; i < part.child_count; i += blockDim.x)
                atomicAdd(output + b * out_width + children[part.child_offset + i], c[i]);
            __syncthreads();
        }
    }
}

template <class T>
__global__ void readout_kernel(const T* layer, std::uint32_t width, const Readout* readouts,
                               std::uint64_t count, T* result, std::uint32_t outputs, std::uint64_t batch) {
    const auto stride = std::uint64_t(gridDim.x) * blockDim.x;
    for (auto item = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         item < batch * count; item += stride) {
        const auto b = item / count;
        const Readout r = readouts[item % count];
        result[b * outputs + r.column] = layer[b * width + r.node];
    }
}

template <class T> struct DeviceLayer {
    Buffer<std::uint64_t> row_offsets;
    Buffer<std::uint32_t> parents, children, indices;
    Buffer<T> coefficients;
    Buffer<PackedEdge> packed;
    Buffer<Partition> partitions;
    std::size_t edges = 0, shared_bytes = 0;
};

template <class T>
std::vector<T> convert(const std::vector<double>& values) {
    std::vector<T> result;
    result.reserve(values.size());
    for (double value : values) {
        T converted = static_cast<T>(value);
        if (!std::isfinite(converted)) throw std::invalid_argument("Coefficient overflows selected precision");
        result.push_back(converted);
    }
    return result;
}

unsigned grid(std::size_t tasks, unsigned threads) {
    return static_cast<unsigned>(std::min<std::size_t>(65535, ceil_div(tasks, threads)));
}

}  // namespace

bool cuda_device_available(int device, std::string* reason) noexcept {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        if (reason) *reason = cudaGetErrorString(status);
        cudaGetLastError();
        return false;
    }
    if (device < 0 || device >= count) {
        if (reason) *reason = "Requested CUDA device does not exist";
        return false;
    }
    return true;
}

template <class T> struct CudaExecutor<T>::Impl {
    Graph topology;
    std::size_t batch;
    CudaOptions options;
    DeviceInfo info;
    Stream stream;
    GraphHandle captured;
    Executable executable;
    Buffer<T> weights, sources, values[2], results;
    std::vector<DeviceLayer<T>> layers;
    std::vector<Buffer<Readout>> readouts;
    bool ready = false, produced = false;

    Impl(const Graph& graph, std::size_t count, CudaOptions opts) : topology(graph), batch(count), options(opts) {
        topology.validate();
        if (!batch || !options.samples_per_group || !options.threads || options.threads % 32 || !options.edges_per_partition)
            throw std::invalid_argument("Positive batch/group/edge cap and a warp-multiple thread count are required");
        backend_name(options.backend);
        BDG_CUDA(cudaSetDevice(options.device));
        cudaDeviceProp prop{};
        BDG_CUDA(cudaGetDeviceProperties(&prop, options.device));
        if (options.threads > static_cast<unsigned>(prop.maxThreadsPerBlock))
            throw std::invalid_argument("Thread count exceeds device limit");
        if (sizeof(T) == 8 && prop.major < 6)
            throw std::runtime_error("Double atomic operations require compute capability 6.0+");
        info.name = prop.name; info.major = prop.major; info.minor = prop.minor;
        info.multiprocessors = prop.multiProcessorCount;
        BDG_CUDA(cudaRuntimeGetVersion(&info.runtime_version));
        BDG_CUDA(cudaDriverGetVersion(&info.driver_version));
        BDG_CUDA(cudaStreamCreateWithFlags(&stream.handle, cudaStreamNonBlocking));
        const auto device_budget = std::max(prop.sharedMemPerBlock, prop.sharedMemPerBlockOptin);
        const auto budget = std::min(options.shared_budget_bytes, device_budget);
        const auto edge_cap = std::min<std::size_t>(options.edges_per_partition, options.threads * edges_per_thread);
        const auto plan = compile_plan(topology, edge_cap, budget, sizeof(T));
        layers.resize(plan.layers.size()); readouts.resize(plan.readouts.size());
        for (std::size_t i = 0; i < plan.readouts.size(); ++i) readouts[i].upload(plan.readouts[i]);
        for (std::size_t i = 0; i < plan.layers.size(); ++i) {
            const auto& host = plan.layers[i]; auto& device = layers[i];
            device.edges = host.parents.size();
            if (options.backend == CudaBackend::shared_partition) {
                device.parents.upload(host.unique_parents); device.children.upload(host.unique_children);
                device.indices.upload(host.unique_weights); device.coefficients.upload(convert<T>(host.unique_coefficients));
                device.packed.upload(host.packed_edges); device.partitions.upload(host.partitions);
                device.shared_bytes = host.max_shared_bytes;
                info.partitions += host.partitions.size();
                info.max_dynamic_shared_bytes = std::max(info.max_dynamic_shared_bytes, device.shared_bytes);
            } else {
                device.parents.upload(host.parents); device.indices.upload(host.weights);
                device.coefficients.upload(convert<T>(host.coefficients));
                if (options.backend == CudaBackend::edge_atomic) device.children.upload(host.children);
                else device.row_offsets.upload(host.row_offsets);
            }
        }
        if (info.max_dynamic_shared_bytes > prop.sharedMemPerBlock)
            BDG_CUDA(cudaFuncSetAttribute(shared_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          static_cast<int>(info.max_dynamic_shared_bytes)));
        weights.allocate(checked_product(batch, topology.weight_count));
        sources.allocate(checked_product(batch, topology.widths.front()));
        for (auto& buffer : values) buffer.allocate(checked_product(batch, topology.max_width()));
        results.allocate(checked_product(batch, topology.outputs.size()));
        info.workspace_bytes = (weights.size + sources.size + values[0].size + values[1].size + results.size) * sizeof(T);
        if (options.cuda_graph) {
            BDG_CUDA(cudaStreamBeginCapture(stream.handle, cudaStreamCaptureModeThreadLocal));
            try {
                sequence();
                BDG_CUDA(cudaStreamEndCapture(stream.handle, &captured.handle));
            } catch (...) {
                cudaGraph_t abandoned = nullptr;
                cudaStreamEndCapture(stream.handle, &abandoned);
                if (abandoned) cudaGraphDestroy(abandoned);
                throw;
            }
            BDG_CUDA(cudaGraphInstantiate(&executable.handle, captured.handle, nullptr, nullptr, 0));
        }
    }

    ~Impl() {
        cudaSetDevice(options.device);
        if (stream.handle) cudaStreamSynchronize(stream.handle);
    }

    void sequence() {
        BDG_CUDA(cudaMemcpyAsync(values[0].ptr, sources.ptr, sources.size * sizeof(T), cudaMemcpyDeviceToDevice, stream.handle));
        for (std::size_t l = 0; l < topology.widths.size(); ++l) {
            T* current = values[l % 2].ptr;
            if (readouts[l].size) {
                readout_kernel<T><<<grid(checked_product(batch, readouts[l].size), options.threads), options.threads, 0, stream.handle>>>(
                    current, topology.widths[l], readouts[l].ptr, readouts[l].size, results.ptr,
                    static_cast<std::uint32_t>(topology.outputs.size()), batch);
                BDG_CUDA(cudaGetLastError());
            }
            if (l == layers.size()) break;
            auto& layer = layers[l]; T* next = values[(l + 1) % 2].ptr;
            const auto iw = topology.widths[l], ow = topology.widths[l + 1];
            if (options.backend != CudaBackend::gather)
                BDG_CUDA(cudaMemsetAsync(next, 0, checked_product(checked_product(batch, ow), sizeof(T)), stream.handle));
            if (options.backend == CudaBackend::gather) {
                gather_kernel<T><<<grid(checked_product(checked_product(batch, ow), 32), options.threads), options.threads, 0, stream.handle>>>(
                    layer.row_offsets.ptr, layer.parents.ptr, layer.indices.ptr, layer.coefficients.ptr,
                    batch, weights.ptr, topology.weight_count, current, next, iw, ow);
            } else if (options.backend == CudaBackend::edge_atomic && layer.edges) {
                edge_kernel<T><<<grid(checked_product(batch, layer.edges), options.threads), options.threads, 0, stream.handle>>>(
                    layer.parents.ptr, layer.children.ptr, layer.indices.ptr, layer.coefficients.ptr,
                    layer.edges, batch, weights.ptr, topology.weight_count, current, next, iw, ow);
            } else if (options.backend == CudaBackend::shared_partition && layer.partitions.size) {
                const auto groups = ceil_div(batch, options.samples_per_group);
                const auto tasks = checked_product(groups, layer.partitions.size);
                shared_kernel<T><<<static_cast<unsigned>(std::min<std::size_t>(tasks, 65535)), options.threads,
                                   layer.shared_bytes, stream.handle>>>(
                    layer.packed.ptr, layer.partitions.ptr, layer.partitions.size, layer.parents.ptr,
                    layer.children.ptr, layer.indices.ptr, layer.coefficients.ptr, weights.ptr,
                    topology.weight_count, current, next, iw, ow, batch, options.samples_per_group, groups);
            }
            BDG_CUDA(cudaGetLastError());
        }
    }

    void enqueue() {
        if (!ready) throw std::logic_error("Call set_inputs before evaluation");
        if (options.cuda_graph) BDG_CUDA(cudaGraphLaunch(executable.handle, stream.handle));
        else sequence();
        produced = true;
    }
};

template <class T>
CudaExecutor<T>::CudaExecutor(const Graph& graph, std::size_t batch, CudaOptions options)
    : impl_(std::make_unique<Impl>(graph, batch, options)) {}
template <class T> CudaExecutor<T>::~CudaExecutor() = default;
template <class T> CudaExecutor<T>::CudaExecutor(CudaExecutor&&) noexcept = default;
template <class T> CudaExecutor<T>& CudaExecutor<T>::operator=(CudaExecutor&&) noexcept = default;

template <class T>
void CudaExecutor<T>::set_inputs(const std::vector<T>& weights, const std::vector<T>& sources) {
    validate_inputs(impl_->topology, impl_->batch, weights, sources);
    BDG_CUDA(cudaSetDevice(impl_->options.device));
    BDG_CUDA(cudaStreamSynchronize(impl_->stream.handle));
    const std::vector<T> ones = sources.empty() ? std::vector<T>(impl_->sources.size, T(1)) : std::vector<T>{};
    const auto& roots = sources.empty() ? ones : sources;
    BDG_CUDA(cudaMemcpyAsync(impl_->weights.ptr, weights.data(), weights.size() * sizeof(T), cudaMemcpyHostToDevice, impl_->stream.handle));
    BDG_CUDA(cudaMemcpyAsync(impl_->sources.ptr, roots.data(), roots.size() * sizeof(T), cudaMemcpyHostToDevice, impl_->stream.handle));
    BDG_CUDA(cudaStreamSynchronize(impl_->stream.handle));
    impl_->ready = true; impl_->produced = false;
}

template <class T> void CudaExecutor<T>::launch() {
    BDG_CUDA(cudaSetDevice(impl_->options.device)); impl_->enqueue();
}
template <class T> void CudaExecutor<T>::synchronize() {
    BDG_CUDA(cudaSetDevice(impl_->options.device)); BDG_CUDA(cudaStreamSynchronize(impl_->stream.handle));
}
template <class T> std::vector<T> CudaExecutor<T>::download_outputs() {
    if (!impl_->produced) throw std::logic_error("No evaluation has been launched");
    synchronize(); std::vector<T> out(impl_->results.size);
    BDG_CUDA(cudaMemcpy(out.data(), impl_->results.ptr, out.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return out;
}
template <class T> Timing CudaExecutor<T>::benchmark(std::size_t iterations, std::size_t warmup) {
    if (!iterations) throw std::invalid_argument("Iterations must be positive");
    BDG_CUDA(cudaSetDevice(impl_->options.device));
    for (std::size_t i = 0; i < warmup; ++i) impl_->enqueue();
    synchronize(); Event start, stop;
    const auto begin = std::chrono::steady_clock::now();
    BDG_CUDA(cudaEventRecord(start.handle, impl_->stream.handle));
    for (std::size_t i = 0; i < iterations; ++i) impl_->enqueue();
    BDG_CUDA(cudaEventRecord(stop.handle, impl_->stream.handle));
    BDG_CUDA(cudaEventSynchronize(stop.handle));
    const auto end = std::chrono::steady_clock::now();
    float elapsed = 0; BDG_CUDA(cudaEventElapsedTime(&elapsed, start.handle, stop.handle));
    if (!(elapsed > 0)) throw std::runtime_error("Timer resolution insufficient; increase iterations");
    return {elapsed / iterations, std::chrono::duration<double, std::milli>(end - begin).count() / iterations};
}
template <class T> const DeviceInfo& CudaExecutor<T>::device_info() const { return impl_->info; }

template class CudaExecutor<float>;
template class CudaExecutor<double>;

}  // namespace bdg
