#include "bdg/cuda.hpp"
#include <stdexcept>

namespace bdg {
const char* backend_name(CudaBackend backend) {
    switch (backend) {
    case CudaBackend::edge_atomic: return "cuda-edge";
    case CudaBackend::shared_partition: return "cuda-shared";
    case CudaBackend::gather: return "cuda-gather";
    }
    throw std::invalid_argument("Unknown CUDA backend");
}
}  // namespace bdg
