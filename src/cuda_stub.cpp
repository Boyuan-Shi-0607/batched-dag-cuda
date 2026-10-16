#include "bdg/cuda.hpp"
#include <stdexcept>

namespace bdg {
bool cuda_device_available(int, std::string* reason) noexcept {
    if (reason) *reason = "This build has no CUDA backend";
    return false;
}
namespace {
[[noreturn]] void unavailable() { throw std::runtime_error("CUDA backend was not built; no CPU fallback is performed"); }
}
template <class T> struct CudaExecutor<T>::Impl {};
template <class T> CudaExecutor<T>::CudaExecutor(const Graph&, std::size_t, CudaOptions) { unavailable(); }
template <class T> CudaExecutor<T>::~CudaExecutor() = default;
template <class T> CudaExecutor<T>::CudaExecutor(CudaExecutor&&) noexcept = default;
template <class T> CudaExecutor<T>& CudaExecutor<T>::operator=(CudaExecutor&&) noexcept = default;
template <class T> void CudaExecutor<T>::set_inputs(const std::vector<T>&, const std::vector<T>&) { unavailable(); }
template <class T> void CudaExecutor<T>::launch() { unavailable(); }
template <class T> void CudaExecutor<T>::synchronize() { unavailable(); }
template <class T> std::vector<T> CudaExecutor<T>::download_outputs() { unavailable(); }
template <class T> Timing CudaExecutor<T>::benchmark(std::size_t, std::size_t) { unavailable(); }
template <class T> const DeviceInfo& CudaExecutor<T>::device_info() const { unavailable(); }
template class CudaExecutor<float>;
template class CudaExecutor<double>;
}  // namespace bdg
