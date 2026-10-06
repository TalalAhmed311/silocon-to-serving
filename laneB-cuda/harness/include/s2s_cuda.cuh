// s2s_cuda.cuh — the local CUDA harness for Lane B and P5: error checks, RAII device buffers, timing with CUDA
// events (warm-up, repeats, median/p90), a measured device-copy bandwidth to report "% of copy", and JSON output.
// Minimum: CUDA 12.4, any sm_75+ GPU. Host-side checks reuse course/common/include/s2s/check.hpp.
#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                       \
  do {                                                                                         \
    cudaError_t _e = (call);                                                                   \
    if (_e != cudaSuccess) {                                                                   \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(_e), __FILE__, __LINE__, \
                   cudaGetErrorString(_e));                                                     \
      std::exit(1);                                                                            \
    }                                                                                          \
  } while (0)

// Launches are asynchronous: check the launch itself, then (in debug/test code) synchronize to surface faults.
#define CUDA_CHECK_LAUNCH() do { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); } while (0)

namespace s2s {

template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) : n_(n) { CUDA_CHECK(cudaMalloc(&p_, std::max<size_t>(1, n) * sizeof(T))); }
  explicit DeviceBuffer(const std::vector<T>& h) : DeviceBuffer(h.size()) { upload(h); }
  ~DeviceBuffer() { if (p_) cudaFree(p_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
  void upload(const std::vector<T>& h) { CUDA_CHECK(cudaMemcpy(p_, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); }
  std::vector<T> download() const {
    std::vector<T> h(n_);
    CUDA_CHECK(cudaMemcpy(h.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
  void zero() { CUDA_CHECK(cudaMemset(p_, 0, n_ * sizeof(T))); }
  T* get() { return p_; }
  const T* get() const { return p_; }
  size_t size() const { return n_; }

 private:
  T* p_ = nullptr;
  size_t n_ = 0;
};

template <class T>
std::vector<T> random_vec(size_t n, double lo = -1, double hi = 1, unsigned seed = 0) {
  std::mt19937 rng(seed);
  std::vector<T> v(n);
  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<long long> d((long long)lo, (long long)hi);
    for (auto& x : v) x = T(d(rng));
  } else {
    std::uniform_real_distribution<double> d(lo, hi);
    for (auto& x : v) x = T(d(rng));
  }
  return v;
}

struct GpuTiming { float median_ms, p90_ms; };

// Times `launch` (which must only enqueue GPU work) with CUDA events. Warm-up removes JIT/first-touch costs.
inline GpuTiming time_gpu(const std::function<void()>& launch, int warmup = 5, int reps = 50) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  std::vector<float> ms(size_t(reps));
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaEventRecord(a));
    launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaEventElapsedTime(&ms[size_t(i)], a, b));
  }
  CUDA_CHECK(cudaGetLastError());
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  std::sort(ms.begin(), ms.end());
  return {ms[ms.size() / 2], ms[std::min(ms.size() - 1, size_t(0.9 * (ms.size() - 1) + 0.5))]};
}

// A plain grid-stride float4 copy: the "100%" that memory-bound kernels are compared against.
__global__ void s2s_copy_kernel(const float4* __restrict__ in, float4* __restrict__ out, size_t n4) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n4; i += size_t(gridDim.x) * blockDim.x) out[i] = in[i];
}

// Measured device-to-device bandwidth in GB/s (read + write), for a buffer much larger than L2.
inline double measure_copy_gbs(size_t bytes = size_t(1) << 28) {
  DeviceBuffer<float> a(bytes / 4), b(bytes / 4);
  CUDA_CHECK(cudaMemset(a.get(), 0, bytes));
  int sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  auto t = time_gpu([&] {
    s2s_copy_kernel<<<sms * 8, 256>>>(reinterpret_cast<const float4*>(a.get()), reinterpret_cast<float4*>(b.get()), bytes / 16);
  });
  return 2.0 * double(bytes) / (t.median_ms * 1e6);
}

inline std::string gpu_name() {
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  return std::string(p.name) + " (sm_" + std::to_string(p.major) + std::to_string(p.minor) + ")";
}

// Prints one Markdown row and appends it to results/<bench>.json (JSON-lines, one object per row).
inline void report(const std::string& bench, const std::string& variant, double size, GpuTiming t, double rate,
                   const char* unit, double peak) {
  const double pct = peak > 0 ? 100.0 * rate / peak : 0.0;
  std::printf("| %s | %.0f | %.4f | %.4f | %.1f %s | %.1f%% |\n", variant.c_str(), size, t.median_ms, t.p90_ms, rate, unit, pct);
  std::system("mkdir -p results");
  std::ofstream f("results/" + bench + ".jsonl", std::ios::app);
  f << "{\"bench\": \"" << bench << "\", \"gpu\": \"" << gpu_name() << "\", \"label\": \"" << variant << "\", \"size\": " << size
    << ", \"median_ms\": " << t.median_ms << ", \"p90_ms\": " << t.p90_ms << ", \"rate\": " << rate << ", \"unit\": \""
    << unit << "\", \"pct_peak\": " << pct << "}\n";
}

inline void table_header(const char* unit) {
  std::printf("\n| variant | problem size | median ms | p90 ms | %s | %% of peak |\n|---|---|---|---|---|---|\n", unit);
}

}  // namespace s2s
