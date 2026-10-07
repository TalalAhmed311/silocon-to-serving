// 05_streams_overlap.cu — overlap H2D copy, compute and D2H copy across chunks with 3 streams (sm_75+, pinned memory).
// Expected: the pipelined version approaches max(copy time, compute time) instead of their sum. Profile with
//   nsys profile -o overlap ./p5.1_05_streams_overlap      → the timeline shows copies and kernels overlapping.
#include <cstdio>

#include "s2s_cuda.cuh"

__global__ void busy(float* x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float v = x[i]; for (int k = 0; k < 200; ++k) v = v * 0.999f + 0.001f; x[i] = v; }
}

int main() {
  const int n = 1 << 24, chunks = 8, cn = n / chunks;
  float* h = nullptr;
  CUDA_CHECK(cudaMallocHost(&h, size_t(n) * 4));
  for (int i = 0; i < n; ++i) h[i] = 1.f;
  s2s::DeviceBuffer<float> d(n);
  cudaStream_t s[3];
  for (auto& x : s) CUDA_CHECK(cudaStreamCreate(&x));
  auto serial = s2s::time_gpu([&] {
    CUDA_CHECK(cudaMemcpy(d.get(), h, size_t(n) * 4, cudaMemcpyHostToDevice));
    busy<<<(n + 255) / 256, 256>>>(d.get(), n);
    CUDA_CHECK(cudaMemcpy(h, d.get(), size_t(n) * 4, cudaMemcpyDeviceToHost));
  }, 1, 5);
  auto piped = s2s::time_gpu([&] {
    for (int c = 0; c < chunks; ++c) {
      cudaStream_t st = s[c % 3];
      float* dp = d.get() + size_t(c) * cn;
      CUDA_CHECK(cudaMemcpyAsync(dp, h + size_t(c) * cn, size_t(cn) * 4, cudaMemcpyHostToDevice, st));
      busy<<<(cn + 255) / 256, 256, 0, st>>>(dp, cn);
      CUDA_CHECK(cudaMemcpyAsync(h + size_t(c) * cn, dp, size_t(cn) * 4, cudaMemcpyDeviceToHost, st));
    }
    CUDA_CHECK(cudaDeviceSynchronize());   // time_gpu records on the default stream: wait for all three
  }, 1, 5);
  std::printf("| schedule | median ms |\n|---|---|\n| serial (copy, compute, copy) | %.2f |\n| 8 chunks on 3 streams | %.2f |\n",
              serial.median_ms, piped.median_ms);
  for (auto& x : s) cudaStreamDestroy(x);
  CUDA_CHECK(cudaFreeHost(h));
}
