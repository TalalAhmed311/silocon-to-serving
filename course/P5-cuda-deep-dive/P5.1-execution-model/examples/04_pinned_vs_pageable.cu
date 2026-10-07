// 04_pinned_vs_pageable.cu — host↔device copy bandwidth from pageable vs pinned (page-locked) host memory (sm_75+).
// Closes the loop with P0.2: pageable copies go through a driver staging buffer; pinned memory can be DMA'd directly.
// Expected: pinned H2D/D2H several× faster than pageable and close to the PCIe link rate (cite it in your notes).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "s2s_cuda.cuh"

int main() {
  const size_t bytes = size_t(256) << 20;
  float* pageable = static_cast<float*>(std::malloc(bytes));
  float* pinned = nullptr;
  CUDA_CHECK(cudaMallocHost(&pinned, bytes));
  std::memset(pageable, 1, bytes);
  std::memset(pinned, 1, bytes);
  s2s::DeviceBuffer<float> d(bytes / 4);
  std::printf("| host memory | direction | GB/s |\n|---|---|---|\n");
  using P = std::pair<const char*, float*>;
  for (auto [name, h] : {P{"pageable", pageable}, P{"pinned", pinned}}) {
    auto h2d = s2s::time_gpu([&] { CUDA_CHECK(cudaMemcpy(d.get(), h, bytes, cudaMemcpyHostToDevice)); }, 2, 10);
    auto d2h = s2s::time_gpu([&] { CUDA_CHECK(cudaMemcpy(h, d.get(), bytes, cudaMemcpyDeviceToHost)); }, 2, 10);
    std::printf("| %s | H2D | %.1f |\n| %s | D2H | %.1f |\n", name, bytes / (h2d.median_ms * 1e6), name, bytes / (d2h.median_ms * 1e6));
  }
  std::printf("Only pinned memory allows cudaMemcpyAsync to overlap with compute (05_streams_overlap).\n");
  CUDA_CHECK(cudaFreeHost(pinned));
  std::free(pageable);
}
