// 02_nvtx_ranges.cu — NVTX ranges make your own phases visible in Nsight Systems (sm_75+; nvtx3 is header-only and
// ships with the CUDA toolkit). Run: nsys profile -o nvtx ./p5.3_02_nvtx_ranges → open in Nsight Systems: the
// "prefill" / "decode step" ranges sit above the kernels they launched, and CPU gaps between steps are obvious.
#include <cstdio>

#include <nvtx3/nvToolsExt.h>

#include "s2s_cuda.cuh"

__global__ void fake_layer(float* x, int n, int work) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float v = x[i]; for (int k = 0; k < work; ++k) v = v * 0.999f + 0.001f; x[i] = v; }
}

int main() {
  const int n = 1 << 20;
  s2s::DeviceBuffer<float> x(n);
  nvtxRangePushA("prefill");
  for (int l = 0; l < 16; ++l) fake_layer<<<n / 256, 256>>>(x.get(), n, 400);
  CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();
  for (int step = 0; step < 20; ++step) {
    nvtxRangePushA("decode step");
    for (int l = 0; l < 16; ++l) fake_layer<<<n / 256 / 64, 256>>>(x.get(), n / 64, 50);   // small kernels: launch-bound
    CUDA_CHECK(cudaDeviceSynchronize());                                                   // a sync per step: CPU gaps
    nvtxRangePop();
  }
  std::printf("done — now look at the nsys timeline: decode steps are mostly launch overhead and gaps (→ CUDA graphs, P6.6)\n");
}
