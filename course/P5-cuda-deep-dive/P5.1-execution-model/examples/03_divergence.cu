// 03_divergence.cu — warp divergence: lanes of one warp taking different branches run both paths serially (sm_75+).
// Expected: "divergent by lane" is ~2× slower than "uniform per warp" for the same total work; the branch-free version
// matches the uniform one. Times from s2s::time_gpu (median of 50).
#include <cstdio>

#include "s2s_cuda.cuh"

__device__ __forceinline__ float work_a(float x) { for (int k = 0; k < 64; ++k) x = sinf(x) * 1.0001f + 0.5f; return x; }
__device__ __forceinline__ float work_b(float x) { for (int k = 0; k < 64; ++k) x = cosf(x) * 0.9999f - 0.5f; return x; }

__global__ void divergent_by_lane(float* x, int n) {         // odd/even lanes split → every warp runs BOTH loops
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] = (threadIdx.x & 1) ? work_a(x[i]) : work_b(x[i]);
}

__global__ void uniform_per_warp(float* x, int n) {          // the condition is the same for all 32 lanes of a warp
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] = ((threadIdx.x >> 5) & 1) ? work_a(x[i]) : work_b(x[i]);
}

__global__ void small_branch_predicated(float* x, int n) {   // tiny branches compile to selects: no divergence cost
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = x[i];
  x[i] = (threadIdx.x & 1) ? v * 2.f : v + 1.f;
}

int main() {
  const int n = 1 << 22;
  s2s::DeviceBuffer<float> x(s2s::random_vec<float>(n));
  const dim3 grid((n + 255) / 256), block(256);
  std::printf("| kernel | median ms |\n|---|---|\n");
  auto t1 = s2s::time_gpu([&] { divergent_by_lane<<<grid, block>>>(x.get(), n); });
  std::printf("| divergent by lane | %.3f |\n", t1.median_ms);
  auto t2 = s2s::time_gpu([&] { uniform_per_warp<<<grid, block>>>(x.get(), n); });
  std::printf("| uniform per warp | %.3f |\n", t2.median_ms);
  auto t3 = s2s::time_gpu([&] { small_branch_predicated<<<grid, block>>>(x.get(), n); });
  std::printf("| small branch (predicated) | %.3f |\n", t3.median_ms);
  std::printf("divergence cost: %.2fx (expect ~2: both paths run with half the lanes masked off)\n", t1.median_ms / t2.median_ms);
}
