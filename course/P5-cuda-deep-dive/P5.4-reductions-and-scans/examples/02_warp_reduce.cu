// 02_warp_reduce.cu — a warp reduction three ways, verified equal: shared memory + __syncwarp, __shfl_down_sync,
// and cooperative groups' cg::reduce (sm_75+). Prints the per-warp sums of the first warps.
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cstdio>

#include "s2s_cuda.cuh"

namespace cg = cooperative_groups;

__global__ void three_ways(const float* in, float* out3) {
  __shared__ float s[32 * 8];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, gw = blockIdx.x * (blockDim.x >> 5) + w;
  const float v = in[blockIdx.x * blockDim.x + threadIdx.x];
  // (a) shared memory, warp-synchronous with explicit __syncwarp (no implicit lockstep since Volta)
  float* ws = s + 32 * w;
  ws[lane] = v;
  __syncwarp();
  for (int o = 16; o > 0; o >>= 1) { if (lane < o) ws[lane] += ws[lane + o]; __syncwarp(); }
  // (b) shuffles: values move register-to-register, no shared memory
  float x = v;
  for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
  // (c) cooperative groups
  auto tile = cg::tiled_partition<32>(cg::this_thread_block());
  const float y = cg::reduce(tile, v, cg::plus<float>());
  if (lane == 0) { out3[3 * gw] = ws[0]; out3[3 * gw + 1] = x; out3[3 * gw + 2] = y; }
}

int main() {
  const int blocks = 4, threads = 256, warps = blocks * threads / 32;
  auto h = s2s::random_vec<int>(blocks * threads, -5, 5);
  std::vector<float> hf(h.begin(), h.end());
  s2s::DeviceBuffer<float> in(hf), out(size_t(3 * warps));
  three_ways<<<blocks, threads>>>(in.get(), out.get());
  CUDA_CHECK_LAUNCH();
  auto o = out.download();
  std::printf("| warp | smem | shfl_down | cg::reduce |\n|---|---|---|---|\n");
  for (int w = 0; w < 6; ++w) std::printf("| %d | %.0f | %.0f | %.0f |\n", w, o[3 * w], o[3 * w + 1], o[3 * w + 2]);
}
