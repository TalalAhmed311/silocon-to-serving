// 01_planted_slow.cu — three kernels that each do the same useful work as `reference_sum_rows`, each slow for ONE
// reason. Diagnose them from Nsight Compute alone (exercise 1), before reading the comments at the bottom.
//   sudo $(which ncu) --set full -o planted ./p5.3_01_planted_slow
// (sm_75+). Task for each: row sums of a rows × 256 float matrix.
#include <cstdio>

#include "s2s_cuda.cuh"

constexpr int COLS = 256;

__global__ void reference_sum_rows(const float* __restrict__ m, float* __restrict__ out, int rows) {
  // one warp per row, lanes stride along the row (coalesced), shuffle reduction
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= rows) return;
  float s = 0.f;
  for (int c = lane; c < COLS; c += 32) s += m[(size_t)warp * COLS + c];
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (lane == 0) out[warp] = s;
}

__global__ void mystery_a(const float* __restrict__ m, float* __restrict__ out, int rows) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float s = 0.f;
  for (int c = 0; c < COLS; ++c) s += m[(size_t)r * COLS + c];
  out[r] = s;
}

__global__ void mystery_b(const float* __restrict__ m, float* __restrict__ out, int rows) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= rows) return;
  if (lane == 0) out[warp] = 0.f;
  __syncwarp();
  for (int c = lane; c < COLS; c += 32) atomicAdd(&out[warp], m[(size_t)warp * COLS + c]);
}

__global__ void __launch_bounds__(32) mystery_c(const float* __restrict__ m, float* __restrict__ out, int rows) {
  __shared__ float pad[12 * 1024];                    // 48 KB per 32-thread block
  const int warp = blockIdx.x, lane = threadIdx.x;
  if (warp >= rows) return;
  float s = 0.f;
  for (int c = lane; c < COLS; c += 32) s += m[(size_t)warp * COLS + c];
  pad[lane] = s;
  __syncwarp();
  s = 0.f;
  for (int k = 0; k < 32; ++k) s += pad[k];
  if (lane == 0) out[warp] = s;
}

int main() {
  const int rows = 1 << 18;
  s2s::DeviceBuffer<float> m(s2s::random_vec<float>(size_t(rows) * COLS)), out(rows);
  std::printf("| kernel | median ms | GB/s |\n|---|---|---|\n");
  auto row = [&](const char* n, s2s::GpuTiming t) {
    std::printf("| %s | %.3f | %.1f |\n", n, t.median_ms, 4.0 * rows * COLS / (t.median_ms * 1e6));
  };
  row("reference", s2s::time_gpu([&] { reference_sum_rows<<<rows / 8, 256>>>(m.get(), out.get(), rows); }));
  row("mystery_a", s2s::time_gpu([&] { mystery_a<<<rows / 256, 256>>>(m.get(), out.get(), rows); }));
  row("mystery_b", s2s::time_gpu([&] { mystery_b<<<rows / 8, 256>>>(m.get(), out.get(), rows); }));
  row("mystery_c", s2s::time_gpu([&] { mystery_c<<<rows, 32>>>(m.get(), out.get(), rows); }));
}

/* Answers (don't peek before exercise 1):
 * a — uncoalesced: thread r walks row r; adjacent lanes are 1 KB apart → 32 sectors per request, high
 *     l1tex sectors/request, memory workload shows low efficiency. Fix: warp per row (the reference).
 * b — atomic contention: 32 lanes atomically add to the same address, serialized; look at lts__t_..._atom metrics and
 *     "Warp State: Stall LG Throttle / membar". Fix: reduce in registers with shuffles, one store.
 * c — occupancy: 32-thread blocks with 48 KB smem each → ~1–2 blocks (1–2 warps) per SM; achieved occupancy tiny,
 *     memory latency not hidden ("Stall Long Scoreboard" dominates). Fix: 8 warps per block, no pointless smem.
 */
