// 03_bank_conflicts.cu — shared memory has 32 banks of 4 bytes; a warp's accesses to distinct addresses in the same
// bank serialize. Column reads of a [32][32] float tile hit one bank 32×; [32][33] spreads them over all banks.
// Expected: "column, no pad" several× slower than "row"; "column, pad +1" ≈ "row". Confirm with
//   ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./p5.2_03_bank_conflicts
#include <cstdio>

#include "s2s_cuda.cuh"

template <int PAD, bool COLUMN>
__global__ void smem_read(float* out, int iters) {
  __shared__ float tile[32][32 + PAD];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < 32 * 32; i += blockDim.x) tile[i / 32][i % 32] = float(i);
  __syncthreads();
  float acc = 0.f;
  for (int it = 0; it < iters; ++it) {
    const int k = (w + it) & 31;
    acc += COLUMN ? tile[lane][k] : tile[k][lane];   // column: lane → row (stride 32+PAD words); row: lane → column
  }
  if (acc == -1.f) out[0] = acc;
}

int main() {
  s2s::DeviceBuffer<float> out(1);
  const int iters = 4096, blocks = 1024;
  std::printf("| access | median ms |\n|---|---|\n");
  auto row = s2s::time_gpu([&] { smem_read<0, false><<<blocks, 256>>>(out.get(), iters); });
  auto col = s2s::time_gpu([&] { smem_read<0, true><<<blocks, 256>>>(out.get(), iters); });
  auto pad = s2s::time_gpu([&] { smem_read<1, true><<<blocks, 256>>>(out.get(), iters); });
  std::printf("| row (conflict-free) | %.3f |\n| column, no pad (32-way) | %.3f |\n| column, pad +1 | %.3f |\n",
              row.median_ms, col.median_ms, pad.median_ms);
}
