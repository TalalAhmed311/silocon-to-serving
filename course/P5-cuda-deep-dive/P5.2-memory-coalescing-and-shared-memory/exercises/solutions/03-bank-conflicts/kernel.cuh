// Exercise 3 reference: same kernel, padded tile: per-tile ROW sums through shared memory. out[r][j] = Σ_{c in tile j} in[r][c],
// in: rows × cols (both multiples of 32), out: rows × (cols/32). The global load is coalesced; the shared-memory read
// (lane tx walks row tx: addresses 32 words apart) is a 32-way bank conflict. Fix it so ncu's
// l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum drops to ~0, with identical results.
#pragma once
#include <cuda_runtime.h>

__global__ void tile_rowsum_kernel(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  __shared__ float t[32][33];                      // +1 word per row: word 33·tx + k → bank (tx + k) mod 32, distinct per lane
  const int tx = threadIdx.x, ty = threadIdx.y;    // block (32, 32)
  t[ty][tx] = in[(size_t)(blockIdx.y * 32 + ty) * cols + blockIdx.x * 32 + tx];   // coalesced: tx along a row
  __syncthreads();
  if (ty == 0) {
    float s = 0.f;
    for (int k = 0; k < 32; ++k) s += t[tx][k];    // lane tx reads row tx: word 32·tx + k → bank k for EVERY lane
    out[(size_t)(blockIdx.y * 32 + tx) * (cols / 32) + blockIdx.x] = s;
  }
}

inline void tile_rowsum(const float* in, float* out, int rows, int cols) {
  tile_rowsum_kernel<<<dim3(cols / 32, rows / 32), dim3(32, 32)>>>(in, out, rows, cols);
}
