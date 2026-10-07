// d4/transpose.cuh — the P5.2 transpose ladder: naive → shared-memory tile → padded tile (no bank conflicts).
// out[c][r] = in[r][c] for a rows × cols row-major matrix. L2 exit check: rung 3 ≥ 80% of copy bandwidth.
#pragma once
#include "common.cuh"

namespace d4 {

constexpr int TT = 32, TROWS = 8;   // 32×32 tile, each thread handles 32/8 = 4 rows

// Rung 1: reads are coalesced (consecutive threads → consecutive columns), writes are strided by `rows`: each warp
// store touches 32 different 32-byte sectors.
__global__ void transpose_naive(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  const int c = blockIdx.x * TT + threadIdx.x;
  for (int k = 0; k < TT; k += TROWS) {
    const int r = blockIdx.y * TT + threadIdx.y + k;
    if (r < rows && c < cols) out[(size_t)c * rows + r] = in[(size_t)r * cols + c];
  }
}

// Rungs 2 and 3: stage the tile in shared memory so BOTH the global read and the global write are coalesced.
// PAD = 0: reading the tile column-wise hits one bank 32 times (32-way conflict). PAD = 1 shifts each row by one bank.
template <int PAD>
__global__ void transpose_smem(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  __shared__ float tile[TT][TT + PAD];
  int c = blockIdx.x * TT + threadIdx.x;
  for (int k = 0; k < TT; k += TROWS) {
    const int r = blockIdx.y * TT + threadIdx.y + k;
    if (r < rows && c < cols) tile[threadIdx.y + k][threadIdx.x] = in[(size_t)r * cols + c];
  }
  __syncthreads();
  c = blockIdx.y * TT + threadIdx.x;                    // output column = input row
  for (int k = 0; k < TT; k += TROWS) {
    const int r = blockIdx.x * TT + threadIdx.y + k;    // output row = input column
    if (r < cols && c < rows) out[(size_t)r * rows + c] = tile[threadIdx.x][threadIdx.y + k];
  }
}

inline void transpose(const float* in, float* out, int rows, int cols, int rung = 3, cudaStream_t s = 0) {
  dim3 block(TT, TROWS), grid(ceil_div(cols, TT), ceil_div(rows, TT));
  if (rung == 1) transpose_naive<<<grid, block, 0, s>>>(in, out, rows, cols);
  else if (rung == 2) transpose_smem<0><<<grid, block, 0, s>>>(in, out, rows, cols);
  else transpose_smem<1><<<grid, block, 0, s>>>(in, out, rows, cols);
}

}  // namespace d4
