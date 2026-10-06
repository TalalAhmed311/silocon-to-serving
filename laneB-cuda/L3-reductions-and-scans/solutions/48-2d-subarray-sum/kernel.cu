// LeetGPU #48 2D Subarray Sum — Lane B L3 solution. output[0] = sum of input[r][c] for r in [S_ROW, E_ROW],
// c in [S_COL, E_COL] (inclusive), row-major N x M ints. Our signature; check the statement.
#include <cuda_runtime.h>

__device__ __forceinline__ int warp_sum(int v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

// 2D grid: blockIdx.y walks rows, threads walk columns (coalesced along a row).
__global__ void rect_sum(const int* __restrict__ in, int* __restrict__ out, int M, int r0, int r1, int c0, int c1) {
  __shared__ int warp_part[32];
  int acc = 0;
  for (int r = r0 + blockIdx.y; r <= r1; r += gridDim.y)
    for (int c = c0 + blockIdx.x * blockDim.x + threadIdx.x; c <= c1; c += gridDim.x * blockDim.x)
      acc += in[size_t(r) * M + c];
  acc = warp_sum(acc);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (lane == 0) warp_part[w] = acc;
  __syncthreads();
  if (w == 0) {
    acc = lane < (blockDim.x >> 5) ? warp_part[lane] : 0;
    acc = warp_sum(acc);
    if (lane == 0) atomicAdd(out, acc);
  }
}

void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL, int E_COL) {
  (void)N;
  cudaMemset(output, 0, sizeof(int));
  const int cols = E_COL - S_COL + 1, rows = E_ROW - S_ROW + 1;
  dim3 block(128);
  dim3 grid((cols + 127) / 128 < 8 ? (cols + 127) / 128 : 8, rows < 1024 ? rows : 1024);
  rect_sum<<<grid, block>>>(input, output, M, S_ROW, E_ROW, S_COL, E_COL);
}
