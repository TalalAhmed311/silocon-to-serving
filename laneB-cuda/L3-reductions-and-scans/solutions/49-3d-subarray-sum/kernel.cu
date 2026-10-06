// LeetGPU #49 3D Subarray Sum — Lane B L3 solution. output[0] = sum over the inclusive box
// [S_DEP..E_DEP] x [S_ROW..E_ROW] x [S_COL..E_COL] of an N x M x K (depth, rows, cols) row-major int volume.
#include <cuda_runtime.h>

__device__ __forceinline__ int warp_sum(int v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

// Flatten the (depth, row) pairs of the box onto blockIdx.y; threads walk the contiguous column run.
__global__ void box_sum(const int* __restrict__ in, int* __restrict__ out, int M, int K,
                        int d0, int r0, int c0, int rows, int cols, int planes) {
  __shared__ int warp_part[32];
  int acc = 0;
  for (int pr = blockIdx.y; pr < planes * rows; pr += gridDim.y) {
    const int d = d0 + pr / rows, r = r0 + pr % rows;
    const size_t base = (size_t(d) * M + r) * K + c0;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < cols; c += gridDim.x * blockDim.x) acc += in[base + c];
  }
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

void solve(const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP, int S_ROW, int E_ROW,
           int S_COL, int E_COL) {
  (void)N;
  cudaMemset(output, 0, sizeof(int));
  const int planes = E_DEP - S_DEP + 1, rows = E_ROW - S_ROW + 1, cols = E_COL - S_COL + 1;
  const int pr = planes * rows;
  dim3 grid((cols + 127) / 128 < 8 ? (cols + 127) / 128 : 8, pr < 4096 ? pr : 4096);
  box_sum<<<grid, 128>>>(input, output, M, K, S_DEP, S_ROW, S_COL, rows, cols, planes);
}
