// Exercise 2 starter: transpose rows×cols row-major `in` into cols×rows `out`. This is rung 1 (naive): reads are
// coalesced, writes are strided. Reach ≥ 80% of copy bandwidth: shared-memory tile + padding (P5.2 §3).
#pragma once
#include <cuda_runtime.h>

__global__ void transpose_kernel(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < rows && c < cols) out[(size_t)c * rows + r] = in[(size_t)r * cols + c];   // TODO: coalesce the writes
}

inline void transpose(const float* in, float* out, int rows, int cols) {
  dim3 block(32, 8), grid((cols + 31) / 32, (rows + 7) / 8);
  transpose_kernel<<<grid, block>>>(in, out, rows, cols);
}
