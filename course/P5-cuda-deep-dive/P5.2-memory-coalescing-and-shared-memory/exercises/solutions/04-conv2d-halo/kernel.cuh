// Exercise 4 reference: shared-memory halo tiles + __constant__ filter.
#pragma once
#include <cuda_runtime.h>

namespace conv {
constexpr int TILE = 16, MAXK = 7;
__constant__ float W[MAXK * MAXK];

// Block = TILE × TILE threads computing TILE × TILE outputs. The input tile with halo is (TILE + K - 1)²; threads
// cooperatively load it (some load several elements), zero-filling outside the image.
__global__ void conv2d_tiled(const float* __restrict__ in, float* __restrict__ out, int H, int Wd, int K) {
  __shared__ float s[TILE + MAXK - 1][TILE + MAXK - 1 + 1];        // +1 column of padding against bank conflicts
  const int R = K / 2, S = TILE + K - 1;
  const int r0 = blockIdx.y * TILE - R, c0 = blockIdx.x * TILE - R;
  for (int i = threadIdx.y; i < S; i += TILE)
    for (int j = threadIdx.x; j < S; j += TILE) {
      const int r = r0 + i, c = c0 + j;
      s[i][j] = (r >= 0 && r < H && c >= 0 && c < Wd) ? in[(size_t)r * Wd + c] : 0.f;
    }
  __syncthreads();
  const int r = blockIdx.y * TILE + threadIdx.y, c = blockIdx.x * TILE + threadIdx.x;
  if (r >= H || c >= Wd) return;
  float acc = 0.f;
  for (int i = 0; i < K; ++i)
    for (int j = 0; j < K; ++j) acc = fmaf(s[threadIdx.y + i][threadIdx.x + j], W[i * K + j], acc);   // W: broadcast
  out[(size_t)r * Wd + c] = acc;
}
}  // namespace conv

inline void conv2d(const float* in, const float* w_host, float* out, int H, int W, int K) {
  cudaMemcpyToSymbol(conv::W, w_host, sizeof(float) * K * K);
  dim3 block(conv::TILE, conv::TILE), grid((W + conv::TILE - 1) / conv::TILE, (H + conv::TILE - 1) / conv::TILE);
  conv::conv2d_tiled<<<grid, block>>>(in, out, H, W, K);
}
