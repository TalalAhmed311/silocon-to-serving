// Exercise 4 starter: 2D convolution (cross-correlation), K×K filter (odd K ≤ 7), zero padding, "same" output size.
// out[r][c] = Σ_{i,j} in[r + i - K/2][c + j - K/2] · w[i][j]
// TODO: (1) a naive kernel, then (2) a shared-memory version that loads a (TILE + K - 1)² input tile including the
// halo once per block, with the filter in __constant__ memory. Keep the launcher signature.
#pragma once
#include <cuda_runtime.h>

inline void conv2d(const float* in, const float* w_host, float* out, int H, int W, int K) {
  (void)in; (void)w_host; (void)out; (void)H; (void)W; (void)K;
}
