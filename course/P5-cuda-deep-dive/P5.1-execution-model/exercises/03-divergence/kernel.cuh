// Exercise 3 starter: a planted divergent kernel. clamp_and_scale(x) = x < 0 ? 0 : (x > 1 ? 1 : x) * scale,
// written with lane-dependent control flow. Make it branch-free (or warp-uniform) with IDENTICAL results.
#pragma once
#include <cuda_runtime.h>

__global__ void clamp_scale_kernel(const float* __restrict__ in, float* __restrict__ out, int n, float scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = in[i];
  if (v < 0.f) {                            // TODO: these branches diverge within a warp on random data
    for (int k = 0; k < 8; ++k) v = 0.f;    // (silly loops planted so the divergence is measurable)
  } else if (v > 1.f) {
    for (int k = 0; k < 8; ++k) v = scale;
  } else {
    for (int k = 0; k < 8; ++k) v = in[i] * scale;
  }
  out[i] = v;
}

inline void clamp_scale(const float* in, float* out, int n, float scale) {
  clamp_scale_kernel<<<(n + 255) / 256, 256>>>(in, out, n, scale);
}
