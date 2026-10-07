// Branch-free: fminf/fmaxf compile to single instructions; every lane executes the same instruction stream.
#pragma once
#include <cuda_runtime.h>

__global__ void clamp_scale_kernel(const float* __restrict__ in, float* __restrict__ out, int n, float scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fminf(fmaxf(in[i], 0.f), 1.f) * scale;
}

inline void clamp_scale(const float* in, float* out, int n, float scale) {
  clamp_scale_kernel<<<(n + 255) / 256, 256>>>(in, out, n, scale);
}
