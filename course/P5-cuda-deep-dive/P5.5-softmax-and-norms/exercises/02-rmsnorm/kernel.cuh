// Exercise 2 starter: RMSNorm, rows × cols fp32: y = x · rsqrt(mean(x²) + eps) · w. One block (or warp) per row,
// float4 loads, fp32 accumulation. Goal: within X% of copy bandwidth (set X after measuring the baseline; README).
#pragma once
#include <cuda_runtime.h>

inline void rmsnorm(const float* x, const float* w, float* y, int rows, int cols, float eps) {
  (void)x; (void)w; (void)y; (void)rows; (void)cols; (void)eps;   // TODO
}
