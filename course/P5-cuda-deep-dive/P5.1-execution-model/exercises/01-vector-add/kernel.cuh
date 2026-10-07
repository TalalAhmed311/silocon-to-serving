// Exercise 1 starter: vector add for ANY n (including n > blocks × threads) with a grid-stride loop.
#pragma once
#include <cuda_runtime.h>

// TODO: write a __global__ kernel with a grid-stride loop, and launch it here with a grid sized to the GPU
// (e.g. 8 blocks per SM × 256 threads) — not one thread per element.
inline void vadd(const float* a, const float* b, float* c, long long n) {
  (void)a; (void)b; (void)c; (void)n;
}
