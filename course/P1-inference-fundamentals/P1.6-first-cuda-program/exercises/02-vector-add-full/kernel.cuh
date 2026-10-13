// Exercise 2 starter: the full five-step host program for c = a + b, for ANY n.
// Host pointers in, host pointer out: this function owns every allocation and copy. (T2. TODO(run-on: T2))
#pragma once
#include <cuda_runtime.h>

#include "s2s_cuda.cuh"  // CUDA_CHECK, CUDA_CHECK_LAUNCH

// TODO A: write the __global__ kernel  add_kernel(const float* a, const float* b, float* c, int n)
//   one thread per element, with the bounds check  if (i < n).

inline void vector_add(const float* h_a, const float* h_b, float* h_c, int n) {
  (void)h_a; (void)h_b; (void)h_c; (void)n;
  // TODO B, the five steps (wrap every CUDA call in CUDA_CHECK):
  //  1. cudaMalloc  d_a, d_b, d_c  (n * sizeof(float) bytes each)
  //  2. cudaMemcpy  h_a -> d_a and h_b -> d_b   (cudaMemcpyHostToDevice)
  //  3. launch add_kernel<<<blocks, threads>>>  with  blocks = ceil(n / threads), threads = 256,
  //     then CUDA_CHECK_LAUNCH();
  //  4. cudaMemcpy  d_c -> h_c   (cudaMemcpyDeviceToHost)
  //  5. cudaFree    d_a, d_b, d_c
}
