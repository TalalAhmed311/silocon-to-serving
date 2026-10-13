// Exercise 2 reference solution.
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

#include "s2s_cuda.cuh"

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];  // the guard: the last block has threads with no element
}

inline void vector_add(const float* h_a, const float* h_b, float* h_c, int n) {
  const size_t bytes = size_t(n) * sizeof(float);
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

  CUDA_CHECK(cudaMalloc(&d_a, bytes));  // 1. allocate device memory
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));  // 2. copy in
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;  // 3. launch, grid rounded up
  add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
  CUDA_CHECK_LAUNCH();

  CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));  // 4. copy out

  CUDA_CHECK(cudaFree(d_a));  // 5. free
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
}
