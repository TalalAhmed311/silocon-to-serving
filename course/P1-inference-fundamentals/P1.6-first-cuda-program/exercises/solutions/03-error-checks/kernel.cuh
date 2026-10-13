// Exercise 3 reference solution.
#pragma once
#include <cuda_runtime.h>

__global__ void scale_kernel(float* x, int n, float factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= factor;
}

__global__ void fault_kernel(float* p) { p[threadIdx.x] = 1.0f; }

// Same two-step check as CUDA_CHECK_LAUNCH(), but it RETURNS the error so a test can look at it.
inline cudaError_t check_launch() {
  cudaError_t e = cudaGetLastError();      // 1. was the launch itself valid? (known at once)
  if (e != cudaSuccess) return e;
  return cudaDeviceSynchronize();          // 2. did the kernel run without faulting? (known when it finishes)
}

inline cudaError_t launch_scale(float* d_x, int n, int threads) {
  const int blocks = (n + threads - 1) / threads;
  scale_kernel<<<blocks, threads>>>(d_x, n, 2.0f);
  return check_launch();
}

inline cudaError_t launch_fault() {
  fault_kernel<<<1, 32>>>(nullptr);
  return check_launch();
}
