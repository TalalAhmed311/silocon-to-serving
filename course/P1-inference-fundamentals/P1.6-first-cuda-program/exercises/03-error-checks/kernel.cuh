// Exercise 3 starter: two planted bugs that fail SILENTLY. Add checks so the error is reported to the caller.
// (T2. TODO(run-on: T2)) Both functions return a cudaError_t; today they always return cudaSuccess.
#pragma once
#include <cuda_runtime.h>

__global__ void scale_kernel(float* x, int n, float factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= factor;
}

__global__ void fault_kernel(float* p) { p[threadIdx.x] = 1.0f; }  // writes through whatever pointer it is given

// Doubles x[0..n). `threads` is chosen by the CALLER. With threads = 2048 the launch is invalid (the limit is
// 1024 threads per block), and nothing tells you.
inline cudaError_t launch_scale(float* d_x, int n, int threads) {
  const int blocks = (n + threads - 1) / threads;
  scale_kernel<<<blocks, threads>>>(d_x, n, 2.0f);
  // TODO 1: report a bad launch configuration. Hint: cudaGetLastError().
  // TODO 2: also report a fault that happens INSIDE the kernel. Hint: cudaDeviceSynchronize().
  return cudaSuccess;  // BUG: always claims success
}

// Launches fault_kernel with a null pointer: every thread writes to an invalid address, later, on the GPU.
inline cudaError_t launch_fault() {
  fault_kernel<<<1, 32>>>(nullptr);
  // TODO: the launch itself is valid, so cudaGetLastError() is not enough here. What do you have to wait for?
  return cudaSuccess;  // BUG: always claims success
}
