// Exercise 1 reference solution.
#pragma once
#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello(int* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  printf("block %d thread %d -> global %d\n", blockIdx.x, threadIdx.x, i);
  out[i] = i;
}

inline void launch_hello(int* d_out) { hello<<<2, 4>>>(d_out); }
