// Exercise 1 starter: your first kernel. Two blocks of four threads; every thread prints its global index and
// records it in out[i]. (T2: needs an NVIDIA GPU. TODO(run-on: T2))
#pragma once
#include <cstdio>
#include <cuda_runtime.h>

// TODO 1: write a __global__ kernel `hello(int* out)`.
//   - compute the global index  i = blockIdx.x * blockDim.x + threadIdx.x
//   - printf("block %d thread %d -> global %d\n", blockIdx.x, threadIdx.x, i);
//   - store it: out[i] = i;

// TODO 2: launch it from here with 2 blocks of 4 threads. Do not synchronize or check errors here:
// the test does that with CUDA_CHECK_LAUNCH().
inline void launch_hello(int* d_out) {
  (void)d_out;
}
