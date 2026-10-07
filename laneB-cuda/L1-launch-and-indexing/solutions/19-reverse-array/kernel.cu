// LeetGPU #19 Reverse Array — Lane B L1 solution. input: device pointer, reversed in place.
#include <cuda_runtime.h>

__global__ void reverse_inplace(float* a, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N / 2) {  // one thread per pair: no two threads touch the same element
    float t = a[i];
    a[i] = a[N - 1 - i];
    a[N - 1 - i] = t;
  }
}

void solve(float* input, int N) {
  const int threads = 256, pairs = N / 2;
  if (pairs == 0) return;
  reverse_inplace<<<(pairs + threads - 1) / threads, threads>>>(input, N);
}
