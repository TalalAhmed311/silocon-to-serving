// LeetGPU #62 Value Clipping — Lane B L1 solution. input/output are device pointers.
#include <cuda_runtime.h>
#include <math.h>

__global__ void kernel_62(const float* __restrict__ input, float* __restrict__ output, int N, float lo, float hi) {
  // Grid-stride loop: correct for any N and any grid size; one launch config fits all problem sizes.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
    const float x = input[i];
    output[i] = fminf(fmaxf(x, lo), hi);
  }
}

void solve(const float* input, float* output, int N, float lo, float hi) {
  const int threads = 256;
  int blocks = (N + threads - 1) / threads;
  if (blocks > 65535) blocks = 65535;  // grid-stride covers the rest
  if (blocks < 1) blocks = 1;
  kernel_62<<<blocks, threads>>>(input, output, N, lo, hi);
}
