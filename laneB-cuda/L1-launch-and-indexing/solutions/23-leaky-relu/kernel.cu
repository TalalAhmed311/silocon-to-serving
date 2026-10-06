// LeetGPU #23 Leaky ReLU — Lane B L1 solution. input/output are device pointers.
#include <cuda_runtime.h>
#include <math.h>

__global__ void kernel_23(const float* __restrict__ input, float* __restrict__ output, int N, float alpha) {
  // Grid-stride loop: correct for any N and any grid size; one launch config fits all problem sizes.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
    const float x = input[i];
    output[i] = x > 0.0f ? x : alpha * x;
  }
}

void solve(const float* input, float* output, int N, float alpha) {
  const int threads = 256;
  int blocks = (N + threads - 1) / threads;
  if (blocks > 65535) blocks = 65535;  // grid-stride covers the rest
  if (blocks < 1) blocks = 1;
  kernel_23<<<blocks, threads>>>(input, output, N, alpha);
}
