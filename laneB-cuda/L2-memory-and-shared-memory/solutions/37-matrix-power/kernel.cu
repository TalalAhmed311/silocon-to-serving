// LeetGPU #37 Matrix Power — Lane B L2 solution: exponentiation by squaring with the tiled matmul.
#include <cuda_runtime.h>

#include <utility>

#include "../2-matrix-multiplication-tiled/kernel.cu"

__global__ void set_identity(float* M, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N * N) M[i] = (i / N == i % N) ? 1.0f : 0.0f;
}

// input: N×N; output: N×N = input^P (P >= 0). Uses two scratch buffers.
void solve_power(const float* input, float* output, int N, int P) {
  float *base, *tmp;
  cudaMalloc(&base, sizeof(float) * N * N);
  cudaMalloc(&tmp, sizeof(float) * N * N);
  cudaMemcpy(base, input, sizeof(float) * N * N, cudaMemcpyDeviceToDevice);
  float* result = output;
  set_identity<<<(N * N + 255) / 256, 256>>>(result, N);
  for (int p = P; p > 0; p >>= 1) {
    if (p & 1) {                     // result = result · base
      solve(result, base, tmp, N, N, N);
      cudaMemcpy(result, tmp, sizeof(float) * N * N, cudaMemcpyDeviceToDevice);  // result must stay in `output`
    }
    if (p > 1) {                     // base = base · base
      solve(base, base, tmp, N, N, N);
      std::swap(base, tmp);
    }
  }
  cudaFree(base);
  cudaFree(tmp);
}
