// Exercise 4: a program with a planted out-of-bounds write. It "works": the output looks right.
// Run it normally, then under compute-sanitizer (see README.md). (T2. TODO(run-on: T2))
#include <cstdio>
#include <vector>

#include "s2s_cuda.cuh"

// BUG: no bounds check. The grid is rounded up, so the last block has threads whose i is >= n.
__global__ void scale_kernel(float* x, int n, float factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  x[i] *= factor;  // missing: if (i < n)
}

int main() {
  const int n = 1000;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;  // 4 blocks = 1,024 threads for 1,000 elements
  std::vector<float> h(n, 1.0f);

  float* d_x = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  scale_kernel<<<blocks, threads>>>(d_x, n, 2.0f);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaMemcpy(h.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_x));

  std::printf("h[0] = %g, h[%d] = %g (expected 2 and 2)\n", h[0], n - 1, h[n - 1]);
  return 0;
}
