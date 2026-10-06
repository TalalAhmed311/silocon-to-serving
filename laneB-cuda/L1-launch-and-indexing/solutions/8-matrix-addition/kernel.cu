// LeetGPU #8 Matrix Addition — Lane B L1 solution. Device pointers; N×N row-major.
#include <cuda_runtime.h>

__global__ void add_flat(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    C[i] = A[i] + B[i];
}

void solve(const float* A, const float* B, float* C, int N) {
  const size_t n = size_t(N) * N;  // size_t: N*N overflows int for N > 46340
  size_t blocks = (n + 255) / 256;
  add_flat<<<unsigned(blocks > 65535 ? 65535 : (blocks ? blocks : 1)), 256>>>(A, B, C, n);
}
