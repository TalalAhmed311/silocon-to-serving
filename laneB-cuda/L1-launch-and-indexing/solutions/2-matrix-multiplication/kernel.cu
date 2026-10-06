// LeetGPU #2 Matrix Multiplication (naive) — Lane B L1 solution. A: M×N, B: N×K, C: M×K, row-major, device pointers.
#include <cuda_runtime.h>

__global__ void matmul_naive(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M,
                             int N, int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // x -> column of C (and of B): coalesced B reads, C writes
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= K) return;
  float sum = 0.0f;
  for (int i = 0; i < N; ++i) sum += A[row * N + i] * B[i * K + col];  // A[row][i] is a broadcast within the warp
  C[row * K + col] = sum;
}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(16, 16);
  dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  matmul_naive<<<grid, block>>>(A, B, C, M, N, K);
}
