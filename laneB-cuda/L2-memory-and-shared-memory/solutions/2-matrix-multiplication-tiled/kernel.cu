// LeetGPU #2 Matrix Multiplication — Lane B L2 solution: shared-memory tiling. A: M×N, B: N×K, C: M×K.
#include <cuda_runtime.h>

constexpr int TS = 16;

__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[TS][TS], Bs[TS][TS];
  const int row = blockIdx.y * TS + threadIdx.y, col = blockIdx.x * TS + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < N; t += TS) {
    // Each thread loads one element of each tile; out-of-range elements become 0 (no branches in the inner loop).
    As[threadIdx.y][threadIdx.x] = (row < M && t + threadIdx.x < N) ? A[size_t(row) * N + t + threadIdx.x] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (t + threadIdx.y < N && col < K) ? B[size_t(t + threadIdx.y) * K + col] : 0.0f;
    __syncthreads();                       // tiles complete before use
    #pragma unroll
    for (int k = 0; k < TS; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();                       // everyone done reading before the next load overwrites
  }
  if (row < M && col < K) C[size_t(row) * K + col] = acc;
}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(TS, TS), grid((K + TS - 1) / TS, (M + TS - 1) / TS);
  matmul_tiled<<<grid, block>>>(A, B, C, M, N, K);
}
