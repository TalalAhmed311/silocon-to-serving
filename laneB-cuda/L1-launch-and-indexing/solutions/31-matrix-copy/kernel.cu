// LeetGPU #31 Matrix Copy — Lane B L1 solution. A, B: device pointers to N×N row-major floats.
#include <cuda_runtime.h>

__global__ void copy2d(const float* __restrict__ A, float* __restrict__ B, int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // x -> column: a warp covers 32 consecutive floats of one row
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < N && col < N) B[size_t(row) * N + col] = A[size_t(row) * N + col];
}

// Deliberately bad mapping, for the bench only: x -> row, so a warp touches 32 different rows.
__global__ void copy2d_uncoalesced(const float* __restrict__ A, float* __restrict__ B, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < N && col < N) B[size_t(row) * N + col] = A[size_t(row) * N + col];
}

void solve(const float* A, float* B, int N) {
  dim3 block(32, 8), grid((N + 31) / 32, (N + 7) / 8);
  copy2d<<<grid, block>>>(A, B, N);
}
