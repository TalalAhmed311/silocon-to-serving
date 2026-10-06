// LeetGPU #69 2D Jacobi Stencil — Lane B L2 solution (harness convention: 4-neighbour average, fixed boundary).
#include <cuda_runtime.h>

#include <utility>

__global__ void jacobi_step(const float* __restrict__ u, float* __restrict__ v, int H, int W) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x <= 0 || y <= 0 || x >= W - 1 || y >= H - 1) return;   // boundary stays as copied
  const size_t i = size_t(y) * W + x;
  v[i] = 0.25f * (u[i - 1] + u[i + 1] + u[i - W] + u[i + W]);
}

// grid: H×W input; output: H×W result after `iters` iterations. Uses one scratch buffer.
void solve(const float* grid, float* output, int H, int W, int iters) {
  float* scratch;
  cudaMalloc(&scratch, sizeof(float) * H * W);
  cudaMemcpy(output, grid, sizeof(float) * H * W, cudaMemcpyDeviceToDevice);
  cudaMemcpy(scratch, grid, sizeof(float) * H * W, cudaMemcpyDeviceToDevice);   // both carry the fixed boundary
  float *cur = output, *nxt = scratch;
  dim3 block(32, 8), g((W + 31) / 32, (H + 7) / 8);
  for (int k = 0; k < iters; ++k) {
    jacobi_step<<<g, block>>>(cur, nxt, H, W);   // kernel boundary = grid-wide barrier between iterations
    std::swap(cur, nxt);
  }
  if (cur != output) cudaMemcpy(output, cur, sizeof(float) * H * W, cudaMemcpyDeviceToDevice);
  cudaFree(scratch);
}
