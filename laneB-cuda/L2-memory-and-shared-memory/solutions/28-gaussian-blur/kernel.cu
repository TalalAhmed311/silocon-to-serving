// LeetGPU #28 Gaussian Blur — Lane B L2 solution (harness convention: given K×K kernel, same-size output, zero padding).
#include <cuda_runtime.h>

constexpr int BT = 16, MAX_KK = 31 * 31;
__constant__ float c_g[MAX_KK];

__global__ void blur_same(const float* __restrict__ in, float* __restrict__ out, int H, int W, int K) {
  extern __shared__ float tile[];
  const int R = K / 2, TW = BT + K - 1;
  const int y0 = blockIdx.y * BT - R, x0 = blockIdx.x * BT - R;   // tile origin includes the left/top halo
  for (int y = threadIdx.y; y < TW; y += BT)
    for (int x = threadIdx.x; x < TW; x += BT) {
      const int gy = y0 + y, gx = x0 + x;
      tile[y * TW + x] = (gy >= 0 && gy < H && gx >= 0 && gx < W) ? in[size_t(gy) * W + gx] : 0.0f;  // zero padding
    }
  __syncthreads();
  const int oy = blockIdx.y * BT + threadIdx.y, ox = blockIdx.x * BT + threadIdx.x;
  if (oy >= H || ox >= W) return;
  float s = 0.0f;
  for (int i = 0; i < K; ++i)
    for (int j = 0; j < K; ++j) s += tile[(threadIdx.y + i) * TW + threadIdx.x + j] * c_g[i * K + j];
  out[size_t(oy) * W + ox] = s;
}

void solve(const float* input, const float* kernel, float* output, int H, int W, int K) {
  if (K * K > MAX_KK || K % 2 == 0) return;          // odd, centered kernels only (harness convention)
  cudaMemcpyToSymbol(c_g, kernel, sizeof(float) * K * K, 0, cudaMemcpyDeviceToDevice);
  dim3 block(BT, BT), grid((W + BT - 1) / BT, (H + BT - 1) / BT);
  blur_same<<<grid, block, sizeof(float) * (BT + K - 1) * (BT + K - 1)>>>(input, output, H, W, K);
}
