// LeetGPU #42 2D Max Pooling — Lane B L2 solution (harness convention: k×k window, stride s, no padding).
#include <cuda_runtime.h>
#include <math.h>

__global__ void maxpool2d(const float* __restrict__ in, float* __restrict__ out, int H, int W, int k, int s, int OH, int OW) {
  const int ox = blockIdx.x * blockDim.x + threadIdx.x, oy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ox >= OW || oy >= OH) return;
  float m = -INFINITY;
  const float* base = in + size_t(oy) * s * W + size_t(ox) * s;
  for (int i = 0; i < k; ++i)
    for (int j = 0; j < k; ++j) m = fmaxf(m, base[size_t(i) * W + j]);
  out[size_t(oy) * OW + ox] = m;
}

void solve(const float* input, float* output, int H, int W, int k, int s) {
  const int OH = (H - k) / s + 1, OW = (W - k) / s + 1;
  if (OH <= 0 || OW <= 0) return;
  dim3 block(32, 8), grid((OW + 31) / 32, (OH + 7) / 8);
  maxpool2d<<<grid, block>>>(input, output, H, W, k, s, OH, OW);
}
