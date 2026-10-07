// LeetGPU #11 3D Convolution — Lane B L2 solution (direct, constant-memory kernel; streaming-in-z is the stretch goal).
#include <cuda_runtime.h>

constexpr int MAX_K3 = 9 * 9 * 9;
__constant__ float c_k3[MAX_K3];

__global__ void conv3d(const float* __restrict__ in, float* __restrict__ out, int D, int H, int W, int KD, int KH, int KW) {
  const int OD = D - KD + 1, OH = H - KH + 1, OW = W - KW + 1;
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, z = blockIdx.z;
  if (x >= OW || y >= OH || z >= OD) return;
  float s = 0.0f;
  for (int a = 0; a < KD; ++a)
    for (int b = 0; b < KH; ++b)
      for (int c = 0; c < KW; ++c) s += in[(size_t(z + a) * H + y + b) * W + x + c] * c_k3[(a * KH + b) * KW + c];
  out[(size_t(z) * OH + y) * OW + x] = s;
}

void solve(const float* input, const float* kernel, float* output, int D, int H, int W, int KD, int KH, int KW) {
  if (KD * KH * KW > MAX_K3) return;
  cudaMemcpyToSymbol(c_k3, kernel, sizeof(float) * KD * KH * KW, 0, cudaMemcpyDeviceToDevice);
  const int OD = D - KD + 1, OH = H - KH + 1, OW = W - KW + 1;
  dim3 block(32, 8), grid((OW + 31) / 32, (OH + 7) / 8, OD);
  conv3d<<<grid, block>>>(input, output, D, H, W, KD, KH, KW);
}
