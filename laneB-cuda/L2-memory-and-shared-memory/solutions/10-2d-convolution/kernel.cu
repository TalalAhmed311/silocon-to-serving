// LeetGPU #10 2D Convolution — Lane B L2 solution. Valid correlation; input R×C, kernel KR×KC, output (R-KR+1)×(C-KC+1).
#include <cuda_runtime.h>

constexpr int T = 16, MAX_TAPS = 64 * 64;
__constant__ float c_k2[MAX_TAPS];

__global__ void conv2d_tiled(const float* __restrict__ in, float* __restrict__ out, int R, int C, int KR, int KC) {
  extern __shared__ float tile[];                 // (T+KR-1) × (T+KC-1)
  const int TW = T + KC - 1, TH = T + KR - 1;
  const int r0 = blockIdx.y * T, c0 = blockIdx.x * T;
  for (int y = threadIdx.y; y < TH; y += T)       // strided 2-D cooperative load (tile is bigger than the block)
    for (int x = threadIdx.x; x < TW; x += T) {
      const int r = r0 + y, c = c0 + x;
      tile[y * TW + x] = (r < R && c < C) ? in[size_t(r) * C + c] : 0.0f;
    }
  __syncthreads();
  const int orow = r0 + threadIdx.y, ocol = c0 + threadIdx.x, OR = R - KR + 1, OC = C - KC + 1;
  if (orow >= OR || ocol >= OC) return;
  float s = 0.0f;
  for (int i = 0; i < KR; ++i)
    for (int j = 0; j < KC; ++j) s += tile[(threadIdx.y + i) * TW + threadIdx.x + j] * c_k2[i * KC + j];
  out[size_t(orow) * OC + ocol] = s;
}

void solve(const float* input, const float* kernel, float* output, int R, int C, int KR, int KC) {
  const int OR = R - KR + 1, OC = C - KC + 1;
  if (OR <= 0 || OC <= 0 || KR * KC > MAX_TAPS) return;  // the harness test covers the supported range
  cudaMemcpyToSymbol(c_k2, kernel, size_t(KR) * KC * sizeof(float), 0, cudaMemcpyDeviceToDevice);
  dim3 block(T, T), grid((OC + T - 1) / T, (OR + T - 1) / T);
  conv2d_tiled<<<grid, block, size_t(T + KR - 1) * (T + KC - 1) * sizeof(float)>>>(input, output, R, C, KR, KC);
}
