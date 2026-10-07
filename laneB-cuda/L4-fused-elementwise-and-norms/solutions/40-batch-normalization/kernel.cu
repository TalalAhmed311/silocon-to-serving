// LeetGPU #40 Batch Normalization — Lane B L4 solution (training-mode forward). x: [N, C] row-major fp32;
// per channel c: μ_c, σ²_c over the N rows (biased variance); y = (x − μ_c)/sqrt(σ²_c + eps)·γ_c + β_c.
// Our layout ([N, C]; for [N, C, H, W] the reduction is over N·H·W per channel) — check the statement.
#include <cuda_runtime.h>
#include <cmath>

// Block = 32 channels (threadIdx.x, coalesced along a row) × 8 row-strides (threadIdx.y). Sums in fp32 with a shifted
// mean (x − x[0][c]) to tame cancellation when |μ| ≫ σ.
__global__ void bn_stats(const float* __restrict__ x, float* __restrict__ mean, float* __restrict__ rstd, int N, int C, float eps) {
  __shared__ float ss[8][33], sq[8][33];
  const int c = blockIdx.x * 32 + threadIdx.x;
  float s = 0.f, q = 0.f;
  const float shift = c < C ? x[c] : 0.f;
  if (c < C)
    for (int r = threadIdx.y; r < N; r += 8) { const float d = x[(size_t)r * C + c] - shift; s += d; q += d * d; }
  ss[threadIdx.y][threadIdx.x] = s;
  sq[threadIdx.y][threadIdx.x] = q;
  __syncthreads();
  if (threadIdx.y == 0 && c < C) {
    for (int k = 1; k < 8; ++k) { s += ss[k][threadIdx.x]; q += sq[k][threadIdx.x]; }
    const float m = s / N;
    mean[c] = shift + m;
    rstd[c] = rsqrtf(fmaxf(q / N - m * m, 0.f) + eps);
  }
}

__global__ void bn_apply(const float* __restrict__ x, const float* __restrict__ mean, const float* __restrict__ rstd,
                         const float* __restrict__ g, const float* __restrict__ b, float* __restrict__ y, int N, int C) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < (long long)N * C; i += (long long)gridDim.x * blockDim.x) {
    const int c = int(i % C);
    y[i] = (x[i] - mean[c]) * rstd[c] * g[c] + b[c];
  }
}

void solve(const float* input, const float* gamma, const float* beta, float* output, int N, int C, float eps) {
  float *mean, *rstd;
  cudaMalloc(&mean, C * sizeof(float));
  cudaMalloc(&rstd, C * sizeof(float));
  bn_stats<<<(C + 31) / 32, dim3(32, 8)>>>(input, mean, rstd, N, C, eps);
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  bn_apply<<<sms * 8, 256>>>(input, mean, rstd, gamma, beta, output, N, C);
  cudaFree(mean);
  cudaFree(rstd);
}
