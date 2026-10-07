// LeetGPU #105 Group Normalization — Lane B L4 solution. x: [N, C, S] (S = spatial size, e.g. H·W), G groups of C/G
// channels; statistics per (n, group) over (C/G)·S CONTIGUOUS elements; per-channel affine γ_c, β_c.
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o); return v; }
__device__ __forceinline__ float block_sum(float v, float* s) {
  const int l = threadIdx.x & 31, w = threadIdx.x >> 5; v = warp_sum(v); __syncthreads();
  if (l == 0) s[w] = v; __syncthreads(); v = l < (int)(blockDim.x >> 5) ? s[l] : 0.f; return warp_sum(v);
}

__global__ void groupnorm_k(const float* __restrict__ x, const float* __restrict__ g, const float* __restrict__ b,
                            float* __restrict__ y, int C, int S, int G, float eps) {
  __shared__ float scratch[32];
  const int n = blockIdx.y, grp = blockIdx.x, cpg = C / G;
  const size_t base = ((size_t)n * C + (size_t)grp * cpg) * S;
  const int len = cpg * S;
  float s = 0.f;
  for (int i = threadIdx.x; i < len; i += blockDim.x) s += x[base + i];
  const float mean = block_sum(s, scratch) / len;
  float q = 0.f;
  for (int i = threadIdx.x; i < len; i += blockDim.x) { const float d = x[base + i] - mean; q += d * d; }   // 2-pass: stable
  const float rstd = rsqrtf(block_sum(q, scratch) / len + eps);
  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    const int c = grp * cpg + i / S;
    y[base + i] = (x[base + i] - mean) * rstd * g[c] + b[c];
  }
}

void solve(const float* input, const float* gamma, const float* beta, float* output, int N, int C, int S, int G, float eps) {
  groupnorm_k<<<dim3(G, N), 256>>>(input, gamma, beta, output, C, S, G, eps);
}
