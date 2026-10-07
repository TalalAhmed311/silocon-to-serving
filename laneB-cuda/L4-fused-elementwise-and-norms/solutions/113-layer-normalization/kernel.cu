// LeetGPU #113 Layer Normalization — Lane B L4 solution. y = (x − mean) / sqrt(var + eps) · gamma + beta, per row.
// Welford's online algorithm per thread (numerically stable single pass), merged across the block with Chan's formula.
#include <cuda_runtime.h>
#include <cmath>

struct Wf { float n, mean, m2; };

__device__ __forceinline__ Wf wf_merge(Wf a, Wf b) {
  if (b.n == 0.f) return a;
  if (a.n == 0.f) return b;
  const float n = a.n + b.n, delta = b.mean - a.mean;
  return {n, a.mean + delta * b.n / n, a.m2 + b.m2 + delta * delta * a.n * b.n / n};
}

__device__ __forceinline__ Wf warp_wf(Wf v) {
  for (int o = 16; o > 0; o >>= 1) {
    Wf u{__shfl_xor_sync(0xffffffffu, v.n, o), __shfl_xor_sync(0xffffffffu, v.mean, o), __shfl_xor_sync(0xffffffffu, v.m2, o)};
    v = wf_merge(v, u);
  }
  return v;
}

__global__ void layernorm_rows(const float* __restrict__ x, const float* __restrict__ g, const float* __restrict__ b,
                               float* __restrict__ y, int cols, float eps) {
  __shared__ float sn[32], smu[32], sm2[32];
  const float* row = x + (size_t)blockIdx.x * cols;
  Wf acc{0.f, 0.f, 0.f};
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {        // Welford update, one value at a time
    const float v = row[c];
    acc.n += 1.f;
    const float d = v - acc.mean;
    acc.mean += d / acc.n;
    acc.m2 += d * (v - acc.mean);
  }
  acc = warp_wf(acc);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (lane == 0) { sn[w] = acc.n; smu[w] = acc.mean; sm2[w] = acc.m2; }
  __syncthreads();
  acc = lane < (int)(blockDim.x >> 5) ? Wf{sn[lane], smu[lane], sm2[lane]} : Wf{0.f, 0.f, 0.f};
  acc = warp_wf(acc);                                             // every warp computes the same total
  const float mean = acc.mean, rstd = rsqrtf(acc.m2 / cols + eps);   // population variance (biased), as LayerNorm uses
  float* out = y + (size_t)blockIdx.x * cols;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) out[c] = (row[c] - mean) * rstd * g[c] + b[c];
}

void solve(const float* input, const float* gamma, const float* beta, float* output, int rows, int cols, float eps) {
  layernorm_rows<<<rows, cols >= 2048 ? 512 : 256>>>(input, gamma, beta, output, cols, eps);
}
