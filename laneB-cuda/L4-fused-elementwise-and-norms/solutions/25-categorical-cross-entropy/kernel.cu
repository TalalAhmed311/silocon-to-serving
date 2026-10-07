// LeetGPU #25 Categorical Cross Entropy Loss — Lane B L4 solution. loss = mean over rows of
// (logsumexp(logits[r]) − logits[r][label[r]]). logits: N × C fp32, labels: N ints, output: 1 float.
// Our signature (mean reduction); check the statement for reduction type.
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o); return v; }
__device__ __forceinline__ float warp_max(float v) { for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o)); return v; }
// block-wide sum/max returned to every thread; blockDim.x multiple of 32; scratch = 32 floats of shared memory
__device__ __forceinline__ float block_sum(float v, float* s) {
  const int l = threadIdx.x & 31, w = threadIdx.x >> 5; v = warp_sum(v); __syncthreads();
  if (l == 0) s[w] = v; __syncthreads(); v = l < (int)(blockDim.x >> 5) ? s[l] : 0.f; return warp_sum(v);
}
__device__ __forceinline__ float block_max(float v, float* s) {
  const int l = threadIdx.x & 31, w = threadIdx.x >> 5; v = warp_max(v); __syncthreads();
  if (l == 0) s[w] = v; __syncthreads(); v = l < (int)(blockDim.x >> 5) ? s[l] : -INFINITY; return warp_max(v);
}

// One block per row: online (max, sum) in one read, label logit picked up in the same pass; atomicAdd of loss/N.
__global__ void ce_rows(const float* __restrict__ logits, const int* __restrict__ labels, float* __restrict__ out, int N, int C) {
  __shared__ float sm[32], sd[32];
  __shared__ float target;
  const float* row = logits + (size_t)blockIdx.x * C;
  const int lab = labels[blockIdx.x];
  float m = -INFINITY, d = 0.f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = row[c];
    if (c == lab) target = v;
    if (v > m) { d = d * __expf(m - v) + 1.f; m = v; } else d += __expf(v - m);
  }
  const float M = block_max(m, sm);
  const float D = block_sum(m == -INFINITY ? 0.f : d * __expf(m - M), sd);    // contains __syncthreads: target visible
  if (threadIdx.x == 0) atomicAdd(out, (M + logf(D) - target) / N);
}

void solve(const float* logits, const int* labels, float* loss, int N, int C) {
  cudaMemset(loss, 0, sizeof(float));
  ce_rows<<<N, C >= 4096 ? 512 : 128>>>(logits, labels, loss, N, C);
}
