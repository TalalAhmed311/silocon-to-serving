// Reference: one block per row; online (max, sum) merged across the block (d4::block_md); the thread that owns the
// target column publishes its logit through shared memory. logsumexp = m + log(d).
#pragma once
#include <d4/softmax.cuh>

__global__ void cross_entropy_k(const float* __restrict__ logits, const int* __restrict__ target, float* __restrict__ loss, int vocab) {
  __shared__ float sm[32], sd[32], tgt;
  const float* row = logits + (size_t)blockIdx.x * vocab;
  const int t = target[blockIdx.x];
  d4::MD acc{-INFINITY, 0.f};
  for (int c = threadIdx.x; c < vocab; c += blockDim.x) {
    const float v = row[c];
    if (c == t) tgt = v;
    if (v > acc.m) { acc.d = acc.d * __expf(acc.m - v) + 1.f; acc.m = v; }
    else acc.d += __expf(v - acc.m);
  }
  acc = d4::block_md(acc, sm, sd);            // contains __syncthreads: tgt is visible afterwards
  if (threadIdx.x == 0) loss[blockIdx.x] = acc.m + logf(acc.d) - tgt;
}

inline void cross_entropy(const float* logits, const int* target, float* loss, int rows, int vocab) {
  cross_entropy_k<<<rows, 512>>>(logits, target, loss, vocab);
}
