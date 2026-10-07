// Exercise 1 starter: rungs 1–2 are given; write rungs 3–6 (see the README) behind the same launcher.
// `partial` holds ceil(n / 256) floats for the per-block partial sums (rungs 1–5).
#pragma once
#include <d4/common.cuh>

constexpr int RT = 256;   // threads per block

__global__ void r1(const float* __restrict__ in, float* __restrict__ partial, int n) {
  extern __shared__ float s[];
  const unsigned tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
  s[tid] = i < n ? in[i] : 0.f;
  __syncthreads();
  for (unsigned st = 1; st < blockDim.x; st *= 2) {
    if (tid % (2 * st) == 0) s[tid] += s[tid + st];
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = s[0];
}

__global__ void r2(const float* __restrict__ in, float* __restrict__ partial, int n) {
  extern __shared__ float s[];
  const unsigned tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
  s[tid] = i < n ? in[i] : 0.f;
  __syncthreads();
  for (unsigned st = 1; st < blockDim.x; st *= 2) {
    const unsigned idx = 2 * st * tid;
    if (idx < blockDim.x) s[idx] += s[idx + st];
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = s[0];
}

__global__ void finish(const float* __restrict__ partial, int m, float* __restrict__ out) {
  __shared__ float scratch[32];
  float acc = 0.f;
  for (int k = threadIdx.x; k < m; k += blockDim.x) acc += partial[k];
  acc = d4::block_sum(acc, scratch);
  if (threadIdx.x == 0) out[0] = acc;
}

// TODO: r3 (sequential addressing), r4 (first add during load), r5 (warp shuffles), r6 (grid-stride float4 + atomic)

inline void reduce_sum(int rung, const float* in, float* out, float* partial, long long n) {
  const int blocks = d4::ceil_div(n, RT);
  if (rung == 1) r1<<<blocks, RT, RT * sizeof(float)>>>(in, partial, int(n));
  else if (rung == 2) r2<<<blocks, RT, RT * sizeof(float)>>>(in, partial, int(n));
  else return;                                   // TODO: rungs 3–6
  finish<<<1, 1024>>>(partial, blocks, out);
}
