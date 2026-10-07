// d4/reduce.cuh — the P5.4 sum-reduction ladder (after Mark Harris, "Optimizing Parallel Reduction in CUDA"),
// modernised: rung 5 uses warp shuffles instead of the volatile-smem unroll. Every rung computes out[0] = Σ in[0..n).
// Rungs 1–5 write one partial per block and finish with `finish_partials`; rung 6 is single-pass with one atomic.
#pragma once
#include "common.cuh"

namespace d4 {

constexpr int RED_THREADS = 256;

// Rung 1: interleaved addressing, divergent branch (tid % 2s): most lanes idle in every step, the warp still runs.
__global__ void reduce_r1(const float* __restrict__ in, float* __restrict__ partial, int n) {
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

// Rung 2: interleaved addressing with a strided index — no divergence, but s[2·st·tid] causes bank conflicts.
__global__ void reduce_r2(const float* __restrict__ in, float* __restrict__ partial, int n) {
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

// Rung 3: sequential addressing — active threads are contiguous (whole warps retire) and accesses conflict-free.
__global__ void reduce_r3(const float* __restrict__ in, float* __restrict__ partial, int n) {
  extern __shared__ float s[];
  const unsigned tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
  s[tid] = i < n ? in[i] : 0.f;
  __syncthreads();
  for (unsigned st = blockDim.x / 2; st > 0; st >>= 1) {
    if (tid < st) s[tid] += s[tid + st];
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = s[0];
}

// Rung 4: first add during load — each block covers 2·blockDim elements, so half the threads aren't idle at step 1.
__global__ void reduce_r4(const float* __restrict__ in, float* __restrict__ partial, int n) {
  extern __shared__ float s[];
  const unsigned tid = threadIdx.x, i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  float v = i < n ? in[i] : 0.f;
  if (i + blockDim.x < n) v += in[i + blockDim.x];
  s[tid] = v;
  __syncthreads();
  for (unsigned st = blockDim.x / 2; st > 0; st >>= 1) {
    if (tid < st) s[tid] += s[tid + st];
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = s[0];
}

// Rung 5: rung 4 + the last 5 steps (and the cross-warp step) done with warp shuffles: no __syncthreads, no smem.
__global__ void reduce_r5(const float* __restrict__ in, float* __restrict__ partial, int n) {
  __shared__ float scratch[32];
  const unsigned i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  float v = i < n ? in[i] : 0.f;
  if (i + blockDim.x < n) v += in[i + blockDim.x];
  v = block_sum(v, scratch);
  if (threadIdx.x == 0) partial[blockIdx.x] = v;
}

// Rung 6: grid-stride with float4 loads (many elements per thread, all in registers), shuffles, ONE atomic per block.
__global__ void reduce_r6(const float* __restrict__ in, float* __restrict__ out, long long n) {
  __shared__ float scratch[32];
  float acc = 0.f;
  const long long n4 = n >> 2, stride = (long long)gridDim.x * blockDim.x;
  const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const float4* in4 = reinterpret_cast<const float4*>(in);
  for (long long k = t; k < n4; k += stride) { const float4 v = in4[k]; acc += (v.x + v.y) + (v.z + v.w); }
  for (long long k = 4 * n4 + t; k < n; k += stride) acc += in[k];
  acc = block_sum(acc, scratch);
  if (threadIdx.x == 0) atomicAdd(out, acc);
}

// One block sums the partials (a few thousand at most) and writes out[0] — no atomics, deterministic.
__global__ void finish_partials(const float* __restrict__ partial, int m, float* __restrict__ out) {
  __shared__ float scratch[32];
  float acc = 0.f;
  for (int k = threadIdx.x; k < m; k += blockDim.x) acc += partial[k];
  acc = block_sum(acc, scratch);
  if (threadIdx.x == 0) out[0] = acc;
}

// Runs rung r (1..6). `partial` must hold ceil(n / RED_THREADS) floats (rungs 1–5); unused for rung 6.
inline void reduce_sum(int rung, const float* in, float* out, float* partial, long long n, cudaStream_t st = 0) {
  const int T = RED_THREADS;
  if (rung == 6) {
    cudaMemsetAsync(out, 0, sizeof(float), st);
    int blocks = sm_count() * 8;
    const int need = ceil_div(n / 4 + 1, T);
    if (need < blocks) blocks = need;
    reduce_r6<<<blocks, T, 0, st>>>(in, out, n);
    return;
  }
  const int per_block = (rung >= 4) ? 2 * T : T;
  const int blocks = ceil_div(n, per_block);
  const size_t smem = T * sizeof(float);
  switch (rung) {
    case 1: reduce_r1<<<blocks, T, smem, st>>>(in, partial, int(n)); break;
    case 2: reduce_r2<<<blocks, T, smem, st>>>(in, partial, int(n)); break;
    case 3: reduce_r3<<<blocks, T, smem, st>>>(in, partial, int(n)); break;
    case 4: reduce_r4<<<blocks, T, smem, st>>>(in, partial, int(n)); break;
    default: reduce_r5<<<blocks, T, 0, st>>>(in, partial, int(n)); break;
  }
  finish_partials<<<1, 1024, 0, st>>>(partial, blocks, out);
}

}  // namespace d4
