// LeetGPU #27 Mean Squared Error — Lane B L3 solution. mse[0] = mean((predictions - targets)^2).
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float block_sum(float v) {
  __shared__ float warp_part[32];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_sum(v);
  if (lane == 0) warp_part[w] = v;
  __syncthreads();
  v = (threadIdx.x < nw) ? warp_part[lane] : 0.f;
  if (w == 0) v = warp_sum(v);
  return v;
}

// Pass 1: one partial sum per block (no atomics: deterministic). Pass 2: one block sums the partials and divides.
__global__ void sq_err_partials(const float* __restrict__ p, const float* __restrict__ t, float* __restrict__ partial, int N) {
  float acc = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
    const float d = p[i] - t[i];
    acc = fmaf(d, d, acc);
  }
  acc = block_sum(acc);
  if (threadIdx.x == 0) partial[blockIdx.x] = acc;
}

__global__ void finalize_mean(const float* __restrict__ partial, int n_partial, float* __restrict__ out, int N) {
  float acc = 0.f;
  for (int i = threadIdx.x; i < n_partial; i += blockDim.x) acc += partial[i];
  acc = block_sum(acc);
  if (threadIdx.x == 0) out[0] = acc / float(N);
}

void solve(const float* predictions, const float* targets, float* mse, int N) {
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int blocks = sms * 8;
  const int need = (N + 255) / 256;
  if (need < blocks) blocks = need > 0 ? need : 1;
  float* partial = nullptr;
  cudaMalloc(&partial, size_t(blocks) * sizeof(float));      // production: preallocate once, reuse
  sq_err_partials<<<blocks, 256>>>(predictions, targets, partial, N);
  finalize_mean<<<1, 1024>>>(partial, blocks, mse, N);
  cudaFree(partial);
}
