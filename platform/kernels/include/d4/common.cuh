// d4/common.cuh — small device helpers shared by every D4 kernel.
#pragma once
#include <cuda_runtime.h>

#include <cfloat>

namespace d4 {

constexpr unsigned FULL_MASK = 0xffffffffu;

__host__ __device__ constexpr int ceil_div(long long a, long long b) { return int((a + b - 1) / b); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL_MASK, v, o);   // xor: every lane ends with the total
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, o));
  return v;
}

// Block-wide reductions; the result is returned to EVERY thread. blockDim.x must be a multiple of 32 (≤ 1024).
// `scratch` must hold 32 floats of shared memory; calling twice in a row is safe (a __syncthreads guards reuse).
__device__ __forceinline__ float block_sum(float v, float* scratch) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_sum(v);
  __syncthreads();                         // previous use of scratch is finished
  if (lane == 0) scratch[w] = v;
  __syncthreads();
  v = lane < nw ? scratch[lane] : 0.f;
  return warp_sum(v);
}

__device__ __forceinline__ float block_max(float v, float* scratch) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_max(v);
  __syncthreads();
  if (lane == 0) scratch[w] = v;
  __syncthreads();
  v = lane < nw ? scratch[lane] : -FLT_MAX;
  return warp_max(v);
}

inline int sm_count() {
  int dev = 0, n = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  return n;
}

inline int compute_capability() {
  int dev = 0, major = 0, minor = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  return major * 10 + minor;
}

}  // namespace d4
