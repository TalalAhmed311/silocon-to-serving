// d4/elementwise.cuh — grid-stride vector add (P5.1) and a float4 version (P5.2).
#pragma once
#include "common.cuh"

namespace d4 {

// One thread per element would need N threads; a grid-stride loop works for ANY N with a fixed, GPU-sized grid.
__global__ void vadd_grid_stride(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    c[i] = a[i] + b[i];
}

// 16-byte loads/stores: a quarter of the memory instructions. Requires 16-byte aligned pointers (cudaMalloc gives 256).
__global__ void vadd_float4(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, long long n) {
  const long long n4 = n / 4, stride = (long long)gridDim.x * blockDim.x;
  const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  for (long long i = t; i < n4; i += stride) {
    const float4 x = reinterpret_cast<const float4*>(a)[i], y = reinterpret_cast<const float4*>(b)[i];
    reinterpret_cast<float4*>(c)[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
  }
  for (long long i = 4 * n4 + t; i < n; i += stride) c[i] = a[i] + b[i];   // tail
}

inline void vadd(const float* a, const float* b, float* c, long long n, bool vec4 = true, cudaStream_t s = 0) {
  const int blocks = sm_count() * 8;
  if (vec4) vadd_float4<<<blocks, 256, 0, s>>>(a, b, c, n);
  else vadd_grid_stride<<<blocks, 256, 0, s>>>(a, b, c, n);
}

}  // namespace d4
