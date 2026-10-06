// LeetGPU #17 Dot Product — Lane B L3 solution. result[0] = sum(A[i] * B[i]). Fused map + reduce: one pass over both inputs.
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

__global__ void dot_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ out, int N) {
  float acc = 0.f;
  const int n4 = N >> 2;
  const float4* a4 = reinterpret_cast<const float4*>(A);
  const float4* b4 = reinterpret_cast<const float4*>(B);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
    const float4 a = a4[i], b = b4[i];
    acc = fmaf(a.x, b.x, acc); acc = fmaf(a.y, b.y, acc); acc = fmaf(a.z, b.z, acc); acc = fmaf(a.w, b.w, acc);
  }
  for (int i = (n4 << 2) + blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) acc = fmaf(A[i], B[i], acc);
  acc = block_sum(acc);
  if (threadIdx.x == 0) atomicAdd(out, acc);
}

void solve(const float* A, const float* B, float* result, int N) {
  cudaMemset(result, 0, sizeof(float));
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int blocks = sms * 8;
  const int need = (N / 4 + 255) / 256;
  if (need < blocks) blocks = need > 0 ? need : 1;
  dot_kernel<<<blocks, 256>>>(A, B, result, N);
}
