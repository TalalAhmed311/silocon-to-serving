// LeetGPU #50 RMS Normalization — Lane B L4 solution. y = x / sqrt(mean(x²) + eps) · w, rows × cols fp32.
// Our signature (with a weight vector); check the statement for eps and whether a weight/gamma is applied.
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

__global__ void rmsnorm_rows(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int cols, float eps) {
  __shared__ float s[32];
  const float* row = x + (size_t)blockIdx.x * cols;
  float* out = y + (size_t)blockIdx.x * cols;
  float ss = 0.f;
  if ((cols & 3) == 0) {                                     // float4 path: 16-byte loads
    const float4* r4 = reinterpret_cast<const float4*>(row);
    for (int c = threadIdx.x; c < cols / 4; c += blockDim.x) { const float4 v = r4[c]; ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w; }
  } else {
    for (int c = threadIdx.x; c < cols; c += blockDim.x) ss += row[c] * row[c];
  }
  const float r = rsqrtf(block_sum(ss, s) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) out[c] = row[c] * r * w[c];
}

void solve(const float* input, const float* weight, float* output, int rows, int cols, float eps) {
  rmsnorm_rows<<<rows, cols >= 2048 ? 512 : 256>>>(input, weight, output, cols, eps);
}
