// LeetGPU #83 Fused Residual Add and RMS Norm — Lane B L4 solution. Per row: r = x + residual (written back to
// residual_out), y = rmsnorm(r) · w. One read of x and residual, one write each of residual_out and y. fp32.
// Our signature; check the statement for which outputs are expected and whether residual is updated in place.
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

__global__ void fused_add_rmsnorm(const float* __restrict__ x, const float* __restrict__ res, const float* __restrict__ w,
                                  float* __restrict__ res_out, float* __restrict__ y, int cols, float eps) {
  __shared__ float s[32];
  extern __shared__ float row[];                                // the summed row stays on-chip between the two phases
  const size_t base = (size_t)blockIdx.x * cols;
  float ss = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = x[base + c] + res[base + c];
    res_out[base + c] = v;
    row[c] = v;
    ss += v * v;
  }
  const float r = rsqrtf(block_sum(ss, s) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) y[base + c] = row[c] * r * w[c];
}

void solve(const float* input, const float* residual, const float* weight, float* residual_out, float* output,
           int rows, int cols, float eps) {
  // cols ≤ 12288 so the row fits in 48 KB of shared memory
  fused_add_rmsnorm<<<rows, cols >= 2048 ? 512 : 256, cols * sizeof(float)>>>(input, residual, weight, residual_out, output, cols, eps);
}
