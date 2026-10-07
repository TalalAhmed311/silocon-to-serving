// LeetGPU #5 Softmax — Lane B L4 solution (exit check). Row-wise softmax of a rows × cols fp32 matrix.
// Our signature (rows × cols); LeetGPU's may be a single vector (rows = 1): check the statement.
// One block per row; online (max, sum) in ONE read, then one normalising read+write.
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

__global__ void softmax_online(const float* __restrict__ x, float* __restrict__ y, int cols) {
  __shared__ float sm[32], sd[32];
  const float* row = x + (size_t)blockIdx.x * cols;
  float m = -INFINITY, d = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = row[c];
    if (v > m) { d = d * __expf(m - v) + 1.f; m = v; } else d += __expf(v - m);
  }
  // merge (m, d) across the block: max first, then rescale each partial sum to the global max
  const float M = block_max(m, sm);
  const float D = block_sum(m == -INFINITY ? 0.f : d * __expf(m - M), sd);
  const float inv = 1.f / D;
  float* out = y + (size_t)blockIdx.x * cols;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) out[c] = __expf(row[c] - M) * inv;
}

void solve(const float* input, float* output, int rows, int cols) {
  softmax_online<<<rows, cols >= 4096 ? 1024 : 256>>>(input, output, cols);
}
