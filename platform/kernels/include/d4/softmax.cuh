// d4/softmax.cuh — row-wise softmax over a rows × cols fp32 matrix, three ways (P5.5):
//   v1 safe softmax, 3 passes over global memory: max → Σ exp(x − max) → normalize
//   v2 online softmax (Milakov & Gimelshein 2018): ONE pass computes (max, sum) together → normalize: 2 reads + 1 write
//   v3 single read: one warp per row holds the row in registers (cols ≤ kMaxColsV3) → 1 read + 1 write
// One block (v1, v2) or one warp (v3) per row; fp32 accumulation. L4 exit check: report it in GB/s.
#pragma once
#include "common.cuh"

namespace d4 {

struct MD { float m, d; };   // running max and running Σ exp(x − m)

__device__ __forceinline__ MD md_merge(MD a, MD b) {
  const float m = fmaxf(a.m, b.m);
  // exp(-inf - (-inf)) is NaN: guard the empty-segment case
  const float da = a.m == -INFINITY ? 0.f : a.d * __expf(a.m - m);
  const float db = b.m == -INFINITY ? 0.f : b.d * __expf(b.m - m);
  return {m, da + db};
}

__device__ __forceinline__ MD warp_md(MD v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    MD u{__shfl_xor_sync(FULL_MASK, v.m, o), __shfl_xor_sync(FULL_MASK, v.d, o)};
    v = md_merge(v, u);
  }
  return v;
}

__device__ __forceinline__ MD block_md(MD v, float* sm, float* sd) {   // sm, sd: 32 floats of shared memory each
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_md(v);
  __syncthreads();
  if (lane == 0) { sm[w] = v.m; sd[w] = v.d; }
  __syncthreads();
  v = lane < nw ? MD{sm[lane], sd[lane]} : MD{-INFINITY, 0.f};
  return warp_md(v);
}

// ---- v1: three passes -------------------------------------------------------------------------------------------
__global__ void softmax_v1(const float* __restrict__ x, float* __restrict__ y, int cols) {
  __shared__ float scratch[32];
  const float* row = x + (size_t)blockIdx.x * cols;
  float* out = y + (size_t)blockIdx.x * cols;
  float m = -INFINITY;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, row[c]);
  m = block_max(m, scratch);
  float s = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) s += __expf(row[c] - m);
  s = block_sum(s, scratch);
  const float inv = 1.f / s;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) out[c] = __expf(row[c] - m) * inv;
}

// ---- v2: online (max, sum) in one pass, then normalize -----------------------------------------------------------
__global__ void softmax_v2(const float* __restrict__ x, float* __restrict__ y, int cols) {
  __shared__ float sm[32], sd[32];
  const float* row = x + (size_t)blockIdx.x * cols;
  float* out = y + (size_t)blockIdx.x * cols;
  MD acc{-INFINITY, 0.f};
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = row[c];
    if (v == -INFINITY) continue;                 // fully masked element (attention masks): contributes nothing
    if (v > acc.m) { acc.d = acc.d * __expf(acc.m - v) + 1.f; acc.m = v; }   // rescale the old sum to the new max
    else acc.d += __expf(v - acc.m);
  }
  acc = block_md(acc, sm, sd);
  const float inv = 1.f / acc.d;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) out[c] = __expf(row[c] - acc.m) * inv;
}

// ---- v3: one warp per row, row held in registers: exactly one read and one write ---------------------------------
template <int VPL>   // values per lane: cols ≤ 32 · VPL
__global__ void softmax_v3(const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= rows) return;
  const float* row = x + (size_t)warp * cols;
  float v[VPL];
  float m = -INFINITY;
#pragma unroll
  for (int k = 0; k < VPL; ++k) {
    const int c = k * 32 + lane;                 // lane-interleaved: each k is a coalesced 128-byte read
    v[k] = c < cols ? row[c] : -INFINITY;
    m = fmaxf(m, v[k]);
  }
  m = warp_max(m);
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < VPL; ++k) { v[k] = __expf(v[k] - m); s += v[k]; }   // exp(-inf) = 0 for the padding
  const float inv = 1.f / warp_sum(s);
  float* out = y + (size_t)warp * cols;
#pragma unroll
  for (int k = 0; k < VPL; ++k) { const int c = k * 32 + lane; if (c < cols) out[c] = v[k] * inv; }
}

constexpr int kMaxColsV3 = 32 * 32;   // 1024 columns per warp-row in registers (32 floats per lane)

inline void softmax(int version, const float* x, float* y, int rows, int cols, cudaStream_t st = 0) {
  if (version == 3 && cols <= kMaxColsV3) {
    const int blocks = ceil_div((long long)rows * 32, 256);
    if (cols <= 256) softmax_v3<8><<<blocks, 256, 0, st>>>(x, y, rows, cols);
    else softmax_v3<32><<<blocks, 256, 0, st>>>(x, y, rows, cols);
    return;
  }
  const int threads = cols >= 1024 ? 1024 : (cols >= 256 ? 256 : 128);
  if (version == 1) softmax_v1<<<rows, threads, 0, st>>>(x, y, cols);
  else softmax_v2<<<rows, threads, 0, st>>>(x, y, cols);   // also the fallback for v3 with long rows
}

}  // namespace d4
