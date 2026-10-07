// d4/norms.cuh — RMSNorm and fused residual-add + RMSNorm (P5.5, #12), fp32 / fp16 / bf16 storage, fp32 math.
//   rmsnorm:          y = x / sqrt(mean(x²) + eps) · w
//   fused_add_rmsnorm: r ← x + r (written back: the next layer's residual), y = rmsnorm(r) · w — ONE read of x and r,
//                      one write of r and y, instead of add (2R+1W) followed by norm (1R+1W).
// One block per row, vectorized 16-byte loads when cols is a multiple of the vector width.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <type_traits>

#include "common.cuh"

namespace d4 {

template <class T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <class T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <class T>
__global__ void rmsnorm_k(const T* __restrict__ x, const T* __restrict__ w, T* __restrict__ y, int cols, float eps) {
  __shared__ float scratch[32];
  const T* row = x + (size_t)blockIdx.x * cols;
  T* out = y + (size_t)blockIdx.x * cols;
  float ss = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) { const float v = to_f(row[c]); ss = fmaf(v, v, ss); }
  ss = block_sum(ss, scratch);
  const float r = rsqrtf(ss / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) out[c] = from_f<T>(to_f(row[c]) * r * to_f(w[c]));
}

// fp32, cols % 4 == 0: float4 loads; the row is read twice (sum of squares, then scale) — the second read usually
// hits L2/L1 for rows up to a few KB. Variant for exercise 2: keep the row in registers instead.
__global__ void rmsnorm_f4(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int cols, float eps) {
  __shared__ float scratch[32];
  const float4* row = reinterpret_cast<const float4*>(x + (size_t)blockIdx.x * cols);
  const float4* w4 = reinterpret_cast<const float4*>(w);
  float4* out = reinterpret_cast<float4*>(y + (size_t)blockIdx.x * cols);
  const int c4 = cols / 4;
  float ss = 0.f;
  for (int c = threadIdx.x; c < c4; c += blockDim.x) { const float4 v = row[c]; ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w; }
  ss = block_sum(ss, scratch);
  const float r = rsqrtf(ss / cols + eps);
  for (int c = threadIdx.x; c < c4; c += blockDim.x) {
    const float4 v = row[c], g = w4[c];
    out[c] = make_float4(v.x * r * g.x, v.y * r * g.y, v.z * r * g.z, v.w * r * g.w);
  }
}

template <class T>
__global__ void fused_add_rmsnorm_k(const T* __restrict__ x, T* __restrict__ resid, const T* __restrict__ w, T* __restrict__ y,
                                    int cols, float eps) {
  __shared__ float scratch[32];
  extern __shared__ float rowbuf[];                       // the summed row, kept on-chip between the two phases
  const size_t base = (size_t)blockIdx.x * cols;
  float ss = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = to_f(x[base + c]) + to_f(resid[base + c]);
    const T vt = from_f<T>(v);
    resid[base + c] = vt;                                 // the residual stream, in its storage precision
    const float vr = to_f(vt);                            // normalize what was STORED (matches unfused numerics)
    rowbuf[c] = vr;
    ss = fmaf(vr, vr, ss);
  }
  ss = block_sum(ss, scratch);
  const float r = rsqrtf(ss / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) y[base + c] = from_f<T>(rowbuf[c] * r * to_f(w[c]));
}

template <class T>
inline void rmsnorm(const T* x, const T* w, T* y, int rows, int cols, float eps = 1e-6f, cudaStream_t st = 0) {
  const int threads = cols >= 2048 ? 512 : 256;
  if constexpr (std::is_same_v<T, float>) {
    if (cols % 4 == 0) { rmsnorm_f4<<<rows, threads, 0, st>>>(x, w, y, cols, eps); return; }
  }
  rmsnorm_k<T><<<rows, threads, 0, st>>>(x, w, y, cols, eps);
}

// cols ≤ 12288 (48 KB of fp32 row buffer); larger hidden sizes need opt-in dynamic smem or a two-read variant.
template <class T>
inline void fused_add_rmsnorm(const T* x, T* resid, const T* w, T* y, int rows, int cols, float eps = 1e-6f, cudaStream_t st = 0) {
  fused_add_rmsnorm_k<T><<<rows, cols >= 2048 ? 512 : 256, cols * sizeof(float), st>>>(x, resid, w, y, cols, eps);
}

}  // namespace d4
