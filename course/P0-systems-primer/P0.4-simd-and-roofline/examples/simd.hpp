// simd.hpp — a 30-line portability layer: the same kernel source compiles to AVX2+FMA, NEON, or scalar.
// It deliberately exposes only what the course needs, so the intrinsic names stay visible in the lesson.
#pragma once
#include <cstdint>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
namespace simd {
constexpr int W = 8;                                   // fp32 lanes
using vf = __m256;
inline vf zero() { return _mm256_setzero_ps(); }
inline vf set1(float x) { return _mm256_set1_ps(x); }
inline vf load(const float* p) { return _mm256_loadu_ps(p); }   // unaligned-safe; same speed when aligned
inline void store(float* p, vf v) { _mm256_storeu_ps(p, v); }
inline vf add(vf a, vf b) { return _mm256_add_ps(a, b); }
inline vf mul(vf a, vf b) { return _mm256_mul_ps(a, b); }
inline vf fma(vf a, vf b, vf c) { return _mm256_fmadd_ps(a, b, c); }   // a*b + c, one rounding
inline float hsum(vf v) {                              // 8 -> 4 -> 2 -> 1
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 sh = _mm_movehdup_ps(lo);
  __m128 s = _mm_add_ps(lo, sh);
  sh = _mm_movehl_ps(sh, s);
  return _mm_cvtss_f32(_mm_add_ss(s, sh));
}
inline const char* name() { return "AVX2+FMA (8 x fp32)"; }
}  // namespace simd
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
namespace simd {
constexpr int W = 4;
using vf = float32x4_t;
inline vf zero() { return vdupq_n_f32(0.0f); }
inline vf set1(float x) { return vdupq_n_f32(x); }
inline vf load(const float* p) { return vld1q_f32(p); }
inline void store(float* p, vf v) { vst1q_f32(p, v); }
inline vf add(vf a, vf b) { return vaddq_f32(a, b); }
inline vf mul(vf a, vf b) { return vmulq_f32(a, b); }
inline vf fma(vf a, vf b, vf c) { return vfmaq_f32(c, a, b); }     // note NEON's argument order: c + a*b
inline float hsum(vf v) { return vaddvq_f32(v); }
inline const char* name() { return "NEON (4 x fp32)"; }
}  // namespace simd
#else
namespace simd {
constexpr int W = 1;
struct vf { float x; };
inline vf zero() { return {0.0f}; }
inline vf set1(float x) { return {x}; }
inline vf load(const float* p) { return {*p}; }
inline void store(float* p, vf v) { *p = v.x; }
inline vf add(vf a, vf b) { return {a.x + b.x}; }
inline vf mul(vf a, vf b) { return {a.x * b.x}; }
inline vf fma(vf a, vf b, vf c) { return {a.x * b.x + c.x}; }
inline float hsum(vf v) { return v.x; }
inline const char* name() { return "scalar fallback (no AVX2/NEON detected)"; }
}  // namespace simd
#endif
