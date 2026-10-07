// Exercise 2 solution.
#pragma once
#include "simd.hpp"
#if defined(__AVX2__)
inline float hsum8(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);        // free: just a register rename
  __m128 hi = _mm256_extractf128_ps(v, 1);      // 1
  __m128 s4 = _mm_add_ps(lo, hi);               // 2: 4 lanes
  __m128 sh = _mm_movehdup_ps(s4);              // 3: (s1, s1, s3, s3)
  __m128 s2 = _mm_add_ps(s4, sh);               // 4: lanes 0 and 2 hold pair sums
  sh = _mm_movehl_ps(sh, s2);                   // 5: bring lane 2 down to lane 0
  return _mm_cvtss_f32(_mm_add_ss(s2, sh));     // 6
}
#elif defined(__ARM_NEON)
inline float hsum4(float32x4_t v) {
  float32x4_t p = vpaddq_f32(v, v);             // (v0+v1, v2+v3, v0+v1, v2+v3)
  p = vpaddq_f32(p, p);
  return vgetq_lane_f32(p, 0);
}
#endif
