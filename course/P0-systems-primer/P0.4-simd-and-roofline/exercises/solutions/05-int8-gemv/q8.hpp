// Exercise 5 solution: q8 blocks (32 x int8 + fp32 scale), integer dot per block, scales applied in float.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#endif

struct BlockQ8 { float scale; int8_t q[32]; };

inline std::vector<BlockQ8> quantize(const float* x, int n) {
  std::vector<BlockQ8> out(size_t(n / 32));
  for (int b = 0; b < n / 32; ++b) {
    const float* xb = x + 32 * b;
    float amax = 0;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(xb[i]));
    // Symmetric range [-127, 127] (not -128): keeps the AVX2 maddubs path free of int16 saturation.
    const float scale = amax / 127.0f, inv = scale > 0 ? 1.0f / scale : 0.0f;
    out[size_t(b)].scale = scale;
    for (int i = 0; i < 32; ++i)
      out[size_t(b)].q[i] = int8_t(std::clamp<long>(std::lround(xb[i] * inv), -127, 127));
  }
  return out;
}

inline int32_t dot32(const int8_t* a, const int8_t* b) {
#if defined(__AVX2__)
  __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
  __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
  __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(va, va), _mm256_sign_epi8(vb, va)),
                                _mm256_set1_epi16(1));
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
  return _mm_cvtsi128_si32(s);
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  int32x4_t acc = vdotq_s32(vdupq_n_s32(0), vld1q_s8(a), vld1q_s8(b));
  acc = vdotq_s32(acc, vld1q_s8(a + 16), vld1q_s8(b + 16));
  return vaddvq_s32(acc);
#else
  int32_t s = 0;
  for (int i = 0; i < 32; ++i) s += int32_t(a[i]) * int32_t(b[i]);
  return s;
#endif
}

inline void gemv_q8(const std::vector<BlockQ8>& W, int rows, int cols, const std::vector<BlockQ8>& x, float* y) {
  const int nb = cols / 32;
  for (int r = 0; r < rows; ++r) {
    float acc = 0;
    for (int b = 0; b < nb; ++b) {
      const BlockQ8& w = W[size_t(r) * nb + b];
      acc += w.scale * x[size_t(b)].scale * float(dot32(w.q, x[size_t(b)].q));  // integer sum, then 1 float mul
    }
    y[r] = acc;
  }
}
