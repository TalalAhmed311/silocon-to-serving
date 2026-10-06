// 05_int8_dot.cpp — int8 dot product with int32 accumulation: scalar, AVX2 (maddubs + sign trick), NEON (vdotq).
// Run:      ./build/examples/05_int8_dot
// Expected: identical integer results for every path; SIMD several times faster than scalar.
// Hardware: T0. The AVX2 path mirrors the core of llama.cpp's ggml_vec_dot_q8_0_q8_0 in
//           ggml/src/ggml-cpu/arch/x86/quants.c (ggml-org/llama.cpp@51ce9c11, MIT) — written from scratch here.
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <s2s/bench.hpp>
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static int32_t dot_scalar(const int8_t* a, const int8_t* b, size_t n) {
  int32_t s = 0;
  for (size_t i = 0; i < n; ++i) s += int32_t(a[i]) * int32_t(b[i]);
  return s;
}

#if defined(__AVX2__)
// maddubs multiplies UNSIGNED bytes by SIGNED bytes. Trick: |a| is unsigned, and sign(a) is moved onto b,
// so |a| * (b * sign(a)) == a * b. Then madd(…, 1) widens adjacent int16 pairs to int32.
static int32_t dot_avx2(const int8_t* a, const int8_t* b, size_t n) {
  __m256i acc = _mm256_setzero_si256();
  const __m256i ones = _mm256_set1_epi16(1);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
    __m256i ua = _mm256_sign_epi8(va, va);       // |a|  (as unsigned bytes)
    __m256i sb = _mm256_sign_epi8(vb, va);       // b with a's sign (and 0 where a == 0)
    __m256i p16 = _mm256_maddubs_epi16(ua, sb);  // 16 x int16: |a0|*sb0 + |a1|*sb1, ...
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, ones));  // widen to 8 x int32 and accumulate
  }
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
  int32_t r = _mm_cvtsi128_si32(s);
  for (; i < n; ++i) r += int32_t(a[i]) * int32_t(b[i]);
  return r;
}
// Caveat: maddubs saturates int16. With a = b = -128 in two adjacent lanes, 128*128*2 = 32768 overflows.
// q8_0 quantizes to [-127, 127], which keeps 2 * 127 * 127 = 32258 < 32767 — that is why the range matters.
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static int32_t dot_neon(const int8_t* a, const int8_t* b, size_t n) {
  int32x4_t acc = vdupq_n_s32(0);
  size_t i = 0;
  for (; i + 16 <= n; i += 16) acc = vdotq_s32(acc, vld1q_s8(a + i), vld1q_s8(b + i));  // 4 lanes x (4 products)
  int32_t r = vaddvq_s32(acc);
  for (; i < n; ++i) r += int32_t(a[i]) * int32_t(b[i]);
  return r;
}
#endif

int main() {
  const size_t n = 1 << 16;
  std::vector<int8_t> a(n), b(n);
  std::mt19937 rng(0);
  for (size_t i = 0; i < n; ++i) { a[i] = int8_t(int(rng() % 255) - 127); b[i] = int8_t(int(rng() % 255) - 127); }
  int32_t want = dot_scalar(a.data(), b.data(), n);
  auto ts = s2s::time_fn([&] { s2s::do_not_optimize(dot_scalar(a.data(), b.data(), n)); }, 5, 50);
  std::printf("| path | result | median µs |\n|---|---|---|\n| scalar | %d | %.1f |\n", want, ts.median_ms * 1e3);
#if defined(__AVX2__)
  int32_t got = dot_avx2(a.data(), b.data(), n);
  auto tv = s2s::time_fn([&] { s2s::do_not_optimize(dot_avx2(a.data(), b.data(), n)); }, 5, 50);
  std::printf("| AVX2 maddubs | %d %s | %.1f |\n", got, got == want ? "(match)" : "(MISMATCH)", tv.median_ms * 1e3);
#endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  int32_t gn = dot_neon(a.data(), b.data(), n);
  auto tn = s2s::time_fn([&] { s2s::do_not_optimize(dot_neon(a.data(), b.data(), n)); }, 5, 50);
  std::printf("| NEON vdotq | %d %s | %.1f |\n", gn, gn == want ? "(match)" : "(MISMATCH)", tn.median_ms * 1e3);
#endif
  return 0;
}
