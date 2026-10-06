// ops.hpp — P0.5 STARTER. Implement rmsnorm, rope, attention and sample_top_p (search for TODO).
// matvec, softmax, silu and argmax are given. The reference is platform/engine/v0/src/ops.hpp — try not to peek.
// Every op is plain C++ over float pointers so it can be read top to bottom; matvec is the only SIMD/threaded op
// because it is ~all of the time (P0.5 lesson §3).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "simd.hpp"
#include "thread_pool.hpp"

namespace ops {

// y[o] = Σ_i W[o, i] x[i], W row-major [out, in]. Rows are split across the pool; each row is a SIMD dot product
// with 4 accumulators (P0.4: one accumulator would be FMA-latency-bound).
inline void matvec(const float* W, const float* x, float* y, int out, int in, s2s::ThreadPool& pool) {
  pool.parallel_for(out, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const float* w = W + r * in;
      simd::vf a0 = simd::zero(), a1 = simd::zero(), a2 = simd::zero(), a3 = simd::zero();
      int i = 0;
      for (; i + 4 * simd::W <= in; i += 4 * simd::W) {
        a0 = simd::fma(simd::load(w + i), simd::load(x + i), a0);
        a1 = simd::fma(simd::load(w + i + simd::W), simd::load(x + i + simd::W), a1);
        a2 = simd::fma(simd::load(w + i + 2 * simd::W), simd::load(x + i + 2 * simd::W), a2);
        a3 = simd::fma(simd::load(w + i + 3 * simd::W), simd::load(x + i + 3 * simd::W), a3);
      }
      float s = simd::hsum(simd::add(simd::add(a0, a1), simd::add(a2, a3)));
      for (; i < in; ++i) s += w[i] * x[i];
      y[r] = s;
    }
  });
}

// out = x / rms(x) * w, rms(x) = sqrt(mean(x²) + eps). Accumulate in double: n can be 4096+ and this keeps the
// C++ result within ~1e-6 of NumPy's float32 (which uses pairwise summation).
inline void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
  // TODO (exercise 1): out[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]. Accumulate the sum of squares in double.
  (void)out; (void)x; (void)w; (void)n; (void)eps;
}

// RoPE, HF "rotate_half" convention: dimension i pairs with i + head_dim/2, angle = pos * theta^(-2i/head_dim).
// Applied in place to `n_heads` consecutive heads of size head_dim.
inline void rope(float* v, int n_heads, int head_dim, int pos, float theta) {
  // TODO (exercise 1): rotate_half RoPE, in place, for n_heads consecutive heads.
  // For i in [0, head_dim/2): angle = pos * theta^(-2i/head_dim); pair (x[i], x[i + head_dim/2]).
  (void)v; (void)n_heads; (void)head_dim; (void)pos; (void)theta;
}

inline void softmax(float* x, int n) {
  const float m = *std::max_element(x, x + n);  // subtract the max so exp() never overflows
  double sum = 0;
  for (int i = 0; i < n; ++i) { x[i] = std::exp(x[i] - m); sum += x[i]; }
  const float inv = float(1.0 / sum);
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

// Causal attention for one query position against cached K/V for positions [0, pos].
// q: [n_heads, hd]; k_cache/v_cache: [max_seq, n_kv_heads * hd] for this layer; out: [n_heads, hd].
// GQA: query head h reads KV head h / (n_heads / n_kv_heads). scratch: >= pos + 1 floats.
inline void attention(const float* q, const float* k_cache, const float* v_cache, float* out, int pos, int n_heads,
                      int n_kv_heads, int hd, float* scratch) {
  // TODO (exercise 2): for each query head h, scores over t = 0..pos against KV head h / (n_heads / n_kv_heads),
  // scaled by 1/sqrt(hd), softmax (use ops::softmax on scratch), then the weighted sum of V rows into out.
  (void)q; (void)k_cache; (void)v_cache; (void)out; (void)pos; (void)n_heads; (void)n_kv_heads; (void)hd; (void)scratch;
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

inline int argmax(const float* x, int n) { return int(std::max_element(x, x + n) - x); }

// Temperature + nucleus (top-p) sampling. u ∈ [0, 1) is supplied by the caller so tests can be deterministic.
// Steps: softmax(logits / T), sort descending, keep the smallest prefix with cumulative prob >= top_p, renormalize,
// then invert the CDF at u. temperature <= 0 means greedy.
inline int sample_top_p(const float* logits, int n, float temperature, float top_p, float u) {
  // TODO (exercise 4): temperature <= 0 -> argmax. Otherwise softmax(logits / T), sort descending (tie-break by
  // index), keep the smallest prefix with cumulative probability >= top_p, and return the token where u * kept_mass
  // falls in the kept prefix's CDF.
  (void)temperature; (void)top_p; (void)u;
  return argmax(logits, n);
}

}  // namespace ops
