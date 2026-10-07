// ops.hpp — the math of one Llama decode step (reference solution for P0.5 exercises 1, 2 and 4).
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
  double ss = 0;
  for (int i = 0; i < n; ++i) ss += double(x[i]) * x[i];
  const float inv = float(1.0 / std::sqrt(ss / n + eps));
  for (int i = 0; i < n; ++i) out[i] = x[i] * inv * w[i];
}

// RoPE, HF "rotate_half" convention: dimension i pairs with i + head_dim/2, angle = pos * theta^(-2i/head_dim).
// Applied in place to `n_heads` consecutive heads of size head_dim.
inline void rope(float* v, int n_heads, int head_dim, int pos, float theta) {
  const int half = head_dim / 2;
  for (int i = 0; i < half; ++i) {
    const double freq = std::pow(double(theta), -2.0 * i / head_dim);
    const double ang = pos * freq;
    const float c = float(std::cos(ang)), s = float(std::sin(ang));
    for (int h = 0; h < n_heads; ++h) {
      float* x = v + h * head_dim;
      const float x0 = x[i], x1 = x[i + half];
      x[i] = x0 * c - x1 * s;          // [x0, x1] rotated by ang
      x[i + half] = x1 * c + x0 * s;
    }
  }
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
  const int group = n_heads / n_kv_heads, kv_dim = n_kv_heads * hd;
  const float scale = 1.0f / std::sqrt(float(hd));
  for (int h = 0; h < n_heads; ++h) {
    const float* qh = q + h * hd;
    const int kvh = h / group;
    for (int t = 0; t <= pos; ++t) {
      const float* k = k_cache + t * kv_dim + kvh * hd;
      float s = 0;
      for (int i = 0; i < hd; ++i) s += qh[i] * k[i];
      scratch[t] = s * scale;
    }
    softmax(scratch, pos + 1);
    float* o = out + h * hd;
    std::fill(o, o + hd, 0.0f);
    for (int t = 0; t <= pos; ++t) {
      const float* v = v_cache + t * kv_dim + kvh * hd;
      const float a = scratch[t];
      for (int i = 0; i < hd; ++i) o[i] += a * v[i];
    }
  }
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

inline int argmax(const float* x, int n) { return int(std::max_element(x, x + n) - x); }

// Temperature + nucleus (top-p) sampling. u ∈ [0, 1) is supplied by the caller so tests can be deterministic.
// Steps: softmax(logits / T), sort descending, keep the smallest prefix with cumulative prob >= top_p, renormalize,
// then invert the CDF at u. temperature <= 0 means greedy.
inline int sample_top_p(const float* logits, int n, float temperature, float top_p, float u) {
  if (temperature <= 0.0f) return argmax(logits, n);
  std::vector<float> p(logits, logits + n);
  for (auto& v : p) v /= temperature;
  softmax(p.data(), n);
  std::vector<int> idx(size_t(n));
  std::iota(idx.begin(), idx.end(), 0);
  // Tie-break on index so the result does not depend on the sort implementation.
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return p[size_t(a)] > p[size_t(b)] || (p[size_t(a)] == p[size_t(b)] && a < b); });
  double cum = 0;
  int keep = 0;
  for (; keep < n; ++keep) {
    cum += p[size_t(idx[size_t(keep)])];
    if (cum >= top_p) { ++keep; break; }
  }
  keep = std::min(keep, n);
  double target = u * cum, acc = 0;  // `cum` is the kept mass: sampling within it = renormalizing
  for (int k = 0; k < keep; ++k) {
    acc += p[size_t(idx[size_t(k)])];
    if (target < acc) return idx[size_t(k)];
  }
  return idx[size_t(keep - 1)];
}

}  // namespace ops
