// attn_ref.hpp — double-precision CPU reference for the L6 tests, mirroring attn::Params (same masks and biases).
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace attnref {

struct Cfg {
  int B = 1, H = 1, Hkv = 1, Lq = 0, Lk = 0, d = 64;
  bool causal = false;
  int window = 0;
  std::vector<float> alibi, sinks;
};

// q: [B, H, Lq, d], k/v: [B, Hkv, Lk, d] contiguous. Returns o: [B, H, Lq, d].
inline std::vector<float> run(const Cfg& c, const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v) {
  std::vector<float> o(size_t(c.B) * c.H * c.Lq * c.d);
  const double sc = 1.0 / std::sqrt(double(c.d));
  const int shift = c.Lk - c.Lq;
  for (int b = 0; b < c.B; ++b)
    for (int h = 0; h < c.H; ++h) {
      const int kh = h / (c.H / c.Hkv);
      const size_t qo = (size_t(b) * c.H + h) * c.Lq * c.d, ko = (size_t(b) * c.Hkv + kh) * c.Lk * c.d;
      for (int i = 0; i < c.Lq; ++i) {
        const int pos = i + shift;
        std::vector<double> s(size_t(c.Lk), -1e300);
        double m = c.sinks.empty() ? -1e300 : c.sinks[size_t(h)];
        for (int j = 0; j < c.Lk; ++j) {
          if (c.causal && j > pos) continue;
          if (c.window > 0 && pos - j >= c.window) continue;
          double a = 0;
          for (int t = 0; t < c.d; ++t) a += double(q[qo + size_t(i) * c.d + t]) * k[ko + size_t(j) * c.d + t];
          a *= sc;
          if (!c.alibi.empty()) a -= c.alibi[size_t(h)] * double(pos - j);
          s[size_t(j)] = a;
          m = std::max(m, a);
        }
        double l = c.sinks.empty() ? 0.0 : std::exp(c.sinks[size_t(h)] - m);
        std::vector<double> acc(size_t(c.d), 0.0);
        for (int j = 0; j < c.Lk; ++j) {
          if (s[size_t(j)] <= -1e299) continue;
          const double pj = std::exp(s[size_t(j)] - m);
          l += pj;
          for (int t = 0; t < c.d; ++t) acc[size_t(t)] += pj * v[ko + size_t(j) * c.d + t];
        }
        for (int t = 0; t < c.d; ++t) o[qo + size_t(i) * c.d + t] = l > 0 ? float(acc[size_t(t)] / l) : 0.f;
      }
    }
  return o;
}

}  // namespace attnref
