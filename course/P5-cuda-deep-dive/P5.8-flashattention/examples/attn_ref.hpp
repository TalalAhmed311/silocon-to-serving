// attn_ref.hpp — shared CPU reference + helpers for the P5.8 tests (double math from fp16-rounded inputs).
#pragma once
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace attnref {
constexpr int D = 64;
inline std::vector<__half> to_h(const std::vector<float>& v) { std::vector<__half> o(v.size()); for (size_t i = 0; i < v.size(); ++i) o[i] = __float2half(v[i]); return o; }
inline std::vector<float> to_f(const std::vector<__half>& v) { std::vector<float> o(v.size()); for (size_t i = 0; i < v.size(); ++i) o[i] = __half2float(v[i]); return o; }

// q, k, v: [BH, N, D] as floats (already fp16-rounded). Returns O [BH, N, D].
inline std::vector<float> attention(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, int BH, int N, bool causal) {
  std::vector<float> o(size_t(BH) * N * D);
  const double sc = 1.0 / std::sqrt(double(D));
  std::vector<double> s(size_t(N));
  for (int bh = 0; bh < BH; ++bh)
    for (int i = 0; i < N; ++i) {
      double m = -1e300, l = 0;
      const int jmax = causal ? i : N - 1;
      for (int j = 0; j <= jmax; ++j) {
        double a = 0;
        for (int d = 0; d < D; ++d) a += double(q[(size_t(bh) * N + i) * D + d]) * k[(size_t(bh) * N + j) * D + d];
        s[size_t(j)] = a * sc;
        m = std::max(m, s[size_t(j)]);
      }
      for (int j = 0; j <= jmax; ++j) { s[size_t(j)] = std::exp(s[size_t(j)] - m); l += s[size_t(j)]; }
      for (int d = 0; d < D; ++d) {
        double a = 0;
        for (int j = 0; j <= jmax; ++j) a += s[size_t(j)] * v[(size_t(bh) * N + j) * D + d];
        o[(size_t(bh) * N + i) * D + d] = float(a / l);
      }
    }
  return o;
}
}  // namespace attnref
