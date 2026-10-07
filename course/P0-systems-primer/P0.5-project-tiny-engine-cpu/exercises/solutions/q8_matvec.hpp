// Exercise 5 solution.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

struct Q8Matrix {
  int rows = 0, cols = 0;
  std::vector<float> scales;
  std::vector<int8_t> q;
};

inline void quantize_block(const float* x, float& scale, int8_t* q) {
  float amax = 0;
  for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
  scale = amax / 127.0f;
  const float inv = scale > 0 ? 1.0f / scale : 0.0f;
  for (int i = 0; i < 32; ++i) q[i] = int8_t(std::clamp<long>(std::lround(x[i] * inv), -127, 127));
}

inline Q8Matrix quantize_matrix(const float* W, int rows, int cols) {
  Q8Matrix m;
  m.rows = rows; m.cols = cols;
  m.scales.resize(size_t(rows) * cols / 32);
  m.q.resize(size_t(rows) * cols);
  for (size_t b = 0; b < m.scales.size(); ++b) quantize_block(W + 32 * b, m.scales[b], m.q.data() + 32 * b);
  return m;
}

inline void matvec_q8(const Q8Matrix& W, const float* x, float* y) {
  const int nb = W.cols / 32;
  std::vector<float> xs(size_t(nb));
  std::vector<int8_t> xq(size_t(W.cols));
  for (int b = 0; b < nb; ++b) quantize_block(x + 32 * b, xs[size_t(b)], xq.data() + 32 * b);  // once per call
  for (int r = 0; r < W.rows; ++r) {
    float acc = 0;
    for (int b = 0; b < nb; ++b) {
      const int8_t* wq = W.q.data() + (size_t(r) * nb + b) * 32;
      const int8_t* aq = xq.data() + size_t(b) * 32;
      int32_t s = 0;
      for (int i = 0; i < 32; ++i) s += int32_t(wq[i]) * aq[i];  // the compiler vectorizes this; P0.4 ex 5 has SIMD
      acc += W.scales[size_t(r) * nb + b] * xs[size_t(b)] * float(s);
    }
    y[r] = acc;
  }
}
