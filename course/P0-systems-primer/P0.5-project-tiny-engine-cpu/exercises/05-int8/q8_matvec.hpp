// Exercise 5 starter: int8 weight-only + int8 activation matvec with per-32-block scales.
#pragma once
#include <cstdint>
#include <vector>

struct Q8Matrix {
  int rows = 0, cols = 0;           // cols % 32 == 0
  std::vector<float> scales;        // rows * cols / 32
  std::vector<int8_t> q;            // rows * cols
};

// Quantize a row-major fp32 matrix: per row, per block of 32: scale = max|w| / 127, q = round(w / scale).
inline Q8Matrix quantize_matrix(const float* W, int rows, int cols) {
  (void)W;
  Q8Matrix m; m.rows = rows; m.cols = cols;
  m.scales.assign(size_t(rows) * cols / 32, 0.0f);
  m.q.assign(size_t(rows) * cols, 0);
  return m;  // TODO
}

// y = W x. Quantize x to q8 blocks once, then per row: Σ_blocks sW * sx * (Σ int8 * int8 in int32).
inline void matvec_q8(const Q8Matrix& W, const float* x, float* y) {
  (void)x;
  for (int r = 0; r < W.rows; ++r) y[r] = 0;  // TODO
}
