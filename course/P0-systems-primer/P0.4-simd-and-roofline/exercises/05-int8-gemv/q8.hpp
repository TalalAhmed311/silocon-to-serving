// Exercise 5 starter.
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

struct BlockQ8 { float scale; int8_t q[32]; };

inline std::vector<BlockQ8> quantize(const float* x, int n) {
  (void)x;
  return std::vector<BlockQ8>(size_t(n / 32));  // TODO
}

inline void gemv_q8(const std::vector<BlockQ8>& W, int rows, int cols, const std::vector<BlockQ8>& x, float* y) {
  (void)W; (void)cols; (void)x;
  for (int r = 0; r < rows; ++r) y[r] = 0;  // TODO
}
