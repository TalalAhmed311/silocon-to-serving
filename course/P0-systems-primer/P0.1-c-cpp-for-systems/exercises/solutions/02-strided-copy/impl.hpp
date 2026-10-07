// Exercise 2 solution. Writes are contiguous (inner loop over j in dst); reads follow the strides.
#pragma once
#include <cstdint>

inline void strided_copy(const float* src, int64_t rows, int64_t cols, int64_t stride0, int64_t stride1,
                         float* dst) {
  for (int64_t i = 0; i < rows; ++i) {
    const float* row = src + i * stride0;  // hoist the row base: one multiply per row, not per element
    float* out = dst + i * cols;
    for (int64_t j = 0; j < cols; ++j) out[j] = row[j * stride1];
  }
}
