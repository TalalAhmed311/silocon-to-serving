// Exercise 4 starter: currently the naive version. Make it blocked.
#pragma once
#include <algorithm>
#include <cstdint>

template <class T, int Block = 32>
void transpose(const T* src, T* dst, int64_t rows, int64_t cols) {
  // TODO: tile this into Block x Block blocks.
  for (int64_t i = 0; i < rows; ++i)
    for (int64_t j = 0; j < cols; ++j) dst[j * rows + i] = src[i * cols + j];
}
