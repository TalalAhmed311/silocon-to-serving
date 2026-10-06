// Exercise 4 solution: cache-blocked transpose.
// Inside one tile we read Block rows of src and write Block rows of dst; 2*Block lines of 64 B fit in L1
// for Block=32 floats (2 * 32 * 128 B = 8 KB), so every fetched line is fully used before eviction.
#pragma once
#include <algorithm>
#include <cstdint>

template <class T, int Block = 32>
void transpose(const T* src, T* dst, int64_t rows, int64_t cols) {
  for (int64_t ii = 0; ii < rows; ii += Block) {
    const int64_t i_end = std::min<int64_t>(ii + Block, rows);
    for (int64_t jj = 0; jj < cols; jj += Block) {
      const int64_t j_end = std::min<int64_t>(jj + Block, cols);
      for (int64_t i = ii; i < i_end; ++i)
        for (int64_t j = jj; j < j_end; ++j) dst[j * rows + i] = src[i * cols + j];
    }
  }
}
