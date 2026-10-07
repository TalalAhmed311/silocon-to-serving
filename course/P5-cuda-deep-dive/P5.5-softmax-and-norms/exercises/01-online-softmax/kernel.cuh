// Exercise 1 starter: row-wise softmax, rows × cols fp32. Target: a single pass over the input that computes the max
// and the normalizer together (online softmax), then one write — and report GB/s (the L4 exit check).
#pragma once
#include <cuda_runtime.h>

inline void softmax_rows(const float* x, float* y, int rows, int cols) {
  (void)x; (void)y; (void)rows; (void)cols;   // TODO
}
