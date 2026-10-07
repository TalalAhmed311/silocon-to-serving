#pragma once
#include <d4/norms.cuh>
inline void rmsnorm(const float* x, const float* w, float* y, int rows, int cols, float eps) {
  d4::rmsnorm(x, w, y, rows, cols, eps);
}
