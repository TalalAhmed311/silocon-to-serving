// Exercise 1 solution.
#pragma once
#include <cstddef>
#include "simd.hpp"

inline void saxpy(float a, const float* x, float* y, size_t n) {
  const simd::vf va = simd::set1(a);
  size_t i = 0;
  for (; i + simd::W <= n; i += simd::W) simd::store(y + i, simd::fma(va, simd::load(x + i), simd::load(y + i)));
  for (; i < n; ++i) y[i] = a * x[i] + y[i];
}
