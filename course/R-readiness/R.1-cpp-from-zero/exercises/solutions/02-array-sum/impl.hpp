#pragma once
#include <cstddef>
#include <limits>

inline float array_sum(const float* a, size_t n) {
  float total = 0.0f;
  for (const float* p = a; p != a + n; ++p) total += *p;
  return total;
}

inline float array_max(const float* a, size_t n) {
  float best = std::numeric_limits<float>::lowest();
  for (const float* p = a; p != a + n; ++p)
    if (*p > best) best = *p;
  return best;
}
