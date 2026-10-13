// Exercise 2 starter. TODO: implement both functions with pointer arithmetic.
// Rule: do not write a[i]. Walk a pointer instead.
#pragma once
#include <cstddef>
#include <limits>

// Sum of a[0] .. a[n-1]. An empty array sums to 0.
inline float array_sum(const float* a, size_t n) {
  // TODO: const float* end = a + n; then walk p from a to end
  (void)a; (void)n;
  return 0.0f;
}

// Largest element. For n == 0 return std::numeric_limits<float>::lowest().
inline float array_max(const float* a, size_t n) {
  // TODO
  (void)a; (void)n;
  return 0.0f;
}
