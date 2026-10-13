#include "stats.hpp"

double mean(const double* a, size_t n) {
  if (n == 0) return 0.0;
  double total = 0.0;
  for (size_t i = 0; i < n; ++i) total += a[i];
  return total / double(n);
}

double variance(const double* a, size_t n) {
  if (n == 0) return 0.0;
  const double m = mean(a, n);
  double total = 0.0;
  for (size_t i = 0; i < n; ++i) total += square(a[i] - m);
  return total / double(n);
}

double range(const double* a, size_t n) {
  if (n == 0) return 0.0;
  double lo = a[0], hi = a[0];
  for (size_t i = 1; i < n; ++i) {
    if (a[i] < lo) lo = a[i];
    if (a[i] > hi) hi = a[i];
  }
  return hi - lo;
}
