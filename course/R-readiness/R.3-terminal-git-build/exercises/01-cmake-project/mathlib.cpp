#include "mathlib.hpp"

int add(int a, int b) { return a + b; }

long long fib(int n) {
  long long prev = 0, cur = 1;
  for (int i = 0; i < n; ++i) {
    long long next = prev + cur;
    prev = cur;
    cur = next;
  }
  return prev;
}

double mean(const double* v, std::size_t n) {
  if (n == 0) return 0.0;
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) sum += v[i];
  return sum / static_cast<double>(n);
}
