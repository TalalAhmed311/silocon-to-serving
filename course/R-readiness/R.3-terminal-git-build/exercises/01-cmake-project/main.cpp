#include <cstdio>
#include "mathlib.hpp"

int main() {
  const double xs[] = {1.0, 2.0, 3.0, 4.0};
  std::printf("add(2, 3)  = %d\n", add(2, 3));
  std::printf("fib(10)    = %lld\n", fib(10));
  std::printf("mean(xs)   = %.2f\n", mean(xs, 4));
  return 0;
}
