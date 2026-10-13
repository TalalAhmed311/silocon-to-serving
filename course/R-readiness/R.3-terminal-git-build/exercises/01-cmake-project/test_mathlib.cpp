// A tiny test: returns 0 when every check passes and 1 otherwise. CTest runs it and reads that exit code.
#include <cstdio>
#include "mathlib.hpp"

static int failures = 0;
#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::printf("FAILED line %d: %s\n", __LINE__, #cond);             \
      ++failures;                                                       \
    }                                                                   \
  } while (0)

int main() {
  CHECK(add(2, 3) == 5);
  CHECK(add(-4, 4) == 0);
  CHECK(fib(0) == 0);
  CHECK(fib(1) == 1);
  CHECK(fib(10) == 55);
  CHECK(fib(50) == 12586269025LL);
  const double xs[] = {1.0, 2.0, 3.0, 4.0};
  CHECK(mean(xs, 4) == 2.5);
  CHECK(mean(xs, 0) == 0.0);
  if (failures == 0) std::printf("all checks passed\n");
  return failures == 0 ? 0 : 1;
}
