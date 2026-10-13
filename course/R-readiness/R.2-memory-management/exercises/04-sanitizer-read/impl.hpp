// Exercise 4 starter. There is ONE planted memory bug in this file.
// TODO 1: run the test and copy the first line of AddressSanitizer's report into the comment below.
// TODO 2: fix the bug.
#pragma once
#include <cstddef>

// ASan's first line (paste it here):
//   ==PID==ERROR: AddressSanitizer: ...

// Returns a new array p of length n where p[i] = 10 * i.
// The caller frees it with delete[].
inline int* make_ramp(size_t n) {
  int* p = new int[n];
  for (size_t i = 0; i <= n; ++i) {
    p[i] = int(10 * i);
  }
  return p;
}
