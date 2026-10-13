#pragma once
#include <cstddef>

// ASan's first line (pasted):
//   ==4242==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x... at pc 0x... bp 0x... sp 0x...
// (next line: WRITE of size 4 ... "0 bytes after 16-byte region" for n == 4)
// Cause: the loop ran with i <= n, so the last iteration wrote p[n], one element past the end.

inline int* make_ramp(size_t n) {
  int* p = new int[n];
  for (size_t i = 0; i < n; ++i) {
    p[i] = int(10 * i);
  }
  return p;
}
