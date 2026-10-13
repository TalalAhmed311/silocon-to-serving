// Exercise 2 starter. TODO: change the return type so the squares are returned BY VALUE.
#pragma once

// Returns the first n squares (1, 4, 9, ...). n is at most 16.
//
// BUG: `out` is a local array. It lives in this function's stack frame, which is gone when the
// function returns. The caller gets a dangling pointer.
//
// Fix: return std::vector<int> (include <vector>) and build it inside the function.
inline int* first_squares(int n) {
  int out[16];
  for (int i = 0; i < n; ++i) out[i] = (i + 1) * (i + 1);
  return out;  // pointer to a local: the compiler warns about this
}
