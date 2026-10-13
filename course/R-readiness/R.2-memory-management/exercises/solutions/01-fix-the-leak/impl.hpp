#pragma once
#include <memory>

// The unique_ptr<long[]> owns the array. Its destructor runs on every way out of the function.
inline long sum_of_squares_below(int n, long limit) {
  auto squares = std::make_unique<long[]>(n);
  for (int i = 0; i < n; ++i) {
    squares[i] = long(i + 1) * (i + 1);
    if (squares[i] > limit) {
      return -1;  // no leak: ~unique_ptr frees the array here
    }
  }
  long total = 0;
  for (int i = 0; i < n; ++i) total += squares[i];
  return total;
}
