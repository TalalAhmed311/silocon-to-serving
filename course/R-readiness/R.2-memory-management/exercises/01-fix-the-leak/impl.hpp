// Exercise 1 starter. TODO: stop the leak by using std::unique_ptr instead of new[]/delete[].
#pragma once
#include <memory>

// Squares 1..n into a temporary array, then returns the sum of the squares.
// If any square is larger than `limit`, it gives up and returns -1.
//
// BUG: the early `return -1` skips the delete[] and leaks the array.
inline long sum_of_squares_below(int n, long limit) {
  long* squares = new long[n];
  for (int i = 0; i < n; ++i) {
    squares[i] = long(i + 1) * (i + 1);
    if (squares[i] > limit) {
      return -1;  // leaks `squares`
    }
  }
  long total = 0;
  for (int i = 0; i < n; ++i) total += squares[i];
  delete[] squares;
  return total;
}
