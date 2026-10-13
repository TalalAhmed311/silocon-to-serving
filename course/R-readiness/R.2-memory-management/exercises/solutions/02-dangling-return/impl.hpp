#pragma once
#include <vector>

// Returns the squares by value. The vector owns its heap buffer and moves out of the function
// without copying the elements.
inline std::vector<int> first_squares(int n) {
  std::vector<int> out;
  out.reserve(n);
  for (int i = 0; i < n; ++i) out.push_back((i + 1) * (i + 1));
  return out;
}
