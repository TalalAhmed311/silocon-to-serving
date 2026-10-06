// Exercise 5 starter.
#pragma once
#include <atomic>
#include <cstddef>

template <class T, size_t N>
class SpscQueue {
  static_assert((N & (N - 1)) == 0, "N must be a power of two");

 public:
  bool push(const T& v) { (void)v; return false; /* TODO */ }
  bool pop(T& out) { (void)out; return false; /* TODO */ }

 private:
  T buf_[N];
  alignas(64) std::atomic<size_t> head_{0};
  alignas(64) std::atomic<size_t> tail_{0};
};
