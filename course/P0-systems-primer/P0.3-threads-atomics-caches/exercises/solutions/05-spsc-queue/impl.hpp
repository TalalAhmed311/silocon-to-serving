// Exercise 5 solution: classic SPSC ring with acquire/release on the indices.
#pragma once
#include <atomic>
#include <cstddef>

template <class T, size_t N>
class SpscQueue {
  static_assert((N & (N - 1)) == 0, "N must be a power of two");

 public:
  bool push(const T& v) {
    const size_t t = tail_.load(std::memory_order_relaxed);   // only we write tail_
    if (t - head_.load(std::memory_order_acquire) == N) return false;  // full; acquire pairs with pop's release
    buf_[t & (N - 1)] = v;
    tail_.store(t + 1, std::memory_order_release);             // publish the slot to the consumer
    return true;
  }
  bool pop(T& out) {
    const size_t h = head_.load(std::memory_order_relaxed);   // only we write head_
    if (h == tail_.load(std::memory_order_acquire)) return false;      // empty; acquire sees the slot write
    out = buf_[h & (N - 1)];
    head_.store(h + 1, std::memory_order_release);             // tell the producer the slot is reusable
    return true;
  }

 private:
  T buf_[N];
  alignas(64) std::atomic<size_t> head_{0};  // separate lines: producer and consumer don't false-share
  alignas(64) std::atomic<size_t> tail_{0};
};
