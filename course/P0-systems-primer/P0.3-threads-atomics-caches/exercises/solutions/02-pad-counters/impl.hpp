// Exercise 2 solution: alignas(64) rounds both alignment and size up to one cache line.
#pragma once
#include <atomic>
#include <cstdint>

struct alignas(64) Slot {
  std::atomic<uint64_t> value{0};
};

template <int N>
struct PerWorkerCounters {
  Slot slots[N];
  void inc(int worker) { slots[worker].value.fetch_add(1, std::memory_order_relaxed); }
  uint64_t total() const { uint64_t s = 0; for (auto& x : slots) s += x.value.load(); return s; }
};
