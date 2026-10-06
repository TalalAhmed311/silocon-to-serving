// Exercise 1 solution: atomics. fetch_add for the sum; a CAS loop for the max.
#pragma once
#include <atomic>
#include <cstdint>

struct TokenStats {
  std::atomic<uint64_t> total{0};
  std::atomic<uint64_t> max_batch{0};
  void record(uint64_t batch_tokens) {
    total.fetch_add(batch_tokens, std::memory_order_relaxed);  // stats only: no data published
    uint64_t cur = max_batch.load(std::memory_order_relaxed);
    // On failure compare_exchange_weak reloads `cur`, so the loop retries with the latest value
    // and stops as soon as someone else already stored something >= ours.
    while (batch_tokens > cur && !max_batch.compare_exchange_weak(cur, batch_tokens, std::memory_order_relaxed)) {}
  }
  uint64_t get_total() const { return total.load(); }
  uint64_t get_max() const { return max_batch.load(); }
};
