// Exercise 1 starter: RACY. Fix record() without changing the interface.
#pragma once
#include <cstdint>

struct TokenStats {
  uint64_t total = 0;
  uint64_t max_batch = 0;
  void record(uint64_t batch_tokens) {
    total += batch_tokens;                         // data race
    if (batch_tokens > max_batch) max_batch = batch_tokens;  // data race (check-then-act)
  }
  uint64_t get_total() const { return total; }
  uint64_t get_max() const { return max_batch; }
};
