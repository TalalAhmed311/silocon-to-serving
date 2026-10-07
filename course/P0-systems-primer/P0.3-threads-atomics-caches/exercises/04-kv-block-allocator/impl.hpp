// Exercise 4 starter (D2). Implement every method; keep it thread-safe.
#pragma once
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

class BlockAllocator {
 public:
  explicit BlockAllocator(int num_blocks) : refcnt_(size_t(num_blocks), 0) {
    // TODO: fill free_ so that block 0 is handed out first
  }
  std::optional<int> allocate() { return std::nullopt; /* TODO */ }
  std::vector<int> allocate_many(int n) { (void)n; return {}; /* TODO */ }
  void share(int block) { (void)block; /* TODO */ }
  void free(int block) { (void)block; /* TODO */ }
  int copy_on_write(int block) { return block; /* TODO */ }
  int refcount(int block) const { std::lock_guard g(mu_); return refcnt_.at(size_t(block)); }
  int num_free() const { std::lock_guard g(mu_); return int(free_.size()); }
  int num_blocks() const { return int(refcnt_.size()); }

 private:
  mutable std::mutex mu_;
  std::vector<int> refcnt_;
  std::vector<int> free_;  // a stack: back() is the next block handed out
};
