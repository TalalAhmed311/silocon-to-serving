// Exercise 4 solution (D2): free-list stack + ref counts, one mutex.
#pragma once
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

class BlockAllocator {
 public:
  explicit BlockAllocator(int num_blocks) : refcnt_(size_t(num_blocks), 0) {
    free_.reserve(size_t(num_blocks));
    for (int b = num_blocks - 1; b >= 0; --b) free_.push_back(b);  // block 0 on top: handed out first
  }

  std::optional<int> allocate() {
    std::lock_guard g(mu_);
    return allocate_locked();
  }

  std::vector<int> allocate_many(int n) {
    std::lock_guard g(mu_);
    if (n < 0 || size_t(n) > free_.size()) return {};  // all-or-nothing: decide before taking anything
    std::vector<int> out;
    out.reserve(size_t(n));
    for (int i = 0; i < n; ++i) out.push_back(*allocate_locked());
    return out;
  }

  void share(int block) {
    std::lock_guard g(mu_);
    check_live(block, "share");
    ++refcnt_[size_t(block)];
  }

  void free(int block) {
    std::lock_guard g(mu_);
    free_locked(block);
  }

  int copy_on_write(int block) {
    std::lock_guard g(mu_);
    check_live(block, "copy_on_write");
    if (refcnt_[size_t(block)] == 1) return block;  // sole owner: write in place
    auto fresh = allocate_locked();
    if (!fresh) throw std::runtime_error("copy_on_write: out of blocks");  // caller must preempt/evict
    free_locked(block);  // drop our reference to the shared block (refcount stays >= 1 for the others)
    return *fresh;
  }

  int refcount(int block) const { std::lock_guard g(mu_); return refcnt_.at(size_t(block)); }
  int num_free() const { std::lock_guard g(mu_); return int(free_.size()); }
  int num_blocks() const { return int(refcnt_.size()); }

 private:
  std::optional<int> allocate_locked() {
    if (free_.empty()) return std::nullopt;
    int b = free_.back();
    free_.pop_back();
    refcnt_[size_t(b)] = 1;
    return b;
  }
  void free_locked(int block) {
    check_live(block, "free");
    if (--refcnt_[size_t(block)] == 0) free_.push_back(block);  // LIFO: hot blocks are reused first
  }
  void check_live(int block, const char* op) const {
    if (block < 0 || size_t(block) >= refcnt_.size()) throw std::logic_error(std::string(op) + ": bad block id");
    if (refcnt_[size_t(block)] == 0) throw std::logic_error(std::string(op) + ": block is free (double free / use after free)");
  }
  mutable std::mutex mu_;
  std::vector<int> refcnt_;
  std::vector<int> free_;
};
