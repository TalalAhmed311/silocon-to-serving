// Exercise 5 solution: the three fixes are marked FIX.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>

class TokenRing {
 public:
  explicit TokenRing(int capacity) : cap_(capacity), buf_(static_cast<int32_t*>(std::malloc(sizeof(int32_t) * capacity))) {}
  ~TokenRing() { std::free(buf_); }
  TokenRing(const TokenRing&) = delete;
  TokenRing& operator=(const TokenRing&) = delete;

  int32_t* push(int32_t tok) {
    if (size_ == cap_) grow();
    int32_t* slot = buf_ + size_;      // FIX: take the pointer *after* a possible reallocation
    *slot = tok;
    ++size_;
    return slot;
  }

  void clear() {
    for (int i = 0; i < cap_; ++i) buf_[i] = 0;    // FIX: < not <=
    size_ = 0;
  }

  uint64_t fingerprint(int n) const {
    uint64_t h = 0;
    // FIX: original shifted by 1 << (i%6+1) ∈ {2,4,...,64}; 64 is UB for a 64-bit value.
    // Shift by (i % 6 + 1) * 8 ∈ {8..48} instead — always < 64.
    for (int i = 0; i < n && i < size_; ++i) h ^= uint64_t(uint32_t(buf_[i])) << ((i % 6 + 1) * 8);
    return h;
  }

  int size() const { return size_; }
  int capacity() const { return cap_; }
  int32_t at(int i) const { return buf_[i]; }

 private:
  void grow() {
    int new_cap = cap_ * 2;
    auto* nb = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * new_cap));
    std::memcpy(nb, buf_, sizeof(int32_t) * size_);
    std::free(buf_);
    buf_ = nb;
    cap_ = new_cap;
  }
  int cap_;
  int size_ = 0;
  int32_t* buf_;
};
