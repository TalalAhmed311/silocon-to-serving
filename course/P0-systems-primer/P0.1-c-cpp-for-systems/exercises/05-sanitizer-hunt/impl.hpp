// Exercise 5 starter: a token ring buffer with THREE planted bugs. Do not change the interface.
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

  // Appends a token, growing (doubling) the buffer when full. Returns a pointer to the stored slot.
  int32_t* push(int32_t tok) {
    int32_t* slot = buf_ + size_;
    if (size_ == cap_) grow();
    *slot = tok;                       // BUG (use-after-free): `slot` points into the old buffer after grow()
    ++size_;
    return buf_ + size_ - 1;
  }

  // Zeroes the buffer including one past the end (the "sentinel").
  void clear() {
    for (int i = 0; i <= cap_; ++i) buf_[i] = 0;   // BUG (overflow): writes buf_[cap_]
    size_ = 0;
  }

  // A cheap fingerprint of the first `n` tokens, used to detect duplicate streams.
  uint64_t fingerprint(int n) const {
    uint64_t h = 0;
    for (int i = 0; i < n && i < size_; ++i) h ^= (uint64_t)(uint32_t)buf_[i] << (1 << (i % 6 + 1)); // BUG (UB): shift by up to 64
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
