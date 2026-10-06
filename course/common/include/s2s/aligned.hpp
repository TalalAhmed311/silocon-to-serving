// s2s/aligned.hpp — a 64-byte-aligned, move-only buffer.
// 64 bytes = one cache line and one AVX-512 register, so SIMD loads never split a line.
#pragma once
#include <cstddef>
#include <cstdlib>
#include <new>
#include <utility>

namespace s2s {

template <class T>
class aligned_buffer {
 public:
  aligned_buffer() = default;
  explicit aligned_buffer(size_t n) : n_(n) {
    // aligned_alloc requires size to be a multiple of the alignment.
    size_t bytes = ((n * sizeof(T) + 63) / 64) * 64;
    p_ = static_cast<T*>(std::aligned_alloc(64, bytes ? bytes : 64));
    if (!p_) throw std::bad_alloc();
  }
  ~aligned_buffer() { std::free(p_); }
  aligned_buffer(const aligned_buffer&) = delete;
  aligned_buffer& operator=(const aligned_buffer&) = delete;
  aligned_buffer(aligned_buffer&& o) noexcept : p_(std::exchange(o.p_, nullptr)), n_(std::exchange(o.n_, 0)) {}
  aligned_buffer& operator=(aligned_buffer&& o) noexcept {
    if (this != &o) { std::free(p_); p_ = std::exchange(o.p_, nullptr); n_ = std::exchange(o.n_, 0); }
    return *this;
  }
  T* data() { return p_; }
  const T* data() const { return p_; }
  size_t size() const { return n_; }
  T& operator[](size_t i) { return p_[i]; }
  const T& operator[](size_t i) const { return p_[i]; }

 private:
  T* p_ = nullptr;
  size_t n_ = 0;
};

}  // namespace s2s
