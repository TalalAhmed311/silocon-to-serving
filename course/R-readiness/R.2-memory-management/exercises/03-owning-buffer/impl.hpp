// Exercise 3 starter. TODO: make FloatBuffer own its memory correctly.
//   constructor: allocate n floats (zeroed) and count the allocation
//   destructor:  free them and uncount
//   copy:        deleted (already written for you)
//   move:        take the other buffer's pointer and leave it empty
#pragma once
#include <cstddef>
#include <utility>

class FloatBuffer {
 public:
  // Number of FloatBuffers that currently own an allocation. The test reads this.
  static int live_allocations() { return live_; }

  explicit FloatBuffer(size_t n) : data_(nullptr), size_(0) {
    // TODO: if n > 0: data_ = new float[n]() ; size_ = n ; ++live_;
    (void)n;
  }

  ~FloatBuffer() {
    // TODO: if data_ is set: delete[] data_ ; --live_
  }

  FloatBuffer(const FloatBuffer&) = delete;
  FloatBuffer& operator=(const FloatBuffer&) = delete;

  FloatBuffer(FloatBuffer&& other) noexcept : data_(nullptr), size_(0) {
    // TODO: steal other's data_ and size_, then leave other empty (nullptr, 0).
    (void)other;
  }

  FloatBuffer& operator=(FloatBuffer&& other) noexcept {
    // TODO: if this != &other: free what we own, steal from other, leave other empty.
    (void)other;
    return *this;
  }

  float* data() { return data_; }
  const float* data() const { return data_; }
  size_t size() const { return size_; }
  float& operator[](size_t i) { return data_[i]; }
  const float& operator[](size_t i) const { return data_[i]; }

 private:
  float* data_;
  size_t size_;
  inline static int live_ = 0;
};
