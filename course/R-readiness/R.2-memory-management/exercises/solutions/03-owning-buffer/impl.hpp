#pragma once
#include <cstddef>
#include <utility>

class FloatBuffer {
 public:
  static int live_allocations() { return live_; }

  explicit FloatBuffer(size_t n) : data_(nullptr), size_(0) {
    if (n > 0) {
      data_ = new float[n]();
      size_ = n;
      ++live_;
    }
  }

  ~FloatBuffer() { release(); }

  FloatBuffer(const FloatBuffer&) = delete;
  FloatBuffer& operator=(const FloatBuffer&) = delete;

  FloatBuffer(FloatBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

  FloatBuffer& operator=(FloatBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  float* data() { return data_; }
  const float* data() const { return data_; }
  size_t size() const { return size_; }
  float& operator[](size_t i) { return data_[i]; }
  const float& operator[](size_t i) const { return data_[i]; }

 private:
  void release() {
    if (data_) {
      delete[] data_;
      data_ = nullptr;
      size_ = 0;
      --live_;
    }
  }

  float* data_;
  size_t size_;
  inline static int live_ = 0;
};
