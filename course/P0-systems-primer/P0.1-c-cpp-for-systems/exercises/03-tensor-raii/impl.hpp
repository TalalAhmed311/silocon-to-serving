// Exercise 3 starter. Fill in every TODO; the test will tell you what is missing.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

class Tensor {
 public:
  Tensor(int64_t rows, int64_t cols) : rows_(rows), cols_(cols) {
    // TODO: allocate rows*cols floats, 64-byte aligned, zero them, ++live_
  }
  ~Tensor() {
    // TODO: free the memory if owned, --live_
  }
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&& o) noexcept {
    // TODO: take o's storage and shape; leave o empty (nullptr, 0x0)
    (void)o;
  }
  Tensor& operator=(Tensor&& o) noexcept {
    // TODO: free our storage (if any), then take o's; handle self-move
    (void)o;
    return *this;
  }
  Tensor clone() const {
    // TODO: new tensor with the same shape and a copy of the data
    return Tensor(rows_, cols_);
  }
  float& operator()(int64_t i, int64_t j) { return p_[i * cols_ + j]; }
  float* data() { return p_; }
  const float* data() const { return p_; }
  int64_t rows() const { return rows_; }
  int64_t cols() const { return cols_; }
  int64_t numel() const { return rows_ * cols_; }
  static int live_count() { return live_; }

 private:
  float* p_ = nullptr;
  int64_t rows_ = 0, cols_ = 0;
  static inline int live_ = 0;
};
