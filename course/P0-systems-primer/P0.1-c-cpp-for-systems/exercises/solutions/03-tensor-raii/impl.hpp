// Exercise 3 solution.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>

class Tensor {
 public:
  Tensor(int64_t rows, int64_t cols) : rows_(rows), cols_(cols) {
    size_t bytes = size_t(rows * cols) * sizeof(float);
    size_t rounded = ((bytes + 63) / 64) * 64;  // aligned_alloc requires a multiple of the alignment
    p_ = static_cast<float*>(std::aligned_alloc(64, rounded ? rounded : 64));
    if (!p_) throw std::bad_alloc();
    std::memset(p_, 0, rounded ? rounded : 64);
    ++live_;
  }
  ~Tensor() { release(); }
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&& o) noexcept
      : p_(std::exchange(o.p_, nullptr)), rows_(std::exchange(o.rows_, 0)), cols_(std::exchange(o.cols_, 0)) {}
  Tensor& operator=(Tensor&& o) noexcept {
    if (this != &o) {  // self-move must not free the storage we are about to "take"
      release();
      p_ = std::exchange(o.p_, nullptr);
      rows_ = std::exchange(o.rows_, 0);
      cols_ = std::exchange(o.cols_, 0);
    }
    return *this;
  }
  Tensor clone() const {
    Tensor t(rows_, cols_);
    if (numel()) std::memcpy(t.p_, p_, size_t(numel()) * sizeof(float));
    return t;  // NRVO or move — never a copy (copy is deleted)
  }
  float& operator()(int64_t i, int64_t j) { return p_[i * cols_ + j]; }
  float* data() { return p_; }
  const float* data() const { return p_; }
  int64_t rows() const { return rows_; }
  int64_t cols() const { return cols_; }
  int64_t numel() const { return rows_ * cols_; }
  static int live_count() { return live_; }

 private:
  void release() {
    if (p_) { std::free(p_); p_ = nullptr; --live_; }
    rows_ = cols_ = 0;
  }
  float* p_ = nullptr;
  int64_t rows_ = 0, cols_ = 0;
  static inline int live_ = 0;
};
