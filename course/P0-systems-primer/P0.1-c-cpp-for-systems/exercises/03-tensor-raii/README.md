# Exercise 3 — The `Tensor` RAII class (medium)

Write a move-only 2-D `Tensor` that owns 64-byte-aligned float storage. This is the shape of the tensor type you will use in the P0.5 engine.

```cpp
class Tensor {
 public:
  Tensor(int64_t rows, int64_t cols);          // zero-initialized, 64-byte aligned
  Tensor(Tensor&&) noexcept; Tensor& operator=(Tensor&&) noexcept;
  Tensor(const Tensor&) = delete; Tensor& operator=(const Tensor&) = delete;
  Tensor clone() const;                         // explicit deep copy: the only way to copy
  float& operator()(int64_t i, int64_t j);      // row-major
  float* data(); int64_t rows() const; int64_t cols() const; int64_t numel() const;
  static int live_count();                      // number of Tensors currently owning memory
};
```

**Hints**

1. Use `std::aligned_alloc(64, bytes)`. The bytes must be rounded up to a multiple of 64.
2. A moved-from tensor must own nothing, have shape `0×0`, and be safe to destroy and to assign into.
3. `live_count` increments when memory is acquired and decrements when it is freed. It should never go negative.

**Test:** checks alignment, zero-init, move construction and assignment, self-move safety, `clone` independence, and `live_count` invariants.
