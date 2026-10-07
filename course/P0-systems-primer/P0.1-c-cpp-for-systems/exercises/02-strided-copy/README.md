# Exercise 2 — `strided_copy`, i.e. `.contiguous()` for 2-D (easy)

Implement

```cpp
void strided_copy(const float* src, int64_t rows, int64_t cols,
                  int64_t stride0, int64_t stride1, float* dst);
```

It writes the logical `(rows, cols)` view of `src` (element `(i,j)` at `src[i*stride0 + j*stride1]`) into `dst` as a **contiguous row-major** array.

**Hints**

1. Loop over the *destination* in row-major order, so the writes are contiguous.
2. The source index is just the stride formula.
3. Negative strides are legal: a flipped view. Use signed `int64_t` arithmetic.

**Test:** compared against a naive reference for row-major, column-major, transposed, sliced (every other column) and flipped views.
