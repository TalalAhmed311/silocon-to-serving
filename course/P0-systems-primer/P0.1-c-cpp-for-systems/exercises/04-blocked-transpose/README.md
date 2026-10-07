# Exercise 4 — Generic blocked `transpose<T, Block>` (medium)

```cpp
template <class T, int Block = 32>
void transpose(const T* src, T* dst, int64_t rows, int64_t cols);  // dst is (cols, rows) row-major
```

The naive double loop writes `dst` with a stride of `rows`, which touches a new cache line on every write. Process the matrix in `Block × Block` tiles instead. Inside a tile, both the reads and the writes stay within `Block` cache lines, which all fit in L1.

**Hints**

1. Use four nested loops: two over tile origins, two inside the tile.
2. Clamp the inner loop bounds with `std::min` so shapes that aren't a multiple of `Block` work.
3. `Block` is a template parameter, so the compiler can unroll the inner loops.

**Test:** `float` and `int8_t`, with shapes including 1×1, 1×N, 33×65 and 128×96, checked against the naive reference. Note that the naive starter already *passes* the correctness test. The benchmark in `bench/` is where you prove the blocked version is faster: it times your `impl.hpp` against naive and against a plain copy.
