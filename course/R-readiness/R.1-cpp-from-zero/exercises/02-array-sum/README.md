# Exercise 2: Sum and max with pointer arithmetic (easy)

`impl.hpp` declares `array_sum` and `array_max` over a raw array: a pointer to the first element and a count `n`. This is how C code (and a CUDA kernel) receives an array.

**Task:** implement both **without indexing**. Do not write `a[i]`. Instead keep a pointer `p`, read the element with `*p`, and move it with `++p`. Stop when `p` reaches `a + n`, the address one past the last element.

**Hints**

1. `a + n` is the address `n * sizeof(float)` bytes after `a`. It is legal to compute and compare, but never to read.
2. When `n == 0` the loop must run zero times. Then `array_sum` returns `0` and `array_max` returns `std::numeric_limits<float>::lowest()`. The test passes `nullptr` here, so do not touch `*a` first.
3. Start `best` at `lowest()`, not at `0`. A max over negative numbers must come out negative.

**Test:** `ctest -R 02-array-sum --output-on-failure`.
