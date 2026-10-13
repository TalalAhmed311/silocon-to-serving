# Exercise 3: An owning `FloatBuffer` (medium)

You write a tiny version of what `std::vector<float>` or a tensor does: own a heap block of floats and release it exactly once.

`impl.hpp` has the class with the fields and the signatures in place. The bodies of the constructor, the destructor and the two move functions are stubs.

**Task**

- **Constructor** `FloatBuffer(size_t n)`: if `n > 0`, allocate `new float[n]()` (the `()` zeroes it), remember the size, and add 1 to `live_`.
- **Destructor**: if the buffer owns memory, `delete[]` it and subtract 1 from `live_`.
- **Copy**: already deleted. Two owners of one block would free it twice.
- **Move constructor**: take the other buffer's pointer and size, then set the other to `nullptr` and `0`. Keep it `noexcept`.
- **Move assignment**: free what you already own, then take the other's pointer and size and empty the other. Guard against `a = std::move(a)`.

**Hints**

1. `std::exchange(x, new_value)` returns the old value of `x` and stores `new_value`. It is a neat way to "take and leave empty" in one expression.
2. Write one private helper `release()` and call it from the destructor and the move assignment.
3. `live_` counts blocks, not floats. The test uses it to prove nothing leaks and nothing is freed twice.
4. Why `noexcept`? `std::vector` only moves its elements when it grows if the move cannot throw. Otherwise it would copy, and copying is deleted here. The test has a vector-growth case.

**Test:** `ctest -R 03-owning-buffer --output-on-failure`. It runs under AddressSanitizer, so a double free or a leak is reported with a stack trace.
