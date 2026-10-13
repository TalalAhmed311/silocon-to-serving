# Exercise 1: Swap two ints (easy)

`impl.hpp` has two empty functions. Both must exchange the values of two `int`s. They differ only in how the caller hands the ints over.

- `swap_by_pointer(int* a, int* b)`: the caller writes `swap_by_pointer(&x, &y)`. You read and write through the pointers with `*a` and `*b`.
- `swap_by_ref(int& a, int& b)`: the caller writes `swap_by_ref(x, y)`. `a` and `b` are aliases, so you use them like plain ints.

**Hints**

1. You need one temporary `int`. Without it, the first assignment destroys a value you still need.
2. `*a = *b;` copies the int that `b` points at into the int that `a` points at.
3. Check what happens when both arguments are the same variable.

**Test:** `ctest -R 01-swap --output-on-failure`.
