# Exercise 1: Fix the leak (easy)

`sum_of_squares_below` allocates a temporary array with `new long[n]` and frees it with `delete[]` at the end. But one path leaves the function early with `return -1;` and never reaches the `delete[]`. The array leaks.

**Task:** change the function so it cannot leak on any path. Replace the raw pointer with a `std::unique_ptr<long[]>` made by `std::make_unique<long[]>(n)`. Delete the `delete[]` line: the smart pointer's destructor does it for you, on every exit.

**Hints**

1. `auto squares = std::make_unique<long[]>(n);` gives you something you can index with `squares[i]`, just like the raw array.
2. Do not call `delete[]` yourself once you use a `unique_ptr`. That would be a double free.
3. The test replaces the global `operator new` and `operator delete` to count live heap blocks. It compares the count before and after each call.

**Test:** `ctest -R 01-fix-the-leak --output-on-failure`. The "early return" test fails until the leak is fixed.
