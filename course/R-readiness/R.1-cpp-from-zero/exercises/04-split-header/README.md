# Exercise 4: Split a module into a header and a source file (medium)

This exercise has two files, like real C++ code:

- `stats.hpp` holds **declarations**: the name, parameters and return type of each function. Anyone who wants to call `mean` writes `#include "stats.hpp"`. The preprocessor pastes the file's text in.
- `stats.cpp` holds the **definitions**: the code. It is compiled once into a **library** (`stats_lib`). The test links that library.

**Task**

1. Replace the stubs in `stats.cpp` with real bodies for `mean`, `variance` and `range`. `variance` should call `mean` and the header's `square` helper.
2. Read the comment on `square` in `stats.hpp`. Remove the word `inline`, rebuild, and read the linker error (a "multiple definition" error). Put `inline` back. That error is the one-definition rule at work.

**Hints**

1. The test only includes `stats.hpp`. If `variance` were missing from `stats.cpp` you would get an *undefined reference* at link time, not a compile error.
2. Use `a[i]` indexing here. The point of this exercise is the file split, not pointer arithmetic.
3. `variance` here is the population variance (divide by `n`).

**Test:** `ctest -R 04-split-header --output-on-failure`.
