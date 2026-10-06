# Exercise 5 — Fix three planted memory bugs (hard)

`impl.hpp` contains a small ring buffer for token IDs. It is the kind of structure a streaming server uses to batch outgoing tokens. It has **three** memory bugs:

- an off-by-one out-of-bounds write
- a use-after-free
- a signed-overflow / bad-shift undefined behavior

In a normal build the test may pass or crash at random. Under AddressSanitizer + UndefinedBehaviorSanitizer it fails deterministically. CMake already builds this exercise with `-fsanitize=address,undefined`.

**Task:** read each sanitizer report, find the line it points at, and fix the bug without changing the public interface.

**Hints**

1. Run `ctest -R 05 --output-on-failure` and read the *first* error only. Fix it, then rerun.
2. A `heap-buffer-overflow ... 0 bytes after` report usually means a `<=` that should be `<`.
3. `heap-use-after-free` means you hold a pointer into memory a `resize` released. Re-fetch it.
4. `shift exponent ... too large` means `1 << n` with `n >= 32`. Use a 64-bit shift.

**Test:** the test passes cleanly under the sanitizers.
