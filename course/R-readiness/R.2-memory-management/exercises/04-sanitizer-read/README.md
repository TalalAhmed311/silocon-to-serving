# Exercise 4: Read a sanitizer report (easy)

`impl.hpp` holds `make_ramp`, a small function with one planted bug. Without a sanitizer it might pass, crash or quietly corrupt memory. CMake builds this test with AddressSanitizer, so the bug stops the program with a report.

**Task**

1. Run the test and let it fail:
   ```bash
   cmake -S exercises -B /tmp/r2-ex && cmake --build /tmp/r2-ex -j
   ctest --test-dir /tmp/r2-ex -R 04-sanitizer-read --output-on-failure
   ```
2. Find the line that starts `ERROR: AddressSanitizer:`. Paste it into the comment in `impl.hpp` (the `ASan's first line` block).
3. Read the next few lines. They give the kind of error, whether it was a read or a write, the size, and a stack trace whose first frame inside your code is the line to look at. Also find the sentence `... is located 0 bytes after 16-byte region`. It tells you the buffer's size and how far past the end you went.
4. Fix the bug and rerun until the test passes.

**Hints**

1. "0 bytes after N-byte region" means the access started exactly at the end of the block: an off-by-one.
2. Compare the loop bound with the array length.
3. The test cannot check your pasted comment. Do it anyway: reading the first line of the report is the skill.
