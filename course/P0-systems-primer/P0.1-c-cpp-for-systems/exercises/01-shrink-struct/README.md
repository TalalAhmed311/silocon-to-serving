# Exercise 1 — Shrink a struct (easy)

`impl.hpp` declares `RequestRecord`, a per-request bookkeeping struct like the one a serving scheduler keeps for every in-flight request. As written it wastes bytes on padding.

**Task:** reorder the fields, without removing, renaming or retyping any, so that `sizeof(RequestRecord) == 32`.

**Hints**

1. List each field's size and alignment.
2. Sort the fields by alignment, largest first.
3. Check yourself with `offsetof` before you run the test.

**Test:** `ctest -R 01-shrink-struct`. The test checks the size, checks that every field still exists with its type, and runs a round-trip.
