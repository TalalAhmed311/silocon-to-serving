# Exercise 1 — Fix the data race (easy)

`impl.hpp` counts tokens produced by several worker threads into one shared `TokenStats`. It loses updates, so the total is usually wrong, and ThreadSanitizer reports a race.

**Task:** make `record()` thread-safe. Try both an `std::atomic` fix and a per-thread-accumulate fix. The interface must not change.

**Hints**

1. `uint64_t total` is read-modify-written by every thread.
2. `std::atomic<uint64_t>::fetch_add` is the minimal fix.
3. `max_batch` needs a compare-exchange loop: an atomic `max` isn't a single instruction.

**Test:** 8 threads × 200,000 records. The total and the max must be exact. Run with `-DS2S_TSAN=ON` too.
