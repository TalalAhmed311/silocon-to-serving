# Exercise 2 — Pad the counters (easy)

`impl.hpp` defines `PerWorkerCounters<N>`, an array of per-worker request counters. Each worker increments only its own slot, but the slots false-share.

**Task:** change `Slot` so each one owns a full 64-byte cache line. Keep `Slot::value` an `std::atomic<uint64_t>`. Then run `bench/false_sharing_bench` and record the speedup.

**Test:** checks `sizeof(Slot) == 64`, `alignof(Slot) == 64`, that consecutive slots sit on different lines, and an exact multi-threaded count.
