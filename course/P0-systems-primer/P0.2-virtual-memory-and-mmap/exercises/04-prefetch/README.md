# Exercise 4 — Prefetch with `MADV_WILLNEED` (hard)

A cold model load faults pages in one at a time, as the forward pass first touches them. That is a disk round trip per fault group. Add a prefetch path:

```cpp
// Map `path` read-only and ask the kernel to start reading it now, in the background.
// Returns the mapping; touching it afterwards should mostly hit the page cache.
Mapping map_with_prefetch(const std::string& path, bool prefetch);
```

Use `madvise(p, len, MADV_WILLNEED)`, which is portable to macOS. On Linux, also try `MAP_POPULATE` and `readahead(2)`, and compare them in the bench.

**Correctness test (ctest):** the checksum of the mapped bytes matches a `read()`-based checksum, with and without prefetch.

**Speed (bench):** extend `bench/mmap_bench.c` with a `mmap + WILLNEED` row. Run it cold (as root: `sync; echo 3 > /proc/sys/vm/drop_caches`) and report the table in your notes. Expected shape: prefetch helps most on spinning or network disks and when compute overlaps I/O; on a fast NVMe with a tight touch loop, the gain can be small.
