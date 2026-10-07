# Exercise 3: block scan + scan-then-propagate (T2)

Implement `inclusive_scan(in, out, n)` in `kernel.cuh`:

1. **Block scan** of one 2048-element tile: a coalesced load into shared memory, each thread scanning 8 consecutive items serially, then an exclusive scan of the 256 thread totals (warp shuffles `__shfl_up_sync` within warps, then across the 8 warp totals), then adding back and storing.
2. **Any n:** tiles write their totals. Scan the totals (recursively, using the same function), then add each tile's carry-in.
3. *(optional)* **Single pass** with decoupled look-back. Read Merrill & Garland (2016) first, then compare with `d4/scan.cuh`.

The test uses small integers stored as floats, so every partial sum is exact and the check is bit-for-bit. `--bench` reports GB/s at 2²⁶. Compare against `examples/03_scan.cu`'s CUB row.
