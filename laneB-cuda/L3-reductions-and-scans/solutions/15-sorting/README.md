# 15: Sorting (L3, core)

Problem: [LeetGPU #15](../../leetgpu-map.md). Sort N floats ascending.

**Hint ladder**

1. **Bitonic sort** is a fixed network of compare-exchanges. Stage (k, j) compares `i` with `i ^ j` and sorts ascending iff `(i & k) == 0`. That's O(N log² N) work, with no data-dependent control flow and perfectly parallel.
2. Pad to a power of two with `+inf`. Padding sorts to the end, so copy back only the first N.
3. Most stages have a small distance j. Run **all** stages with `j < TILE` inside shared memory: one global read and one write per tile per merge level, instead of one per stage. Only `j ≥ TILE` stages touch global memory.
4. Enumerate *pairs*, not elements. `i = 2t − (t & (j − 1))` maps pair t to its lower index, so no thread idles.

**Solution outline:** `bitonic_tile<true>` sorts each 2048-element tile (directions alternate by global index). Then for each k > 2048: `bitonic_global` for j ≥ 2048, and `bitonic_tile<false>` finishes j < 2048 in shared memory.

**Why this is (reasonably) fast:** with log²(N/TILE) global passes instead of log² N, it's bandwidth-friendly for moderate N. For large N, a radix sort (#36, and CUB's `DeviceRadixSort`) wins at O(N · passes). Bitonic's strength is small fixed-size sorts inside a block, which you'll reuse in top-k (#29).
