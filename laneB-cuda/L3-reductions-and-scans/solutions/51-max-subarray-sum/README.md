# 51: Max Subarray Sum (L3, stretch)

Problem: [LeetGPU #51](../../leetgpu-map.md). Kadane's algorithm, made parallel. (If your statement fixes the subarray **length**, it's a different problem: sliding-window sums from a prefix sum, then a max-reduction.)

**Hint ladder**

1. Kadane is sequential, but a segment's answer can be built from a **summary** of its two halves: `(total, best prefix, best suffix, best)`.
2. `combine(L, R) = (L.t + R.t, max(L.pre, L.t + R.pre), max(R.suf, R.t + L.suf), max(L.best, R.best, L.suf + R.pre))` is **associative but not commutative**. Any reduction tree works, as long as it keeps left-to-right order.
3. So no atomics, and no "reduce in whatever order the warps finish". Use contiguous chunks per block and per thread, an in-order tree, and an in-order final pass.

**Solution outline:** `seg_partials` (each thread folds a contiguous run, then an in-order pairwise tree in shared memory) → `seg_final` (one thread folds the ≤ 4 × SMs block summaries).

**Why this is fast:** O(N) work and one read pass. Each thread's contiguous run isn't coalesced, so the next step is to stage each block's chunk through shared memory with coalesced loads, then fold per thread from shared memory. The test checks `combine` associativity directly.
