# 38: Nearest Neighbor (L3, practice)

Problem: [LeetGPU #38](../../leetgpu-map.md). For each 3D point, find the index of its nearest other point.

**Hint ladder**

1. Brute force is O(N²) distance evaluations, so it's **compute-bound**. Memory only matters if every thread re-reads every point from global memory.
2. Tile the "other" points through shared memory. Each block loads 256 points once, and every thread compares its own point against all 256. The smem reads are **broadcasts**: the whole warp reads the same `sx[k]`.
3. Store the tile as structure-of-arrays (`sx`, `sy`, `sz`) in shared memory, compare squared distances (no `sqrt`), and use `fmaf`.
4. Tie-breaking: scanning `j` in increasing order with a strict `<` gives the smallest index.

**Solution outline:** one thread per query point, a loop over 256-point tiles, `best`/`best_j` kept in registers.

**Why this is fast:** global traffic drops from N² to N²/256 point loads, and the inner loop is 3 subtractions + 3 FMAs per pair at shared-memory broadcast speed. For large N, a spatial grid or k-d tree beats brute force asymptotically, but that's a different problem.
