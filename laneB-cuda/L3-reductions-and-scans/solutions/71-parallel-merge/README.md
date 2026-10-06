# 71: Parallel Merge (L3, stretch)

Problem: [LeetGPU #71](../../leetgpu-map.md). Merge two sorted arrays.

**Hint ladder**

1. Give each thread a fixed slice of the **output**, e.g. 8 elements. The question is then where in A and B that slice starts.
2. **Merge path**: the first `d` outputs consist of `i` elements of A and `d − i` of B, for the unique `i` where `A[i−1] ≤ B[d−i]` and `B[d−i−1] < A[i]`. Binary-search `i` along the diagonal: O(log(m+n)) per thread.
3. Then merge sequentially from `(i, d − i)` for 8 outputs. Ties: `<=` takes A first, which keeps the merge stable.
4. Faster version: one merge-path search per **block** puts the block's A and B ranges into shared memory, then a per-thread search inside shared memory. That's coalesced loads, no global binary searches per thread.

**Solution outline:** `merge_path` (binary search on the diagonal) + an 8-item sequential merge per thread.

**Why this is fast:** perfectly load-balanced. Every thread does exactly ITEMS outputs whatever the data distribution, which a "split A evenly" scheme can't promise. Merge path is the core of GPU merge sort and of sorted-run joins.
