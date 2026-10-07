# 72: Stream Compaction (L3, core)

Problem: [LeetGPU #72](../../leetgpu-map.md). Keep the elements that satisfy a predicate, preserving order.

**Hint ladder**

1. A kept element's output index is **the number of kept elements before it**, which is an exclusive scan of the 0/1 flags.
2. Three passes: flag → scan (reuse #16 with `int +`) → scatter `out[pos] = in[i]` where the flag is set. The scan's last element is the count.
3. To fuse: within a block, `__ballot_sync(mask, pred)` + `__popc` gives warp-level offsets with no shared memory, and one atomic per block reserves the block's output range. Order across blocks then needs a decoupled look-back (or ordered tickets), or else accept unordered output if the problem allows it.

**Solution outline:** `flag_kernel` → generic `inclusive_scan` over ints (in place) → `scatter` (the position is `pos[i−1]` when `pos[i]` changed).

**Why this is fast:** every phase is coalesced except the scatter writes, which are still in increasing order, so neighbors in a warp write nearby addresses. Compaction is how a serving engine removes finished sequences from a batch (P6).
