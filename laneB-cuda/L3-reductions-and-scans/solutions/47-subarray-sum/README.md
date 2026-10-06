# 47: Subarray Sum (L3, practice)

Problem: [LeetGPU #47](../../leetgpu-map.md). Sum of `input[S..E]`.

**Hint ladder**

1. With **one** query, this is just a reduction over the range. Start the grid-stride loop at `S`, stop at `E`.
2. With **many** queries over the same array, build an (exclusive) prefix sum once (#16), and each query becomes `P[E+1] − P[S]`: O(1) per query. That's the "prefix-sum trick". Know which situation you're in.
3. Integers make `atomicAdd` exact and order-independent. Watch for overflow: does the statement's range fit in 32 bits?

**Solution outline:** `range_sum`: grid-stride over `[S, E]`, warp shuffles, one integer atomic per block. Launch only as many blocks as the range needs.

**Why this is fast:** it reads only `E − S + 1` elements, once, coalesced.
