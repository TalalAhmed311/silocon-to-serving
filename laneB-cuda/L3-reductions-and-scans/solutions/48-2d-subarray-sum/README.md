# 48: 2D Subarray Sum (L3, practice)

Problem: [LeetGPU #48](../../leetgpu-map.md). Sum a sub-rectangle of a row-major matrix.

**Hint ladder**

1. Map threads along **columns**, so a warp reads 32 consecutive ints of one row (coalesced). Map `blockIdx.y` to rows.
2. Cap the grid and loop, grid-stride in both dimensions, so a huge rectangle doesn't launch millions of blocks that each do one atomic.
3. For many queries: a 2D **summed-area table** (scan rows, then scan columns) answers each query with 4 lookups: `P[r1+1][c1+1] − P[r0][c1+1] − P[r1+1][c0] + P[r0][c0]`.

**Solution outline:** `rect_sum` with a 2D grid (≤ 8 × 1024 blocks), coalesced row reads, warp shuffles, one integer atomic per block.

**Why this is fast:** each row segment is read with full 128-byte transactions. Narrow rectangles (a few columns) waste most of each warp, so try one warp per row in that case.
