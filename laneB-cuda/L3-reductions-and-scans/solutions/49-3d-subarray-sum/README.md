# 49: 3D Subarray Sum (L3, stretch)

Problem: [LeetGPU #49](../../leetgpu-map.md). Sum an inclusive box of a 3D row-major volume.

**Hint ladder**

1. Only the innermost dimension is contiguous. Give it to `threadIdx.x`.
2. Flatten (depth, row) pairs into one index on `blockIdx.y` (`pr / rows`, `pr % rows`). That's one launch, with no 3D grid limits to worry about.
3. A 3D summed-area table answers repeated queries with 8 lookups (inclusion–exclusion), if you ever need many.

**Solution outline:** `box_sum`: blocks iterate over (plane, row) pairs, threads over the column run, then warp shuffles and one integer atomic per block.

**Why this is fast:** the same as #48. Coalesced reads along the contiguous axis, bounded atomics. Thin boxes (few columns) underuse warps.
