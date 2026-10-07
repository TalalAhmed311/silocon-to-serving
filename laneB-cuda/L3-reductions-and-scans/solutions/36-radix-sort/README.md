# 36: Radix Sort (L3, stretch)

Problem: [LeetGPU #36](../../leetgpu-map.md). Sort 32-bit unsigned keys.

**Hint ladder**

1. **LSD radix sort**: sort by the lowest 8 bits, then the next 8, and so on: 4 passes. It works only if each pass is **stable**.
2. One pass = histogram + scan + scatter. Count each digit per tile (#13). Exclusive-scan the `(digit, tile)` table in **digit-major** order (#16), so each tile's slice of digit d lands right after all earlier tiles' digit-d keys. Then scatter.
3. Stability inside a tile: sort the tile locally by the digit first. Eight stable 1-bit **splits** (zeros first, ones after, each via a block scan) give a stable 8-bit sort. A key's global position is then `offset[d][tile] + (local position − local start of d)`.
4. The local sort also makes the scatter **semi-coalesced**: keys with the same digit are contiguous in the tile and land contiguously in the output.

**Solution outline:** `tile_sort_and_count` (8 splits in shared memory → digit histogram) → generic `inclusive_scan` over `tiles × 256` counts → `scatter`. Four passes with ping-pong between `data` and a temp buffer, so the result ends in `data`.

**Why this is fast (and how CUB is faster):** O(N) work per pass and no comparisons. CUB's onesweep fuses passes with decoupled look-back and uses wider digit-processing per thread. Compare the two with `--bench`.
