# 3 — Matrix Transpose (L2 exit check)

Problem: [LeetGPU #3](../../leetgpu-map.md). Transpose a `rows × cols` row-major matrix into `cols × rows`.

**Exit check:** ≥ 80% of the bandwidth of a plain device copy (`--bench` prints both).

**Hint ladder**

1. Naive (`out[c][r] = in[r][c]`, with x → c): reads coalesce, but the warp's 32 writes land 32 rows apart, 32 separate sectors. Measure it first.
2. Stage a 32×32 tile in **shared memory**: read the tile coalesced (x → column of `in`), `__syncthreads()`, then write it coalesced (x → column of `out`) by reading the tile *transposed*.
3. Reading `tile[threadIdx.x][threadIdx.y]` has a stride of 32 floats, so every lane hits the same **bank**: a 32-way conflict. Pad to `tile[32][33]`. Then use 32×8 threads per block, each looping over 4 rows, so each thread moves 4 elements.

**Solution outline:** `tile[32][33]`, block (32, 8), and each thread copies 4 elements in and 4 out, with bounds checks on both the read and the write.

**Why this is fast:** both global phases are fully coalesced (128 B per warp access), shared memory turns the transpose into on-chip data movement, and the +1 padding makes column reads hit 32 distinct banks. What's left is DRAM bandwidth, so the result approaches copy speed.
