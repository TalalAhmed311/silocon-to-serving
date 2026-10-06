# 18: Sparse Matrix-Vector Multiplication (L3, core)

Problem: [LeetGPU #18](../../leetgpu-map.md). `y = A·x` with A in CSR. If your statement gives a dense matrix, see the note at the top of `kernel.cu`.

**Hint ladder**

1. Thread-per-row is the obvious mapping. Neighboring threads then read *different* rows' `vals`, so the reads are uncoalesced, and long rows serialize.
2. **Warp per row**: the 32 lanes stride over one row's nonzeros, so `vals` and `col_idx` reads are coalesced. Finish with a 5-step `__shfl_down_sync` reduction.
3. Pick the mapping by row length. When the average nnz per row is below ~4, a warp is mostly idle, and thread-per-row (or a few rows per warp) wins.
4. `x[col_idx[k]]` is a gather. Read it through the read-only cache (`__ldg`) and hope for locality. Reordering the matrix (RCM) improves it.

**Solution outline:** `spmv_warp_per_row` (lanes stride over the row, shuffle-reduce, lane 0 writes) and `spmv_thread_per_row`, chosen by average nnz per row.

**Why this is fast:** SpMV has ~0.25 FLOP/byte and is entirely memory-bound. The goal is to read `vals` and `col_idx` once with full transactions. Report GB/s against the copy bandwidth, not GFLOP/s.
