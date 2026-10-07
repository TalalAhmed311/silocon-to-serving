# 75: Sparse × Dense Matrix Multiplication (L3, stretch)

Problem: [LeetGPU #75](../../leetgpu-map.md). `C = A_sparse · B_dense`, with A in CSR.

**Hint ladder**

1. Each output row `C[r, :]` is a weighted sum of rows of B: `Σ_k a(r,k) · B[k, :]`. Rows of B are contiguous, so map **threads to columns** of C and all reads of B are coalesced.
2. Use a block per row of A. Stage the row's `(k, a)` pairs through shared memory in chunks, so every thread reads them as a broadcast instead of re-reading global memory.
3. The work per row varies with nnz, which causes load imbalance. A grid-stride loop over rows helps, and binning rows by length helps more.
4. Unlike SpMV, B's rows are reused across the nonzeros of many A rows. For a large P, tile C's columns and keep B tiles in shared memory, or use cuSPARSE `SpMM` as the baseline.

**Solution outline:** `spmm_row`: a block per row (grid-stride), a 128-column strip per pass, nonzeros staged in shared-memory chunks of 128, and `fmaf` accumulation in registers.

**Why this is fast:** B reads are fully coalesced (each nonzero reads a contiguous 512-byte strip of a B row), and A's nonzeros are read once per strip. SpMM is how sparse attention and graph neural network layers multiply.
