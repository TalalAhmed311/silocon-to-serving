# Exercise 4 — Multithread D1 (medium)

Implement `sgemm_threads(M, N, K, A, B, C, pool)` with the P0.3 thread pool. Split the **rows of C** into contiguous blocks with `pool.parallel_for(M, ...)` and call your `d1::sgemm_simd` on each block's sub-matrices (`A + i0*K`, `C + i0*N`).

**Why rows:** different row blocks write disjoint parts of C, so no locks are needed and no output lines are false-shared. All threads read all of B; it is shared read-only, which is cheap.

**Test:** equal to `sgemm_simd` single-threaded within `rtol=1e-6` (the per-row arithmetic is identical). Also prints a scaling table (1, 2, 4, … threads), which you should paste into your notes.
