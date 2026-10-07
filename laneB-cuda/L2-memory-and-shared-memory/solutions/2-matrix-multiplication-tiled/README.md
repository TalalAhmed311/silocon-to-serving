# 2 (revisited) — Shared-memory tiled matmul (L2)

Problem: [LeetGPU #2](../../leetgpu-map.md), the same problem as the L1 exit check, now fast. Harness convention: `A: M×N`, `B: N×K`, `C: M×K`.

**Hint ladder**

1. In the naive kernel, each thread reads a row of A and a column of B from global memory. A 16×16 block reads each A element 16 times and each B element 16 times.
2. Load a 16×16 tile of A and a 16×16 tile of B into shared memory, `__syncthreads()`, accumulate 16 products from shared memory, `__syncthreads()` again, then move to the next tile along N.
3. Bounds: pad out-of-range tile elements with 0, so the inner loop needs no conditionals.

**Solution outline:** the classic PMPP tiled matmul, with two `__syncthreads()` per tile step (one after loading, one before overwriting).

**Why this is fast:** global traffic drops by the tile width (16×), so arithmetic intensity rises from ~0.25 to ~4 FLOP/byte. It is still far from cuBLAS: register blocking, vectorized loads, double buffering and tensor cores are L5's ladder (P5.6/P5.7).
