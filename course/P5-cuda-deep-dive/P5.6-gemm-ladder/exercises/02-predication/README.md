# Exercise 2 (hard): any shape, with predication

Rungs 4–7 assume M, N, K are multiples of the tile. Real shapes aren't: decode GEMMs have M = batch size (1–64), and vocab projections have N = 128,256.

Implement `sgemm_any` (in `kernel.cuh`) with the rung-5 structure, plus:

- **zero-filled** loads for tile elements outside the matrix (a 0 contributes nothing to the dot product)
- **guarded** stores
- *(stretch)* a block-uniform fast path: if the whole tile is in range, skip the per-element checks. Is the difference measurable?

The test covers 1×1×1, a single-row M = 1 (decode-shaped), K = 3, and non-multiples everywhere. Then bench M = 1, 8, 64 at N = K = 4096 against cuBLAS, which picks different kernels per shape. How far behind are you at M = 1, and why? (It's a GEMV: memory-bound. Your 128-row tile wastes 127/128 of its compute.)
