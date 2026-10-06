# 69 — 2D Jacobi Stencil (L2, practice)

Problem: [LeetGPU #69](../../leetgpu-map.md). Repeatedly replace each interior point with the average of its 4 neighbours, keeping the boundary fixed, for `iters` iterations. **Check the statement** for the exact update formula (some variants include the centre point) and the boundary rule.

**Hint ladder**

1. Each iteration reads the *old* grid and writes the *new* one. In-place updates are Gauss-Seidel, a different algorithm. Use two buffers and swap the pointers.
2. One kernel launch per iteration is the simplest correct barrier between iterations (a grid-wide sync without cooperative groups).
3. Shared-memory tiles with a 1-cell halo cut the 5 reads per point to ~1 from DRAM. *Temporal blocking* (several iterations per launch on a shrinking tile) is the advanced version.

**Solution outline:** two buffers, one launch per iteration, and the boundary copied once at the start.

**Why it matters:** stencils are memory-bound (5 reads + 1 write for 4 FLOPs), so every optimization is about reusing loaded data: spatial reuse via the tile, temporal reuse via blocking.
