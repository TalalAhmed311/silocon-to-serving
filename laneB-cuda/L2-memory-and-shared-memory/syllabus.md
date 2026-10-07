# L2 — Coalescing, shared memory, `__syncthreads`, bank conflicts, atomics, occupancy

**Weeks 7–14 (alongside P2) · 3–4 problems/week · Tier T1 + T2 (bandwidth numbers need a real GPU and the harness)**

**Objectives.**
1. Make every global access coalesced.
2. Stage tiles in shared memory with correct barriers.
3. Remove bank conflicts by padding.
4. Load halos for stencils and convolutions.
5. Privatize atomics (per-block histograms).
6. Read achieved occupancy and choose a block size.

**Problems.** See [leetgpu-map.md](leetgpu-map.md). Core:
- Matrix Transpose
- 1D Convolution
- 2D Convolution
- Histogramming
- Count Array Element
- tiled Matrix Multiplication (revisited)

**Predict first.** Before benchmarking transpose, predict naive vs tiled GB/s from sectors fetched per warp, as in the P5.2 coalescing lesson. Lane B is ahead of Lane A here; the P5.2 animation is linked.

**Exit check.** Transpose reaches ≥80% of the bandwidth of a plain device copy kernel on the same GPU, measured with the harness. Report the table and the `ncu` sectors/request metric. `TODO(run-on: g4dn.xlarge)` until run.

**Sources.**
- PMPP ch 5–7
- CUDA Best Practices Guide (coalescing, shared memory)
- handbook MemoryCoalescingVisualizer
