# L3 — Tree reduction, warp shuffles, scan, cooperative groups

**Weeks 15–24 (alongside P3–P4) · 3–4 problems/week · Tier T1 + T2 (the CUB baseline needs a local GPU)**

**Objectives.**
1. Climb Harris's reduction ladder.
2. Use `__shfl_down_sync` / `__shfl_xor_sync` and cooperative groups.
3. Write work-efficient scans (Blelloch) and decoupled look-back.
4. Use scan as a building block (compaction, radix sort, segmented ops).
5. Do selection with top-k, and write a warp-per-row CSR SpMV.

**Problems.** See [leetgpu-map.md](leetgpu-map.md). Core:
- Reduction
- Dot Product
- Prefix Sum
- Stream Compaction
- Top K Selection
- Sorting
- Sparse Matrix-Vector Multiplication

**Exit check.** Sum-reduction of 2²⁴–2²⁸ floats, within 10% of `cub::DeviceReduce::Sum` (CCCL `v3.5.0`) on the same GPU, as the median of 100 runs. `TODO(run-on: g4dn.xlarge)`.

**Sources.**
- Mark Harris reduction PDF
- PMPP ch 10–12
- CCCL/CUB
- Lane A P5.4 (taken later, in week 28; L3 is deliberately earlier and more hands-on)
