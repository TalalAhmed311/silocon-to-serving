# 22: General Matrix Multiplication (L5, core, exit check via the D4 ladder)

Problem: [LeetGPU #22](../../leetgpu-map.md). `C = α·A·B + β·C`.

**Hint ladder:** this problem *is* P5.6. Climb the seven rungs there (naive → coalesced → shared tiles → 1D → 2D register blocking → vectorized → warp tiling + double buffering) with `platform/kernels/include/d4/gemm.cuh` as the reference. The **L5 exit check** (≥ 70% of cuBLAS) is measured on that ladder (`p5.6_01-gemm-rungs --bench`).

For LeetGPU's arbitrary shapes, add **predication**: zero-fill out-of-range tile loads (a zero contributes nothing) and guard the stores (P5.6 exercise 2). And `β = 0` must **not** read C: it may hold NaNs, and `0·NaN = NaN`.

**Solution outline:** a 64×64×16 tile, 4×4 outputs per thread, A staged k-major in shared memory, predicated loads and stores, and the α/β epilogue.

**Why this is fast enough:** 2D register blocking gives 4 FMAs per shared-memory load. It's slower than rung 7 at big aligned sizes, and correct everywhere.
