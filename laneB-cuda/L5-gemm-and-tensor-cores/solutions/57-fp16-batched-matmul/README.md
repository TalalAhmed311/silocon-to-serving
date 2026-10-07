# 57: FP16 Batched Matrix Multiplication (L5, core; exit check via P5.7)

Problem: [LeetGPU #57](../../leetgpu-map.md). Batched GEMM with fp16 inputs on tensor cores.

**Hint ladder**

1. Use WMMA 16×16×16 fragments with fp32 accumulators (P5.7 exercise 1). Batch in `gridDim.z` (#30).
2. Stage A and B tiles in shared memory with **zero-fill**, so any M, N, K works. WMMA itself needs full 16×16 fragments.
3. Store accumulators to **shared memory**, then do a guarded copy to global (converting to fp16). Tensor-core fragments can't be stored partially.
4. Then `mma.sync` + `ldmatrix` + `cp.async` (P5.7 exercises 2–3) for speed. The **L5 HGEMM exit check (≥ 50% of cuBLAS)** is measured there.

**Solution outline:** `hbmm_wmma`: 64×64 tiles, 4 warps × (2×2 fragments), smem staging both ways.

**Tolerance:** fp16 × fp16 products are exact in fp32, so the main error is the single fp16 rounding of the output (2⁻¹¹ relative). Hence `2e-3`.
