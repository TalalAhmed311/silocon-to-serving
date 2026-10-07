# 30: Batched Matrix Multiplication (L5, core)

Problem: [LeetGPU #30](../../leetgpu-map.md). Many independent GEMMs of the same shape.

**Hint ladder**

1. Don't loop over the batch on the host. Put it in **`gridDim.z`**: one launch, and every SM stays busy even when each GEMM is small.
2. Offset the pointers by `blockIdx.z × (matrix size)` at the top of the kernel. Everything else is your GEMM (#22).
3. Small matrices (attention heads: 64×64 per head) don't fill one tile's worth of reuse. Small tiles or one warp per matrix can win. Measure.
4. Strided-batched layouts (cuBLAS `gemmStridedBatched`) generalize this with arbitrary batch strides.

**Solution outline:** `bmm_k` = #22's tile core + a `blockIdx.z` pointer offset.

**Why it matters:** attention's `QKᵀ` and `PV` per head *are* batched GEMMs. MoE experts are grouped GEMMs (different M per expert): the next generalization.
