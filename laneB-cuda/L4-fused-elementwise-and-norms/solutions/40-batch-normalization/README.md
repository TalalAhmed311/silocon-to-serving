# 40: Batch Normalization (L4, practice)

Problem: [LeetGPU #40](../../leetgpu-map.md). Normalize each **channel** over the batch.

**Hint ladder**

1. With `[N, C]` row-major, a channel is a **column**: one thread per channel walking down the rows is uncoalesced. Instead, put 32 channels across `threadIdx.x` (coalesced along a row) and spread rows over `threadIdx.y`.
2. Two kernels: statistics (mean, variance per channel), then apply. The apply step is elementwise and grid-stride.
3. `E[x²] − E[x]²` cancels badly when |μ| ≫ σ. Shift by any sample of the channel first (the first row here), or use Welford (#113).
4. Training-mode BN uses the **batch** statistics with biased variance. Inference uses running statistics, which is just an affine.

**Solution outline:** `bn_stats` (32×8 block, shifted sums, a smem merge across y) → `bn_apply`.

**Contrast with #113/#50:** LayerNorm and RMSNorm reduce along a contiguous row. BatchNorm reduces across rows, so the access pattern is the whole problem.
