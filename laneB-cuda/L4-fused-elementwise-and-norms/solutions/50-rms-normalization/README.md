# 50: RMS Normalization (L4, core)

Problem: [LeetGPU #50](../../leetgpu-map.md). `y = x · rsqrt(mean(x²) + ε) · w`.

**Hint ladder**

1. One block per row. Each thread accumulates `x²` over a strided slice, then block-reduce (warp shuffles + one shared slot per warp).
2. `rsqrtf` once per row, then a second pass scales. That second read usually hits in L1/L2 for rows of a few KB.
3. `float4` loads when `cols % 4 == 0`: a quarter of the load instructions.
4. Accumulate in fp32 even when the data is fp16/bf16 (P5.5 §3).

**Solution outline:** `rmsnorm_rows`: a strided sum of squares (vectorized path) → `block_sum` → scale.

**Why this is fast:** memory-bound at ~1 read + 1 write per element, and the reduction is a handful of shuffles. Compare GB/s with copy.
