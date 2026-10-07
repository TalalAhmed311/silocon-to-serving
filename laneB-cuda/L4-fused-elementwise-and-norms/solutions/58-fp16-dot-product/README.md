# 58: FP16 Dot Product (L4, core)

Problem: [LeetGPU #58](../../leetgpu-map.md). Dot product of fp16 vectors.

**Hint ladder**

1. Load `__half2`: two halves per 4-byte access, so half the instructions.
2. **Accumulate in fp32.** fp16 has an 11-bit significand: a running sum past 2048 can't absorb +1 anymore (the test proves it with 100,000 ones). fp16 also overflows at 65504.
3. A fp16 × fp16 product is exact in fp32 (11 + 11 bits ≤ 24). Only the summation order differs from the reference, so the tolerance is the same as #17's.
4. Grid-stride, warp shuffles, one atomic per block (#4, #17). Handle an odd `n`.

**Solution outline:** `dot_half2`: `__half22float2` → `fmaf` into an fp32 accumulator → warp/block reduce → `atomicAdd`.

**Why this is fast:** it reads 4 bytes per element pair vs 8 for fp32: half the traffic for the same answer quality.
