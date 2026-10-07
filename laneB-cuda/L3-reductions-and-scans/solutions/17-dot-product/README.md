# 17: Dot Product (L3, core)

Problem: [LeetGPU #17](../../leetgpu-map.md). `result = Σ A[i]·B[i]`.

**Hint ladder**

1. Don't write `A[i]*B[i]` to a temporary array and then reduce it. That's two extra passes over memory. **Fuse** the multiply into the reduction's load loop.
2. Reuse your #4 reduction: grid-stride loop, `float4` loads from both arrays, `fmaf` into a register accumulator.
3. Block-reduce with warp shuffles, then one `atomicAdd` per block.

**Solution outline:** identical to #4, with `acc = fmaf(a, b, acc)` in the load loop.

**Why this is fast:** it reads 8 bytes per element and writes nothing, so the bound is read bandwidth, the same as #4. The FMA is free next to the memory traffic: arithmetic intensity is 2 FLOPs per 8 bytes.
