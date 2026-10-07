# 105: Group Normalization (L4, practice)

Problem: [LeetGPU #105](../../leetgpu-map.md). Statistics per (sample, group of channels) over all of the group's spatial positions.

**Hint ladder**

1. In `[N, C, S]` layout, a (sample, group) is one **contiguous** run of `(C/G)·S` values: one block per (n, group), and coalesced strided loops.
2. Two passes over global memory (mean, then centered variance) are stable and simple. Welford (#113) makes it one.
3. The affine is per **channel**, not per group: channel = `group·(C/G) + i / S`.
4. G = 1 is LayerNorm over (C, S). G = C is InstanceNorm. Make sure both work.

**Solution outline:** `groupnorm_k`: block sum → mean → block sum of squared deviations → rstd → normalize + per-channel affine.

**Why it matters:** diffusion models (L6 #116, DiT) use GroupNorm heavily. It's the same reduction skeleton as #50 and #113, with a different grouping.
