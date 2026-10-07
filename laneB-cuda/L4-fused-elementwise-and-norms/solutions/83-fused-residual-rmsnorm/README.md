# 83: Fused Residual Add and RMS Norm (L4, core)

Problem: [LeetGPU #83](../../leetgpu-map.md). `r = x + residual`, `y = rmsnorm(r)·w`, in one kernel.

**Hint ladder**

1. Unfused: an add kernel (read x, read residual, write r), then RMSNorm (read r, write y). That's 5 row-passes.
2. Fused: compute `x + residual` once, write `r` (the next layer needs it), **keep it on-chip** (registers or shared memory) for the sum of squares and the scale. That's 4 row-passes and one launch.
3. Watch the numerics: if the residual is stored in bf16, normalize the *stored* (rounded) value so the result matches the unfused path bit for bit (P5.5 §3, D4 `fused_add_rmsnorm`).

**Solution outline:** one block per row. Pass 1: add, store `r`, stash it in dynamic shared memory, and accumulate `Σr²`. `block_sum`, `rsqrt`. Pass 2: scale from shared memory.

**Why this is fast:** fewer bytes and fewer launches. This exact fusion is a vLLM hot path, and you'll swap your own version into vLLM in P5.10 (#12).
