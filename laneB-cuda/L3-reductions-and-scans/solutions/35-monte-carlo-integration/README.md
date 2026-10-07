# 35: Monte Carlo Integration (L3, practice)

Problem: [LeetGPU #35](../../leetgpu-map.md). Estimate `∫_a^b f(x) dx ≈ (b − a) · mean(f(x_i))` for uniform `x_i`.

**Hint ladder**

1. If the samples are given, this is a reduction (#4) followed by one multiply.
2. Scale each block's partial by `(b − a)/n` before the atomic. Partials stay O(1) and float error doesn't grow with n.
3. If you generate samples on the GPU, use a **counter-based RNG** (cuRAND Philox): `curand_init(seed, subsequence = thread id, 0)`. Each thread gets an independent stream with no shared state. Never use one global RNG state with atomics.

**Solution outline:** `scaled_sum` (grid-stride sum → block shuffle-reduce → `atomicAdd(out, partial · scale)`), plus `mc_x2`, which generates `x` with Philox and integrates x² as a convergence demo.

**Why this is fast:** memory-bound for given samples, and compute-bound (RNG) for generated ones. The error falls as 1/√n, so 4× the samples halves it. The test's tolerance is 5σ of that.
