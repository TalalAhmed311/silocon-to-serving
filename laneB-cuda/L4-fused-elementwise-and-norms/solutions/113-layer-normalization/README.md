# 113: Layer Normalization (L4, core)

Problem: [LeetGPU #113](../../leetgpu-map.md). `y = (x − μ)/√(σ² + ε) · γ + β` per row.

**Hint ladder**

1. Two passes (mean, then variance) is correct but reads the row twice before the output pass.
2. One pass with `Σx` and `Σx²` and `var = E[x²] − E[x]²` is fast, but **catastrophic cancellation** hits when |μ| ≫ σ (the test's `large_mean` case).
3. **Welford** per thread (`mean += δ/n; m2 += δ·(x − mean)`) is stable. Merge partial states with Chan's formula: `δ = μ_b − μ_a; m2 = m2_a + m2_b + δ²·n_a·n_b/n`.
4. LayerNorm uses the **biased** (population) variance: divide by n, not n − 1.

**Solution outline:** a per-thread Welford loop → shuffle merges within warps → one shared slot per warp → a second shuffle merge → the normalize + affine pass.

**Why this is fast:** still one read for the statistics and one read+write for the output, and stable at any mean. Compare with your RMSNorm (#50): LayerNorm needs the extra mean term, and that's most of the difference.
