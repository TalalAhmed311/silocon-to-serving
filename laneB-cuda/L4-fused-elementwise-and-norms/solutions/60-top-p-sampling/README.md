# 60: Top-p Sampling (L4, stretch: hints and outline only)

Problem: [LeetGPU #60](../../leetgpu-map.md). Sample from the smallest set of tokens whose probability mass ≥ p.

**Hint ladder**

1. The textbook version sorts probabilities descending, prefix-sums them, cuts at p, renormalizes and samples. A full sort of 128k values per sequence per step is expensive.
2. **Sort-free:** find the probability threshold τ such that `Σ_{p_i ≥ τ} p_i ≥ p` by **binary search on τ**. Each probe is one parallel reduction over the vocabulary (`Σ p_i · [p_i ≥ τ]`). About 20 probes give float precision.
3. Then sample by inverse CDF among tokens with `p_i ≥ τ`: draw `u·mass`, and scan (block prefix sum, L3 #16) to find the token where the cumulative mass crosses it.
4. Do it in **one block per sequence**, with the row in shared memory if it fits, else in tiles.

**Solution outline:** softmax (#5) → binary search for τ with block reductions → masked block scan → pick the index. Exercise: implement it, then compare with a CUB `DeviceSegmentedRadixSort` + scan version. P6.5 uses this for GPU sampling.
