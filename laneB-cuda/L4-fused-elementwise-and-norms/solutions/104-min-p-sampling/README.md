# 104: Min-P Sampling (L4, stretch: hints and outline only)

Problem: [LeetGPU #104](../../leetgpu-map.md). Keep tokens with `p_i ≥ min_p · max_j p_j`, renormalize, sample.

**Hint ladder**

1. Unlike top-p, the cutoff is explicit once you know the max: **one** max reduction, no search.
2. Pass 1: online softmax statistics (#5) give `max` and the normalizer. Pass 2: mask `p_i < min_p · p_max`, compute the kept mass (block sum).
3. Sample by inverse CDF over the kept tokens: a block prefix sum and a search for `u · kept_mass`.
4. Edge cases: `min_p = 0` is plain sampling, `min_p = 1` is greedy (argmax, with ties).

**Solution outline:** one block per sequence: a max/sum pass → a masked mass pass → a scan-and-pick pass. It's simpler and cheaper than top-p, which is part of why engines added it.
