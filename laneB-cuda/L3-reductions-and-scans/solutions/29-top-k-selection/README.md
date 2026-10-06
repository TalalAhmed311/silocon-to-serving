# 29: Top-K Selection (L3, core)

Problem: [LeetGPU #29](../../leetgpu-map.md). Return the k largest values, in descending order.

**Hint ladder**

1. Sorting everything (#15) and taking k works, but it does O(N log² N) work for a k ≪ N answer.
2. **Tournament rounds.** Split the input into tiles, sort each tile in shared memory, and keep only its top K' (K' = next power of two ≥ k). The true top-k survive every round, and the candidate set shrinks by 2048/K' per round.
3. Use the *local* index for the bitonic direction so every tile sorts the same way (unlike #15, which alternates on purpose).
4. Large k (> 1024 here) means a round can't shrink the set. Switch to a full sort, or to a **radix select** that finds the k-th value's bits one digit at a time with histograms (#13 + #16), then compact everything ≥ it (#72).

**Solution outline:** `tile_topk` (load with `-inf` padding → full bitonic sort, descending → write the first `keep`), repeated with ping-pong buffers until one tile remains. Sampling top-k (P2 decode) works on the same idea, over a vocabulary of ~128k logits.

**Your task:** the `k > 1024` branch is left as a `TODO(learner)`. Implement the fallback, then add a test with `k = 5000`.
