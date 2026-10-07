# 59: Sliding Window Self-Attention (L6, practice)

Problem: [LeetGPU #59](../../leetgpu-map.md). Each query sees only the last w keys.

**Hint ladder**

1. Mask `i − j ≥ w` (and `j > i` when causal).
2. Skip K/V tiles outside the band on **both** ends. The loop starts at `i0 − w + 1` (rounded down to a tile) and ends at the causal bound, so the cost is O(N·w).
3. A window larger than N is just causal attention (the test covers it).

**Solution outline:** the shared core with `causal = true, window = w`.
