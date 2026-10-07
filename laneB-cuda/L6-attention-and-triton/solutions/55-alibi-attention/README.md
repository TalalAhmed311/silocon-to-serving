# 55: Attention with Linear Biases (L6, practice)

Problem: [LeetGPU #55](../../leetgpu-map.md). Add `−m_h·(i − j)` to every score. No positional embeddings.

**Hint ladder**

1. The bias depends only on (head, distance), so compute it in the score loop. Never materialize an N×N bias matrix.
2. Slopes: a geometric sequence `2^{−8(h+1)/H}` for H a power of two (the paper has a rule for other H).
3. Add the bias **before** the running max, because it changes which score is largest.

**Solution outline:** the shared core with an `alibi` slope array.
