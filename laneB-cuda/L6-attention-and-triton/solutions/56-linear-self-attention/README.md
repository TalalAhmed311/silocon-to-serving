# 56: Linear Self-Attention (L6, stretch: hints and outline only)

Problem: [LeetGPU #56](../../leetgpu-map.md). Replace softmax with a feature map φ: `out_i = φ(q_i)ᵀ Σ_j φ(k_j) v_jᵀ / φ(q_i)ᵀ Σ_j φ(k_j)`.

**Hint ladder**

1. Associativity again (L5 #85): compute `S = Σ_j φ(k_j) v_jᵀ` (a d×d matrix) and `z = Σ_j φ(k_j)` **once**. Then each query costs O(d²), so the total is O(N·d²) instead of O(N²·d).
2. Causal linear attention needs **prefix** sums of `φ(k_j) v_jᵀ`, which is a scan over matrices (L3 #16/#82 style, chunked).
3. φ is typically `elu(x) + 1` (Katharopoulos et al.). Check the statement.

**Solution outline:** non-causal: a reduction to S, z, then a per-query GEMV. Causal: a chunked scan of (S, z) with intra-chunk attention.
