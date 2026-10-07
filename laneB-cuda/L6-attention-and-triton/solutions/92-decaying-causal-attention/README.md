# 92: Decaying Causal Attention (L6, stretch: hints and outline only)

Problem: [LeetGPU #92](../../leetgpu-map.md). Causal attention with a distance decay `γ^{i−j}` (RetNet-style retention).

**Hint ladder**

1. If there's no softmax, it's linear attention with a decay: the recurrent form is `S_i = γ S_{i−1} + k_i v_iᵀ`, `out_i = q_i S_i`, which is a **linear recurrence** over d×d states (L3 #82).
2. The parallel (training) form is `(Q Kᵀ ⊙ D) V` with `D_ij = γ^{i−j}` for j ≤ i. Fold D into the score tile like ALiBi (#55), with a multiplicative instead of an additive bias.
3. The chunkwise form combines both: parallel inside chunks, recurrent across chunks.

**Solution outline:** start from the core's tile loop with `score · γ^{pos−j}` (no softmax), then try the chunkwise recurrence and compare.
