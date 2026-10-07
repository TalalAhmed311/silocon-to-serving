# 111: Softmax Attention Backward (L6, stretch: hints and outline only)

Problem: [LeetGPU #111](../../leetgpu-map.md). Gradients dQ, dK, dV of softmax attention.

**Hint ladder**

1. With `P = softmax(S)`, `O = PV`: `dV = Pᵀ dO`, `dP = dO Vᵀ`, `dS = P ⊙ (dP − D)` with `D_i = Σ_j dO_ij O_ij` (row-wise), `dQ = dS K / √d`, `dK = dSᵀ Q / √d`.
2. FlashAttention's backward **recomputes** P tile by tile from Q, K and the saved log-sum-exp `L_i = m_i + log l_i` (save L in the forward pass: one float per row). It never stores P.
3. FA-2's backward parallelizes over **K/V tiles** (each block owns dK, dV for its tile and loops over Q tiles), with dQ accumulated by atomics or a separate pass.

**Solution outline:** forward saves L → preprocess D = rowsum(dO ⊙ O) → per K/V tile: loop Q tiles, recompute P, accumulate dV, dK, atomically add dQ. Check against autograd of a naive PyTorch implementation.
