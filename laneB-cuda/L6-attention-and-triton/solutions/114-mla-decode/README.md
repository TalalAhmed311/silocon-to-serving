# 114: Multi-Head Latent Attention Decode (L6, stretch: hints and outline only)

Problem: [LeetGPU #114](../../leetgpu-map.md). DeepSeek-style MLA: the cache stores a low-rank **latent** `c` per token instead of per-head K/V.

**Hint ladder**

1. K and V are up-projections of the latent: `k_h = W_uk,h · c`, `v_h = W_uv,h · c`. Decompressing per token at decode time is wasteful.
2. **Absorb** the up-projections: `q_h · k_h = (W_uk,hᵀ q_h) · c`. Project the query once into latent space, then dot it with cached latents directly. Similarly, accumulate `Σ p_j c_j` and apply `W_uv,h` once at the end.
3. Then it's decode attention (#96 without int8) over a smaller cache, plus two small GEMVs. Handle the decoupled RoPE part (a small rotary key shared across heads) as a separate dot product added to the score.

**Solution outline:** q_latent = W_ukᵀ·q (per head) → warp-split decode attention over latents (+ rope term) → `W_uv · (Σ p c)`.
