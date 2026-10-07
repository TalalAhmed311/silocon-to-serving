# 61: Rotary Positional Embedding (L4, core)

Problem: [LeetGPU #61](../../leetgpu-map.md). Rotate pairs of dimensions of q/k by position-dependent angles.

**Hint ladder**

1. Know the **convention**. *rotate_half* (Llama/HF) pairs `(i, i + D/2)`. *Interleaved* (GPT-NeoX-style, GPT-J) pairs `(2i, 2i+1)`. Same math, different pairs: the wrong one passes shape checks and fails numerics. #0's engine (P0.5) uses rotate_half.
2. Angle `θ_i(p) = p · base^{−2i/D}`. Compute it on the fly with `sincosf`, or precompute a `[max_pos, D/2]` cos/sin table (memory vs compute; measure both).
3. One thread per pair, a block per (token, head). In place is fine: each pair is read and written by one thread.
4. Precision: the angle can be thousands of radians, and `sincosf`'s absolute error grows with |angle|. Real implementations compute `p·inv_freq` in fp32 (or fp64 for the table) and that's why the tolerance scales.

**Solution outline:** `rope_k`: `powf` for the inverse frequency, `sincosf`, a 2×2 rotation.

**Why it matters:** RoPE runs on q and k every layer, every token. It's usually fused into the QKV-projection epilogue (L5 #115) or into the attention kernel.
