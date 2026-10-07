# 116: Diffusion Transformer Block (L6, hard: assemble from your kernels)

Problem: [LeetGPU #116](../../leetgpu-map.md). A DiT block: adaLN-Zero modulation from the conditioning vector, self-attention, MLP, gated residuals.

**Outline:** conditioning MLP → per-block (shift, scale, gate) × 2. LayerNorm (L4 #113) with fused `(1 + scale)·x + shift` modulation → non-causal MHA (#12) → `x + gate ⊙ attn_out` → modulated LayerNorm → MLP (GELU) → gated residual. GroupNorm (L4 #105) appears in the surrounding U-Net/VAE parts. Test against a PyTorch DiT block (Peebles & Xie). Note the attention here is non-causal over image patches: long sequences, where FA-2 matters most.
