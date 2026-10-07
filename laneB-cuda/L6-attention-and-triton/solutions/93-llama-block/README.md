# 93: Llama Transformer Block (L6, hard: assemble from your kernels)

Problem: [LeetGPU #93](../../leetgpu-map.md). RMSNorm → QKV → RoPE → GQA causal attention → output projection + residual → RMSNorm → SwiGLU MLP + residual.

**Outline:** RMSNorm (L4 #50) → QKV GEMM (L5 #22; or #115 fused with RoPE) → RoPE (L4 #61, rotate_half) → GQA attention (#80, causal) → GEMM → fused residual + RMSNorm (L4 #83) → SwiGLU MLP (L5 #84) → residual. The reference is `platform/engine/v0/reference/llama_numpy.py` (P0.5). Match it at fp32 `atol 1e-4`. This block is what #0 v1 runs per layer in P6.7.
