# 74: GPT-2 Transformer Block (L6, hard: assemble from your kernels)

Problem: [LeetGPU #74](../../leetgpu-map.md). One full GPT-2 block: LayerNorm → QKV GEMM → causal MHA → output projection + residual → LayerNorm → MLP (GELU) + residual.

**Outline:** LayerNorm (L4 #113) → GEMM (L5 #22) → split heads (strides) → causal attention (#53/#12, the shared core) → GEMM + fused residual add → LayerNorm → GEMM + fused GELU epilogue (L4 #65 activation) → GEMM + residual. Test against a NumPy/PyTorch GPT-2 block with the same random weights (llm.c's `train_gpt2.c` forward is the CPU reference structure; read it, MIT). Then profile with nsys: which kernel dominates at batch 1, and at 4096 tokens?
