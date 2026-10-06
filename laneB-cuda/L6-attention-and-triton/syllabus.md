# L6 — Tiled attention, FlashAttention v1→v2, paged/decode attention, Triton rewrites

**Weeks 32–44 (P5.8 → P6 → Capstone) · daily · Tier T2 (the seq-4k timing needs a real GPU)**

**Objectives.**
1. Climb from naive attention to FA-1 and then FA-2 forward.
2. Add causal and sliding-window tile skipping.
3. Handle GQA, variable-length batching and decode attention over a quantized or paged KV cache.
4. Verify speculative decoding on the GPU.
5. Write a second pass of the key kernels in Triton.
6. Tackle the hard leaderboard problems: whole GPT-2 and Llama blocks.

**Problems.** See [leetgpu-map.md](leetgpu-map.md). Core:
- Softmax Attention
- Causal Self-Attention
- Multi-Head Attention
- Grouped Query Attention
- Variable-Length Causal Attention
- INT8 KV-Cache Attention
- Speculative Decoding Verification

**Exit check.** FA-2 forward is ≥5× faster than your naive attention at seq_len 4096 (fp16, head_dim 64/128, batch × heads stated). Check this in the harness against PyTorch SDPA for correctness. `TODO(run-on: g6.xlarge)`.

**Triton pass.** Rewrite softmax, RMSNorm, GEMM and FA-2 in Triton (Lane A P5.9) and compare them with your CUDA versions.

**Sources.**
- FA-1/FA-2 papers
- flash-attention-minimal
- Dao-AILab/flash-attention
- Triton tutorial `06-fused-attention.py`
- Lane A P5.8, P6.3
