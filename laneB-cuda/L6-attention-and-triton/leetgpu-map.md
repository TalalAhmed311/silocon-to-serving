# LeetGPU map — L6 — Tiled attention, FlashAttention v1→v2, decode attention, Triton

Challenge titles and difficulty come from [`AlphaGPU/leetgpu-challenges@37a1253`](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646), which is CC BY-NC-ND 4.0. We **do not** reproduce problem statements, starters or tests. Open each problem on https://leetgpu.com/challenges (the per-problem site URL is UNVERIFIED; the repo link below is verified).

Priority: **core** problems are required, **practice** problems are recommended, **stretch** and **hard** problems are optional.

| # | Challenge | Difficulty | Concept tested | Priority | Hint ladder | Solution |
|---|---|---|---|---|---|---|
| 6 | [Softmax Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/6_softmax_attention) | medium | naive → tiled → FA-1 → FA-2 forward | core (exit check: FA-2 ≥5× naive at seq 4k, local harness) | TODO | TODO |
| 53 | [Causal Self-Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/hard/53_casual_attention) | hard | causal masking + skipping fully masked tiles | core | TODO | TODO |
| 12 | [Multi-Head Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/hard/12_multi_head_attention) | hard | heads in the grid | core | TODO | TODO |
| 80 | [Grouped Query Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/80_grouped_query_attention) | medium | GQA: K/V shared across head groups | core | TODO | TODO |
| 59 | [Sliding Window Self-Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/hard/59_sliding_window_attn) | hard | banded mask, tile skipping | practice | TODO | TODO |
| 55 | [Attention with Linear Biases](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/55_attn_w_linear_bias) | medium | ALiBi bias in the score tile | practice | TODO | TODO |
| 102 | [Variable-Length Causal Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/102_varlen_causal_attention) | medium | varlen batching with `cu_seqlens` | core | TODO | TODO |
| 96 | [INT8 KV-Cache Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/96_int8_kv_cache_attention) | medium | decode attention over an int8 KV cache | core | TODO | TODO |
| 112 | [Attention with Sinks](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/112_attention_with_sinks) | medium | attention sinks | practice | TODO | TODO |
| 114 | [Multi-Head Latent Attention Decode](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/hard/114_multi_head_latent_attention) | hard | MLA decode (latent KV) | stretch | TODO | TODO |
| 119 | [Block-Sparse KV Selection Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/hard/119_block_sparse_kv_selection) | hard | block-sparse KV selection | stretch | TODO | TODO |
| 87 | [Speculative Decoding Verification](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/87_speculative_decoding_verification) | medium | accept/reject on GPU (ties to P6.5) | core | TODO | TODO |
| 111 | [Softmax Attention Backward](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/111_softmax_attention_backward) | medium | FA backward pass | stretch | TODO | TODO |
| 26 | [Multi-Head Cross-Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/hard/26_multi_head_cross_attention) | hard | cross-attention: Lq ≠ Lk | practice | TODO | TODO |
| 56 | [Linear Self-Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/hard/56_linear_attention) | hard | linear attention | stretch | TODO | TODO |
| 92 | [Decaying Causal Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/92_decaying_causal_attention) | medium | decaying causal attention | stretch | TODO | TODO |
| 74 | [GPT-2 Transformer Block](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/hard/74_gpt2_block) | hard | whole GPT-2 block from your own kernels | hard | TODO | TODO |
| 93 | [Llama Transformer Block](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/hard/93_llama_transformer_block) | hard | whole Llama block | hard | TODO | TODO |
| 116 | [Diffusion Transformer Block](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/hard/116_dit_block) | hard | DiT block | hard | TODO | TODO |
