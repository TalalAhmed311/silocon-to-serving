# LeetGPU map — L4 — Online softmax, fused norms, `float4`, fp16/bf16

Challenge titles and difficulty come from [`AlphaGPU/leetgpu-challenges@37a1253`](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646), which is CC BY-NC-ND 4.0. We **do not** reproduce problem statements, starters or tests. Open each problem on https://leetgpu.com/challenges (the per-problem site URL is UNVERIFIED; the repo link below is verified).

Priority: **core** problems are required, **practice** problems are recommended, **stretch** and **hard** problems are optional.

| # | Challenge | Difficulty | Concept tested | Priority | Hint ladder | Solution |
|---|---|---|---|---|---|---|
| 5 | [Softmax](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/5_softmax) | medium | single-pass online softmax (running max + sum) | core (exit check, report GB/s) | TODO | TODO |
| 50 | [RMS Normalization](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/50_rms_normalization) | medium | one block per row, warp reduction, `float4` loads | core | TODO | TODO |
| 113 | [Layer Normalization](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/113_layer_normalization) | medium | Welford or two-moment, fused affine | core | TODO | TODO |
| 83 | [Fused Residual Add and RMS Norm](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/83_fused_residual_add_rms_norm) | medium | fusion saves a full HBM round trip | core | TODO | TODO |
| 40 | [Batch Normalization](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/40_batch_normalization) | medium | per-channel reduction across batch | practice | TODO | TODO |
| 105 | [Group Normalization](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/105_group_normalization) | medium | grouped reduction | practice | TODO | TODO |
| 25 | [Categorical Cross Entropy Loss](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/25_categorical_cross_entropy_loss) | medium | log-sum-exp + gather, numerically stable | core | TODO | TODO |
| 54 | [Swish-Gated Linear Unit](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/54_swiglu) | easy | gated activation fused | core | TODO | TODO |
| 65 | [Gaussian Error Gated Linear Unit](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/65_geglu) | easy | GELU-family gate (stands in for GELU) | core | TODO | TODO |
| 58 | [FP16 Dot Product](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/58_fp16_dot_product) | medium | `half2` math, fp32 accumulation | core | TODO | TODO |
| 61 | [Rotary Positional Embedding](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/61_rope_embedding) | medium | RoPE: pairwise rotation, sin/cos table vs on-the-fly | core | TODO | TODO |
| 106 | [Token Embedding Layer](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/106_token_embedding_layer) | medium | embedding gather, memory-bound | practice | TODO | TODO |
| 64 | [Weight Dequantization](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/64_weight_dequantization) | medium | dequant fused into the read path | practice | TODO | TODO |
| 60 | [Top-p Sampling](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/60_top_p_sampling) | medium | sort-free top-p via threshold search | stretch | TODO | TODO |
| 104 | [Min-P Sampling](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/104_min_p_sampling) | medium | min-p filtering | stretch | TODO | TODO |
| 67 | [MoE Top-K Gating](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/67_moe_topk_gating) | medium | MoE router: softmax + top-k per token | stretch | TODO | TODO |
