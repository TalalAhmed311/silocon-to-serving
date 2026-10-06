# LeetGPU map — L5 — GEMM ladder, tensor cores, async copies, double buffering

Challenge titles and difficulty come from [`AlphaGPU/leetgpu-challenges@37a1253`](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646), which is CC BY-NC-ND 4.0. We **do not** reproduce problem statements, starters or tests. Open each problem on https://leetgpu.com/challenges (the per-problem site URL is UNVERIFIED; the repo link below is verified).

Priority: **core** problems are required, **practice** problems are recommended, **stretch** and **hard** problems are optional.

| # | Challenge | Difficulty | Concept tested | Priority | Hint ladder | Solution |
|---|---|---|---|---|---|---|
| 22 | [General Matrix Multiplication (GEMM)](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/22_gemm) | medium | GEMM ladder rungs 1–7 on one problem | core (exit check: SGEMM ≥70% cuBLAS, local harness) | TODO | TODO |
| 30 | [Batched Matrix Multiplication](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/30_batched_matrix_multiplication) | medium | batched GEMM: grid z = batch | core | TODO | TODO |
| 57 | [FP16 Batched Matrix Multiplication](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/57_fp16_batched_matmul) | medium | WMMA / `mma.sync` HGEMM | core (exit check: HGEMM ≥50% cuBLAS) | TODO | TODO |
| 32 | [INT8 Quantized MatMul](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/32_int8_quantized_matmul) | medium | INT8 `dp4a` / IMMA, int32 accumulate, requantize | core | TODO | TODO |
| 81 | [INT4 Weight-Only Quantized MatMul](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/81_int4_matmul) | medium | INT4 unpack in registers, weight-only | stretch | TODO | TODO |
| 85 | [LoRA Linear](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/85_lora_linear) | medium | two GEMMs, low-rank fused epilogue | practice | TODO | TODO |
| 84 | [SwiGLU MLP Block](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/84_swiglu_mlp_block) | medium | GEMM + fused SwiGLU epilogue | practice | TODO | TODO |
| 115 | [Fused QKV Projection with RoPE and KV Cache Update](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/115_fused_qkv_rope_kv_cache_update) | medium | GEMM epilogue writes into a KV cache | stretch | TODO | TODO |
