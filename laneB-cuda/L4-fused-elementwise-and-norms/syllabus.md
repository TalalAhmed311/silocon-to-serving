# L4 — Online softmax, fused norms, `float4` loads, fp16/bf16

**Weeks 25–27 (with P5.1–P5.3) · daily ≈ 1 h · Tier T1 + T2**

**Objectives.**
1. Write single-pass online softmax.
2. Write row-wise RMSNorm and LayerNorm with warp/block reductions.
3. Fuse a residual add into a norm.
4. Use vectorized `float4`/`half2` loads.
5. Do fp16/bf16 math with fp32 accumulation, and derive the test tolerances.
6. Write RoPE and gated activations (SwiGLU, GEGLU).

**Problems.** See [leetgpu-map.md](leetgpu-map.md). Core:
- Softmax
- RMS Normalization
- Layer Normalization
- Fused Residual Add and RMS Norm
- Categorical Cross Entropy Loss
- SwiGLU
- GEGLU
- FP16 Dot Product
- Rotary Positional Embedding

The prompt asks for GELU. The challenge set has no plain GELU, so GEGLU stands in; see SOURCES.md §3.

**Exit check.** A single-pass online softmax over rows of 4k–32k elements, reported in GB/s and as % of copy bandwidth.

**Sources.**
- llm.c `dev/cuda/softmax_forward.cu` and `layernorm_forward.cu` (annotated)
- Lane A P5.5
