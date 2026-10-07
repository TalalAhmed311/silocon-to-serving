# 64: Weight Dequantization (L4, practice)

Problem: [LeetGPU #64](../../leetgpu-map.md). Expand quantized weights to floats: `w = q · scale` (− zero-point for asymmetric schemes), with one scale per **group** of columns.

**Hint ladder**

1. Elementwise with a gather of the scale. Make each thread handle 4 int8 values: a 4-byte `char4` load, one scale, a 16-byte store.
2. Group-wise scales (group = 32/64/128 along K) are what GPTQ/AWQ-style formats use (P2.5). Per-tensor or per-channel scales are special cases.
3. int4 formats pack two values per byte: unpack with shifts and masks (`(b & 0xF) − 8`, `(b >> 4) − 8` for symmetric int4). The exact packing order is format-specific, so read the spec.
4. The real lesson: **don't materialize the dequantized weights.** Fuse dequant into the GEMM/GEMV that consumes them (L5 #81: W4A16), so only the compressed bytes cross HBM. That's the whole point of weight-only quantization for memory-bound decode.

**Solution outline:** `dequant_k`: a grid-stride loop over 4-element chunks.

**Why it matters:** 4× (int8) or 8× (int4) fewer weight bytes per decode step, as long as dequantization happens in registers.
