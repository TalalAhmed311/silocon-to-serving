# 32: INT8 Quantized MatMul (L5, core)

Problem: [LeetGPU #32](../../leetgpu-map.md). int8 × int8 → int32 accumulation → requantize to int8.

**Hint ladder**

1. `__dp4a(a, b, c)` (sm_61+) does 4 int8 multiply-adds into an int32 in one instruction, with operands packed 4 per 32-bit register.
2. dp4a wants both operands **contiguous in k**. A (row-major) already is, but B (K×N row-major) isn't, so pack B's columns while loading the tile into shared memory (a transpose on load).
3. int32 accumulation is exact. Requantize once at the end: `round(acc · sA·sB/sC)`, then clamp to [−128, 127]. Precompute the combined scale. Per-channel scales make it a vector.
4. Tensor-core int8 (IMMA: `mma.sync … .s8.s8.s32`, sm_80+) is the next 2–4× (P5.7 exercise 5).

**Solution outline:** `qgemm_dp4a`: a 32×32 tile, A rows packed, B columns packed on load, 8 dp4a per output per k-tile, and a fused requantize epilogue.

**Why it matters:** W8A8 serving (P2.5) runs exactly this, with the requantization (and often the next layer's activation quantization) fused into the epilogue.
