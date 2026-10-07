# 96: INT8 KV-Cache Attention (L6, core)

Problem: [LeetGPU #96](../../leetgpu-map.md). Decode attention where the cached K and V are int8 with scales.

**Hint ladder**

1. Decode attention is **memory-bound on the KV cache** (one query per sequence, P1.2). int8 halves the bytes vs fp16. That's the point, so the dequantization must happen **in registers** on the fly, never as a separate pass.
2. With per-token scales, scale the **dot product** once (`s = (q·k_q)·k_scale`) instead of each element. For V, fold the scale into the softmax weight (`p·v_scale`).
3. Layout: one block per (sequence, query head). Warps take tokens round-robin, and lanes split the head dimension. Each warp keeps its own online-softmax state, and the warps merge at the end (P5.8 exercise 4).
4. GQA: several query heads read the same KV head, so `h / (H / Hkv)`.

**Solution outline:** `int8_decode`: per-lane dims, a warp shuffle dot product, scale, online update, int8 V accumulate, and the warp merge.

**Why it matters:** FP8/INT8 KV caches double the batch (and context) that fits in GPU memory (P2.3, D3's `kv_dtype`). This kernel is why it costs almost nothing in speed.
