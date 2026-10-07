# 26: Multi-Head Cross-Attention (L6, practice)

Problem: [LeetGPU #26](../../leetgpu-map.md). Decoder queries attend to encoder keys/values: Lq ≠ Lk, no causal mask.

**Hint ladder**

1. The only differences from #12 are two lengths and no mask. The grid covers Lq, and the K/V loop covers Lk.
2. Encoder K/V are fixed across decoding steps, so cache them once. That's why cross-attention K/V is computed outside the decode loop in encoder-decoder models.

**Solution outline:** the shared core with `Lq ≠ Lk` and packed-head strides.
