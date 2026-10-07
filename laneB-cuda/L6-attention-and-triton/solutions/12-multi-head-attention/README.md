# 12: Multi-Head Attention (L6, core)

Problem: [LeetGPU #12](../../leetgpu-map.md). h independent heads packed in the model dimension.

**Hint ladder**

1. Don't transpose `[N, h, d_h]` to `[h, N, d_h]` with a separate kernel. Put heads in the grid (`blockIdx.y`) and address with strides: token stride `d_model`, head stride `d_h`.
2. Everything else is #6 per head.

**Solution outline:** the shared core with `qi = d_model, qh = d_h`. The output is written in the same packed layout, ready for the output projection.
