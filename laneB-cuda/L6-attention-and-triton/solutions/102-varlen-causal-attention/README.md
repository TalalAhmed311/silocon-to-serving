# 102: Variable-Length Causal Attention (L6, core)

Problem: [LeetGPU #102](../../leetgpu-map.md). A batch of sequences of different lengths, **packed** without padding (`cu_seqlens` offsets).

**Hint ladder**

1. Padding to the max length wastes compute and memory on short sequences. Pack them along one token axis and pass prefix offsets `cu_seqlens[B+1]`.
2. The grid is (tiles of the **longest** sequence, B·H). Each block reads its sequence's offset and length, and exits at once if its tile is past that sequence's end.
3. Attention never crosses sequence boundaries, so offsets select the K/V range too.
4. This is the layout continuous batching produces (P2.3, P6.2), and FlashAttention's `varlen` API takes exactly these offsets.

**Solution outline:** the shared core with `cu_q`, packed `[T, H, d]` strides and `causal = true`.
