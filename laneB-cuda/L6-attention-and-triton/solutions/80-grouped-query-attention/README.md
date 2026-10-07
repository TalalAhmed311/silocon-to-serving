# 80: Grouped Query Attention (L6, core)

Problem: [LeetGPU #80](../../leetgpu-map.md). H query heads share Hkv < H key/value heads.

**Hint ladder**

1. Query head h reads KV head `h / (H / Hkv)`. That's the whole algorithm change. Hkv = 1 is MQA, and Hkv = H is MHA (the test covers both).
2. The win is memory: the KV cache shrinks by H/Hkv. A kernel that processes a whole **group** of query heads per block reads each K/V tile once instead of H/Hkv times. That's the next optimization: try it.

**Solution outline:** the shared core with the head mapping, one query head per block.
