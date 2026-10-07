# 119: Block-Sparse KV Selection Attention (L6, stretch: hints and outline only)

Problem: [LeetGPU #119](../../leetgpu-map.md). Each query block attends only to a selected subset of KV blocks (a list or bitmask per query block).

**Hint ladder**

1. Start from the FA-2 core. Instead of looping over all K/V tiles, loop over the **selected block indices** for this query tile (read them from a per-tile index list).
2. The online softmax doesn't care which tiles you visit or in what order. Unvisited tiles are simply −∞.
3. The selection itself (top-k blocks by a cheap score, e.g. mean-pooled keys) is a separate small kernel: a top-k per query block (L3 #29).

**Solution outline:** `for idx in selected[q_tile]: load K/V tile idx → same update as FA-2`. The cost is proportional to the number of selected blocks.
