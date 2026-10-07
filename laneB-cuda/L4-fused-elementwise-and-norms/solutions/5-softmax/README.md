# 5: Softmax (L4, core, exit check)

Problem: [LeetGPU #5](../../leetgpu-map.md). Numerically stable softmax. Our harness does it row-wise.

**Hint ladder**

1. Subtract the row max before `exp`, or large logits overflow. That gives the 3-pass version: max, sum, normalize.
2. **Online softmax:** keep `(m, d)` together. On a new value `v > m`, `d ← d·e^{m−v} + 1`. Otherwise `d += e^{v−m}`. One read gives both.
3. Merge per-thread states across the block: take the block max `M`, rescale each thread's `d` by `e^{m−M}`, then block-sum.
4. If the row fits in registers (≤ ~1024 floats per warp), keep it there, so the normalizing pass doesn't reread HBM (P5.5 v3).

**Solution outline:** one block per row: an online loop → `block_max` → rescaled `block_sum` → a normalizing loop.

**Why this is fast:** 2 reads + 1 write instead of 3 + 1, and the reductions are shuffles. Report **GB/s against copy**. That's the L4 exit check (`--bench`, rows of 4k–32k). `TODO(run-on: g4dn.xlarge)`
