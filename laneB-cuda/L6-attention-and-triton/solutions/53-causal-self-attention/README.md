# 53: Causal Self-Attention (L6, core)

Problem: [LeetGPU #53](../../leetgpu-map.md).

**Hint ladder**

1. Mask `j > i` to −∞ **before** the max, so masked keys don't affect m.
2. **Skip** K/V tiles that are entirely in the future of the whole query tile. The loop bound becomes `i0 + tile + 1`, which is about half the work.
3. Only the diagonal tile needs per-element masking.

**Solution outline:** the shared core with `causal = true` (its loop bound implements the skip).
