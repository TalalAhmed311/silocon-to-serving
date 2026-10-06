# 70: Segmented Exclusive Prefix Sum (L3, practice)

Problem: [LeetGPU #70](../../leetgpu-map.md). A prefix sum that restarts at every segment head.

**Hint ladder**

1. A segmented scan is an ordinary scan over **pairs** `(flag, value)` with the operator `(a ⊕ b) = (a.f | b.f, b.f ? b.v : a.v + b.v)`. Check for yourself that it's associative.
2. So reuse the #16 machinery unchanged with that operator. The carry from the previous tile is cut off automatically when the tile contains a head.
3. Exclusive from inclusive: `out[i] = 0` at a head, else `incl[i − 1].v`. Don't compute `incl[i] − v[i]`, which loses precision for floats.

**Solution outline:** `pack` (values + flags → pairs) → the generic `inclusive_scan` with `SegAdd` → `to_exclusive`.

**Why this is fast:** it's the same memory-bound tile scan as #16, with 8-byte elements instead of 4. A segmented scan underlies sparse kernels (CSR row sums), ragged batches, and per-sequence operations in serving.
