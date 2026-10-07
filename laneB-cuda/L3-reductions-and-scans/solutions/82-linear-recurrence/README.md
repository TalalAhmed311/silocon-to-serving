# 82: Linear Recurrence (L3, stretch)

Problem: [LeetGPU #82](../../leetgpu-map.md). `x[t] = a[t]·x[t−1] + b[t]`, which looks inherently sequential.

**Hint ladder**

1. Each step is an **affine map** `x ↦ a·x + b`. Composing two affine maps gives another affine map, and composition is associative.
2. So the prefix compositions are a **scan** with `compose(f, g) = (g.a·f.a, g.a·f.b + g.b)` and identity `(1, 0)`.
3. Applied to `x[−1] = 0`, the prefix map at t gives `x[t] = B_t`. Reuse the #16 scan with this operator. Mind the order: `op(earlier, later)`.

**Solution outline:** `pack` → generic `inclusive_scan` with `Compose` → `unpack` (take `.b`).

**Why this matters:** this is exactly how linear RNNs and state-space models (S4/Mamba-style "parallel scan") train in O(log n) depth, and it's #110's GAE in disguise. Numerics: products of many `a` shrink (`|a| < 1`) or blow up (`|a| > 1`), so the test uses stable coefficients and a relative tolerance.
