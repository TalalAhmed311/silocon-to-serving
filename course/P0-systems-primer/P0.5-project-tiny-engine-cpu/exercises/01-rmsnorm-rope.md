# Exercise 1 — RMSNorm and RoPE (easy)

Implement `ops::rmsnorm` and `ops::rope` in [`ops/ops.hpp`](ops/ops.hpp).

- `rmsnorm(out, x, w, n, eps)`: `out[i] = x[i] / sqrt(mean(x²) + eps) * w[i]`. It must work **in place** (`out == x`), because the engine's final norm calls it that way.
- `rope(v, n_heads, head_dim, pos, theta)`: rotate_half RoPE, in place. For `i < head_dim/2`, with `angle = pos · theta^(−2i/head_dim)`:
  - `x[i] ← x[i]·cos − x[i+h]·sin`
  - `x[i+h] ← x[i+h]·cos + x[i]·sin`

  where `h = head_dim/2`.

**Hints**

1. Compute `inv = 1/sqrt(...)` once, then multiply. Don't divide n times.
2. In `rope`, save `x[i]` and `x[i+h]` in locals before writing either one.
3. Compute the angle in `double`. At pos = 100k, float loses the low bits of `pos · freq`.

**Test:** `ctest -R 01` (fixtures from `make_fixtures.py`: 4 rows × 64; RoPE at positions 0, 1, 7, 100; `atol=1e-5`).
