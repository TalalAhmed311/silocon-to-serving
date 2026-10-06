# Exercise 4 — RoPE's relative-position property (hard)

RoPE rotates each (i, i + h/2) pair of q at position m by angle `m·ωᵢ`, and of k at position n by `n·ωᵢ`. Because rotations compose, `⟨R(m)q, R(n)k⟩ = ⟨q, R(n − m)k⟩`. The attention score depends only on the **offset** `n − m`, not on absolute positions.

Implement `score(q, k, m, n, theta=10000.0) -> float` in `rope_rel.py`, applying RoPE in the rotate_half convention (reuse `platform/engine/v0/reference/llama_numpy.py`). Then the test checks:

1. `score(q, k, m, n) == score(q, k, m + s, n + s)` for random shifts `s` (`rtol = 1e-4` in float64)
2. it is *not* invariant to changing only one of the two positions
3. **(the derivation)** your `score_via_offset(q, k, offset)`, which never uses `m` or `n`, matches `score` for all (m, n) with the same offset

**Hint:** write the rotation of one pair as a 2×2 matrix. What is `R(a)ᵀ R(b)`?
