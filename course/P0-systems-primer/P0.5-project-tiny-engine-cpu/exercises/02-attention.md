# Exercise 2 — Attention with a KV cache, GQA (medium)

Implement `ops::attention(q, k_cache, v_cache, out, pos, n_heads, n_kv_heads, hd, scratch)`.

- `q` is `[n_heads, hd]` for the current position.
- `k_cache` and `v_cache` are `[max_seq, n_kv_heads·hd]` for one layer. Rows `0..pos` are valid, **including** `pos`, which the engine wrote just before calling you.
- Query head `h` uses KV head `h / (n_heads / n_kv_heads)`.
- `out[h]` = Σₜ softmax(q_h · k_t / √hd)ₜ · v_t.

**Hints**

1. Fill `scratch[0..pos]` with the scaled scores, then call `ops::softmax(scratch, pos + 1)`.
2. Zero `out[h]` before accumulating.
3. The offset of KV head `j` at position `t` is `t · kv_dim + j · hd`.

**Test:** `ctest -R 02`. It checks every position 0..9 against NumPy (4 query heads, 2 KV heads, hd 16), plus a 1-key sanity case.
