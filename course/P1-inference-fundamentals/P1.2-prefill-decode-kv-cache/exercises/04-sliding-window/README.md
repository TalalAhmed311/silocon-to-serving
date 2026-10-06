# Exercise 4 — Sliding-window KV (hard)

Some models (Mistral-style) attend only to the last `W` tokens. The cache then needs only `W` slots, used as a **ring buffer**, so memory stays O(W) instead of O(t).

Implement `RingKV(window, layers, kv_dim)` in `ring.py`:
- `write(layer, pos, k, v)` stores into slot `pos % window`
- `read(layer, pos) -> (K, V, positions)` returns the keys and values of positions `max(0, pos − W + 1) .. pos`, **in position order**, together with their absolute positions

Then implement `windowed_attention(q, K, V, n_heads, n_kv_heads, hd)` for one position.

**Tests:**
1. memory stays `window × kv_dim` per layer however long you decode
2. for `pos < W`, the windowed attention equals full attention exactly
3. `read` returns the right positions in order after wrap-around

**Report (not tested):** on the tiny model, run full-attention vs window = 4/8/16 decoding for 64 steps from the same prompt, and tabulate `window | KV bytes | mean |Δlogit| vs full`. Explain why a *random* model is a bad proxy for the quality impact on a real one.
