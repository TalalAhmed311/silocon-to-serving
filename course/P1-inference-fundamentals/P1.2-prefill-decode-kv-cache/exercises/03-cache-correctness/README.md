# Exercise 3 — KV cache correctness (medium)

Implement `PrefillThenDecode` in `cached.py`. It is a NumPy model with:

- `prefill(tokens) -> logits_last`, which processes the whole prompt **in one batched pass**: `[T, d]` matmuls and a causal attention mask. It fills the KV cache for positions `0..T-1`.
- `decode(token) -> logits`, which processes one new token at position `T`, `T+1`, …, reading the cache.

Reuse the helpers in `platform/engine/v0/reference/llama_numpy.py` (`rmsnorm`, `rope_tables`, `apply_rope`, `softmax`, `silu`) and its weight dict. The difference from the reference is that prefill is a **real batched prefill** (a GEMM), not a loop of decodes.

**Test:** for a tiny random model, `prefill(prompt)` gives the same logits as the reference's last step, and every subsequent `decode` matches the reference step by step (`atol=1e-4`). An extra check: no future-token leakage. Changing a *later* prompt token must not change the logits computed for earlier positions.
