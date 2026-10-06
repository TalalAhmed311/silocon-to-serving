# Exercise 1 — KV size function (easy)

Implement in `kv.py`:

```python
kv_bytes(layers, kv_heads, head_dim, tokens, bytes_per_elem=2) -> int
max_tokens(budget_bytes, layers, kv_heads, head_dim, bytes_per_elem=2) -> int    # floor
max_sequences(budget_bytes, seq_len, layers, kv_heads, head_dim, bytes_per_elem=2) -> int
```

**Test:** hand-computed cases: an MHA 7B shape, a GQA 8B shape, an FP8 KV cache, zero tokens, and a budget smaller than one token.
