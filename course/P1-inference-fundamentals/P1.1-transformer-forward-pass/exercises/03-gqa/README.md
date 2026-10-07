# Exercise 3 — GQA: how much KV cache does it save? (easy)

Implement in `gqa.py`:

- `kv_bytes_per_token(cfg, bytes_per_elem=2) -> int`: K and V for one token across all layers.
- `kv_saving_vs_mha(cfg) -> float`: the factor relative to the same model with `num_key_value_heads = num_attention_heads`.

**Test:** MHA (H_kv = H), GQA (H_kv = H/4) and MQA (H_kv = 1) on a 32-head, 4096-d, 32-layer config. For example MHA fp16: `2 · 32 · 32 · 128 · 2 = 524,288` bytes = 0.5 MiB per token.

**Think:** at 0.5 MiB/token, how many tokens of KV cache fit in the ~8 GB left on a 24 GB GPU after a 16 GB model? And with 8 KV heads? (P1.2 makes this the capacity calculation.)
