"""Exercise 1 solution."""


def count_params(cfg: dict) -> int:
    d, L, V = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    H = cfg["num_attention_heads"]
    Hkv = cfg.get("num_key_value_heads", H)
    h, f = d // H, cfg["intermediate_size"]
    per_layer = 2 * d * H * h + 2 * d * Hkv * h + 3 * d * f + 2 * d  # q,o + k,v + gate,up,down + 2 norms
    head = 0 if cfg.get("tie_word_embeddings", False) else V * d
    return L * per_layer + V * d + head + d
