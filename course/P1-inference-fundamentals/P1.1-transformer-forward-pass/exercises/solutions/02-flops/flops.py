"""Exercise 2 solution."""


def decode_flops(cfg: dict, t: int) -> int:
    d, L, V = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    H = cfg["num_attention_heads"]
    Hkv = cfg.get("num_key_value_heads", H)
    h, f = d // H, cfg["intermediate_size"]
    matrices_per_layer = 2 * d * H * h + 2 * d * Hkv * h + 3 * d * f
    attention_per_layer = 4 * H * h * t  # q·K over t keys + softmax·V over t values, per query head
    return L * (2 * matrices_per_layer + attention_per_layer) + 2 * V * d  # + LM head (tied or not, it's a matvec)
