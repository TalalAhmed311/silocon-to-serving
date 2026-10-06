"""Exercise 3 solution."""


def kv_bytes_per_token(cfg: dict, bytes_per_elem: int = 2) -> int:
    H = cfg["num_attention_heads"]
    h = cfg["hidden_size"] // H
    Hkv = cfg.get("num_key_value_heads", H)
    return 2 * cfg["num_hidden_layers"] * Hkv * h * bytes_per_elem  # 2 = K and V


def kv_saving_vs_mha(cfg: dict) -> float:
    return cfg["num_attention_heads"] / cfg.get("num_key_value_heads", cfg["num_attention_heads"])
