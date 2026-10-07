"""Exercise 1 solution."""


def kv_bytes(layers: int, kv_heads: int, head_dim: int, tokens: int, bytes_per_elem: int = 2) -> int:
    return 2 * layers * kv_heads * head_dim * bytes_per_elem * tokens  # 2 = K and V


def max_tokens(budget_bytes: int, layers: int, kv_heads: int, head_dim: int, bytes_per_elem: int = 2) -> int:
    return budget_bytes // kv_bytes(layers, kv_heads, head_dim, 1, bytes_per_elem)


def max_sequences(budget_bytes: int, seq_len: int, layers: int, kv_heads: int, head_dim: int, bytes_per_elem: int = 2) -> int:
    return budget_bytes // kv_bytes(layers, kv_heads, head_dim, seq_len, bytes_per_elem)
