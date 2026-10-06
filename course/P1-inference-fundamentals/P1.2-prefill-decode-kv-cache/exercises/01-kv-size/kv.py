"""Exercise 1 starter."""


def kv_bytes(layers: int, kv_heads: int, head_dim: int, tokens: int, bytes_per_elem: int = 2) -> int:
    raise NotImplementedError


def max_tokens(budget_bytes: int, layers: int, kv_heads: int, head_dim: int, bytes_per_elem: int = 2) -> int:
    raise NotImplementedError


def max_sequences(budget_bytes: int, seq_len: int, layers: int, kv_heads: int, head_dim: int, bytes_per_elem: int = 2) -> int:
    raise NotImplementedError
