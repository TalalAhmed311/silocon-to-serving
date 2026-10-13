"""Exercise 4 (solution): from logits to a next token."""
import numpy as np


def greedy(logits: np.ndarray, vocab: list[str]) -> str:
    return vocab[int(np.argmax(logits))]       # argmax returns the first maximum


def top_k(logits: np.ndarray, vocab: list[str], k: int, temperature: float = 1.0) -> list[tuple[str, float]]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    z = np.asarray(logits, dtype=float) / temperature
    e = np.exp(z - z.max())
    p = e / e.sum()
    order = np.argsort(-p, kind="stable")[:k]
    return [(vocab[i], float(p[i])) for i in order]
