"""Exercise 3 (solution): a two-layer MLP forward pass with SiLU."""
import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / (1.0 + np.exp(-x))


def mlp_forward(x, W1, b1, W2, b2):
    hidden = silu(W1 @ x + b1)
    return W2 @ hidden + b2


def param_count(d_in: int, d_hidden: int, d_out: int) -> int:
    return d_hidden * d_in + d_hidden + d_out * d_hidden + d_out
