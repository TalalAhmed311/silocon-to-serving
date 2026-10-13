"""Exercise 3: a two-layer MLP forward pass with SiLU."""
import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))."""
    # TODO
    raise NotImplementedError


def mlp_forward(x, W1, b1, W2, b2):
    """y = W2 @ silu(W1 @ x + b1) + b2.

    x: (d_in,)  W1: (d_hidden, d_in)  b1: (d_hidden,)  W2: (d_out, d_hidden)  b2: (d_out,)
    """
    # TODO: hidden = silu(W1 @ x + b1); return W2 @ hidden + b2
    raise NotImplementedError


def param_count(d_in: int, d_hidden: int, d_out: int) -> int:
    """Number of learnable numbers in W1, b1, W2 and b2."""
    # TODO
    raise NotImplementedError
