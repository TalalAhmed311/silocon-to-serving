"""Exercise 2: a numerically stable softmax."""
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Turn scores into probabilities along `axis`: exp(x) / sum(exp(x)).

    Must not overflow for large inputs such as [1000, 1001].
    """
    # TODO: subtract the max along `axis` (keepdims=True) first, then exp, then divide by the sum.
    raise NotImplementedError


def naive_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """The textbook formula with no max subtraction. Given for comparison: do not change it."""
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
