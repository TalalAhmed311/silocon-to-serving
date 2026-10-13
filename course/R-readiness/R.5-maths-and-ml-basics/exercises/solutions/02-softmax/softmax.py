"""Exercise 2 (solution): a numerically stable softmax."""
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def naive_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
