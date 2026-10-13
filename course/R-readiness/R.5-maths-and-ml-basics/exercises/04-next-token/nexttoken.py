"""Exercise 4: from logits to a next token."""
import numpy as np


def greedy(logits: np.ndarray, vocab: list[str]) -> str:
    """The token with the highest logit. On a tie, the lowest index wins."""
    # TODO
    raise NotImplementedError


def top_k(logits: np.ndarray, vocab: list[str], k: int, temperature: float = 1.0) -> list[tuple[str, float]]:
    """The k most likely tokens as (token, probability), best first.

    Probabilities are softmax(logits / temperature) over the WHOLE vocabulary
    (do not renormalise over the k). temperature must be > 0, else raise ValueError.
    """
    # TODO: divide by temperature, stable softmax, sort by probability (stable, descending), take k.
    raise NotImplementedError
