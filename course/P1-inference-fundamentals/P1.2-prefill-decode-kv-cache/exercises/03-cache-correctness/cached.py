"""Exercise 3 starter."""
import numpy as np


class PrefillThenDecode:
    def __init__(self, model_dir):
        raise NotImplementedError  # TODO: load config + weights (see llama_numpy.LlamaNumpy.__init__)

    def prefill(self, tokens: list[int]) -> np.ndarray:
        raise NotImplementedError  # TODO: batched [T, d] forward with a causal mask; fill the cache; return last logits

    def prefill_all_logits(self, tokens: list[int]) -> np.ndarray:
        raise NotImplementedError  # TODO: like prefill, but return [T, V] (used by the leakage test)

    def decode(self, token: int) -> np.ndarray:
        raise NotImplementedError  # TODO
