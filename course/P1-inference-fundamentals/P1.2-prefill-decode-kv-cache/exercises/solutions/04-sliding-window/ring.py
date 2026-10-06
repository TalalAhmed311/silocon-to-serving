"""Exercise 4 solution."""
import numpy as np


class RingKV:
    def __init__(self, window: int, layers: int, kv_dim: int):
        self.W = window
        self.K = np.zeros((layers, window, kv_dim), np.float32)   # O(W), never grows
        self.V = np.zeros_like(self.K)

    def write(self, layer, pos, k, v):
        self.K[layer, pos % self.W] = k
        self.V[layer, pos % self.W] = v

    def read(self, layer, pos):
        positions = np.arange(max(0, pos - self.W + 1), pos + 1)
        slots = positions % self.W            # gather in position order, undoing the wrap-around
        return self.K[layer, slots], self.V[layer, slots], positions

    def nbytes(self):
        return self.K.nbytes + self.V.nbytes


def windowed_attention(q, K, V, n_heads, n_kv_heads, hd):
    group, n = n_heads // n_kv_heads, K.shape[0]
    Kh, Vh = K.reshape(n, n_kv_heads, hd), V.reshape(n, n_kv_heads, hd)
    out = np.empty((n_heads, hd), np.float32)
    for h in range(n_heads):
        s = Kh[:, h // group] @ q[h] / np.sqrt(hd)
        e = np.exp(s - s.max())
        out[h] = (e / e.sum()) @ Vh[:, h // group]
    return out
