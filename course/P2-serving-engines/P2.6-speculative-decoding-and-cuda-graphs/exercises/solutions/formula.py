"""Exercise 2 solution."""
import numpy as np


def simulate(alpha, k, rounds=20000, seed=0):
    rng, total = np.random.default_rng(seed), 0
    for _ in range(rounds):
        a = 0
        while a < k and rng.random() < alpha:
            a += 1
        total += a + 1          # accepted + (correction or bonus)
    return total / rounds
