"""Your ring collectives on simulated ranks (lists of NumPy arrays). See 01-ring.md.

Every function returns (result buffers, number of communication steps). In one step, every rank sends one message to
its right neighbour (r → r+1 mod n) at the same time.
"""
from __future__ import annotations

import numpy as np


def reduce_scatter(bufs: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    """After: rank r holds the full sum of chunk r of np.array_split(buf, n) (other chunks: anything). TODO."""
    raise NotImplementedError


def all_gather(chunks: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    """Input: rank r holds chunk r. After: every rank holds the concatenation of all chunks in rank order. TODO."""
    raise NotImplementedError


def all_reduce(bufs: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    """Ring all-reduce = reduce_scatter then all_gather. Must take 2(n-1) steps. TODO."""
    raise NotImplementedError
