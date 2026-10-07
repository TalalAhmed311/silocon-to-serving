"""Exercise 2 starter."""
from typing import Iterable


def count_faults(addresses: Iterable[int], page: int = 4096, around: int = 1) -> int:
    """Number of page faults when touching `addresses` in order on a fresh mapping.

    around: pages mapped per fault, as an aligned group (1 = no fault-around).
    """
    raise NotImplementedError  # TODO: track which pages are mapped
