"""Exercise 2 solution."""
from typing import Iterable


def count_faults(addresses: Iterable[int], page: int = 4096, around: int = 1) -> int:
    mapped: set[int] = set()
    faults = 0
    for a in addresses:
        p = a // page
        if p in mapped:
            continue
        faults += 1
        group = p - p % around          # fault-around maps the aligned group containing p
        mapped.update(range(group, group + around))
    return faults
