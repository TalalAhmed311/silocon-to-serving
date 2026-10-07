"""Your radix prefix cache. Same interface as platform/engine/v1/s2s_engine/radix_cache.py (README §1)."""
from __future__ import annotations

import itertools

_clock = itertools.count()


class Node:
    __slots__ = ("children", "parent", "key", "value", "ref", "last_access")

    def __init__(self, parent=None, key: tuple = (), value: tuple = ()):
        self.children: dict = {}
        self.parent = parent
        self.key = key
        self.value = value
        self.ref = 0
        self.last_access = next(_clock)


class RadixCache:
    def __init__(self):
        self.root = Node()
        self.root.ref = 1
        self.size = 0          # cached tokens
        self.locked = 0        # tokens on nodes with ref > 0 (excluding root)

    def match_prefix(self, tokens):
        """TODO (ex. 1): → (matched length, list of slots, last matched node). Split an edge on a partial match.
        Update last_access on every node you touch."""
        raise NotImplementedError

    def insert(self, tokens, values) -> int:
        """TODO (ex. 1): add tokens with their slots; return the length that was already cached."""
        raise NotImplementedError

    def lock(self, node) -> None:
        """TODO (ex. 2): ref += 1 from node up to (not including) the root; maintain self.locked."""
        raise NotImplementedError

    def unlock(self, node) -> None:
        """TODO (ex. 2)."""
        raise NotImplementedError

    def evict(self, num_tokens: int) -> list:
        """TODO (ex. 2): free ≥ num_tokens from LRU unlocked leaves; return the freed slots."""
        raise NotImplementedError

    def evictable(self) -> int:
        return self.size - self.locked
