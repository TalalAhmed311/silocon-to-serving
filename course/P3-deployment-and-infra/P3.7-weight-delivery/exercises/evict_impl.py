"""Your eviction planner. Contract: platform/weights/cache.py:plan_evictions (see 02-lru.md)."""
from __future__ import annotations

from weights.cache import CacheFull  # noqa: F401  (raise this)


def plan_evictions(lru: dict[str, float], sizes: dict[str, int], pinned: set[str], need: int, max_bytes: int) -> list[str]:
    return []  # TODO
