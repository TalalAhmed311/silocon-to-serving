import os

import pytest

from s2s.exercise import load_impl
from weights import cache as ref

if os.environ.get("S2S_SOLUTIONS") == "1":
    plan = ref.plan_evictions
else:
    plan = load_impl(__file__, "evict_impl").plan_evictions

LRU = {"a": 1.0, "b": 2.0, "c": 3.0}        # a is oldest
SIZES = {"a": 40, "b": 30, "c": 20}          # used = 90


def test_fits_without_eviction():
    assert plan(LRU, SIZES, set(), need=10, max_bytes=100) == []


def test_evicts_oldest_first_and_stops_early():
    assert plan(LRU, SIZES, set(), need=40, max_bytes=100) == ["a"]
    assert plan(LRU, SIZES, set(), need=75, max_bytes=100) == ["a", "b"]


def test_skips_pinned():
    assert plan(LRU, SIZES, {"a"}, need=40, max_bytes=100) == ["b"]


def test_too_big_raises():
    with pytest.raises(ref.CacheFull):
        plan(LRU, SIZES, set(), need=101, max_bytes=100)


def test_pins_block_raises():
    with pytest.raises(ref.CacheFull):
        plan(LRU, SIZES, {"a", "b"}, need=70, max_bytes=100)
