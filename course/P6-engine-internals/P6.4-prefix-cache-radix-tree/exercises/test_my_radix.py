"""Tests for exercises/my_radix.py (S2S_SOLUTIONS=1: against the reference). T0."""
import importlib.util
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "platform" / "engine" / "v1"))
from s2s_engine.radix_cache import NaiveTrie  # noqa: E402

if os.environ.get("S2S_SOLUTIONS") == "1":
    from s2s_engine.radix_cache import RadixCache  # noqa: E402
else:
    _spec = importlib.util.spec_from_file_location("my_radix", Path(__file__).with_name("my_radix.py"))
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    RadixCache = _m.RadixCache


def check_tree(rc: RadixCache):
    """Structural invariants: edges non-empty, keys/values aligned, child map keyed by first token, size adds up."""
    total, stack = 0, [rc.root]
    while stack:
        n = stack.pop()
        for first, ch in n.children.items():
            assert ch.parent is n and ch.key and ch.key[0] == first and len(ch.key) == len(ch.value)
            total += len(ch.key)
            stack.append(ch)
    assert total == rc.size


def test_match_agrees_with_naive_trie():
    rng = random.Random(1)
    for _ in range(50):
        rc, naive, slot_of = RadixCache(), NaiveTrie(), {}
        for _ in range(40):
            toks = [rng.randint(0, 3) for _ in range(rng.randint(1, 12))]
            if rng.random() < 0.5:
                vals = [hash((tuple(toks[:i + 1]),)) for i in range(len(toks))]   # slot ≡ identity of the prefix
                already = rc.insert(toks, vals)
                assert already == naive.match_len(toks)
                naive.insert(toks)
                for i in range(len(toks)):
                    slot_of[tuple(toks[:i + 1])] = vals[i]
            else:
                n, slots, _ = rc.match_prefix(toks)
                assert n == naive.match_len(toks)
                assert slots == [slot_of[tuple(toks[:i + 1])] for i in range(n)]
            check_tree(rc)


def test_split_on_partial_match():
    rc = RadixCache()
    rc.insert([1, 2, 3, 4], [10, 20, 30, 40])
    n, slots, node = rc.match_prefix([1, 2, 9])
    assert (n, slots) == (2, [10, 20]) and node.key == (1, 2)
    assert rc.insert([1, 2, 9], [10, 20, 99]) == 2
    check_tree(rc)


def test_eviction_is_lru_and_respects_locks():
    rc = RadixCache()
    rc.insert([1, 1, 1], [1, 2, 3])
    rc.insert([2, 2, 2], [4, 5, 6])
    rc.insert([3, 3, 3], [7, 8, 9])
    _, _, hot = rc.match_prefix([1, 1, 1])              # touch: [1,1,1] is now most recent
    _, _, locked = rc.match_prefix([3, 3, 3])
    rc.lock(locked)
    assert rc.evictable() == 6
    freed = rc.evict(1)
    assert freed == [4, 5, 6]                          # [2,2,2] is the LRU unlocked leaf
    freed = rc.evict(100)
    assert freed == [1, 2, 3]                          # the locked branch survives
    assert rc.size == 3
    rc.unlock(locked)
    assert rc.evict(100) == [7, 8, 9] and rc.size == 0


def test_evicting_children_exposes_parent():
    rc = RadixCache()
    rc.insert([1, 2, 3], [1, 2, 3])
    rc.insert([1, 2, 4], [1, 2, 5])                    # shared [1,2] edge with two leaves
    freed = rc.evict(3)
    assert sorted(freed) == [1, 2, 3, 5] and rc.size == 0
