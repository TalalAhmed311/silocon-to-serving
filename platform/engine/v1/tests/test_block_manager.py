"""Block manager properties (P6.3 exercises 1–3). T0."""
import random

import pytest

from s2s_engine.block_manager import BlockManager, NoFreeBlocks
from s2s_engine.sequence import Sequence


def grow(bm, seq, n):
    bm.allocate(seq, n)
    seq.num_computed += n
    bm.commit(seq)


def test_random_ops_keep_invariants():
    rng = random.Random(0)
    for trial in range(30):
        bm = BlockManager(num_blocks=rng.randint(4, 40), block_size=rng.choice([1, 2, 4, 8]))
        live = []
        for _ in range(200):
            op = rng.random()
            if op < 0.4 and len(live) < 8:
                s = Sequence([rng.randint(0, 3) for _ in range(rng.randint(1, 20))])
                bm.match_prefix(s)
                n = len(s.prompt) - s.num_computed
                if bm.can_allocate(s, n):
                    grow(bm, s, n)
                    live.append(s)
                else:
                    bm.free_seq(s)
            elif op < 0.7 and live:
                s = rng.choice(live)
                if bm.can_allocate(s, 1):
                    s.output.append(rng.randint(0, 3))
                    grow(bm, s, 1)
            elif op < 0.8 and live:
                parent = rng.choice(live)
                child = Sequence(list(parent.prompt))
                child.output = list(parent.output)
                bm.fork(parent, child)
                live.append(child)
            elif live:
                s = live.pop(rng.randrange(len(live)))
                bm.free_seq(s)
            bm.take_copy_ops()
            bm.check_invariants(live)


def test_slot_mapping():
    bm = BlockManager(8, 4)
    s = Sequence(list(range(10)))
    grow(bm, s, 10)
    assert len(s.block_table) == 3
    for pos in range(10):
        assert bm.slot(s, pos) == s.block_table[pos // 4] * 4 + pos % 4


def test_prefix_reuse_full_blocks_only_and_leaves_one_token():
    bm = BlockManager(16, 4)
    a = Sequence(list(range(12)))
    grow(bm, a, 12)
    bm.free_seq(a)                                  # blocks go to cached_free, still matchable
    b = Sequence(list(range(12)))
    assert bm.match_prefix(b) == 8                  # 12 tokens = 3 full blocks, but the last token must be computed
    c = Sequence(list(range(6)) + [99, 99, 99])
    assert bm.match_prefix(c) == 4                  # only the first full block matches
    d = Sequence([1] + list(range(1, 12)))
    assert bm.match_prefix(d) == 0                  # chained hash: block 2 never matches if block 1 differs
    bm.check_invariants([b, c, d])


def test_cached_blocks_are_evicted_lru_when_needed():
    bm = BlockManager(4, 2)
    a = Sequence([1, 2, 3, 4])
    grow(bm, a, 4)
    bm.free_seq(a)
    assert bm.num_free == 4 and len(bm.cached_free) == 2
    big = Sequence(list(range(100, 108)))
    grow(bm, big, 8)                               # needs all 4 blocks → evicts the cached ones
    assert bm.stats["evicted_cached"] == 2
    again = Sequence([1, 2, 3, 4, 5])
    bm.free_seq(big)
    assert bm.match_prefix(again) == 0            # the old prefix is gone
    bm.check_invariants([again])


def test_copy_on_write_after_fork():
    bm = BlockManager(8, 4)
    p = Sequence([1, 2, 3, 4, 5, 6])              # 1.5 blocks: last block partial
    grow(bm, p, 6)
    c = Sequence(list(p.prompt))
    bm.fork(p, c)
    shared_last = p.block_table[-1]
    assert bm.ref[shared_last] == 2
    c.output.append(7)
    grow(bm, c, 1)                                 # child writes into the partial block → copy first
    ops = bm.take_copy_ops()
    assert ops == [(shared_last, c.block_table[-1])]
    assert c.block_table[0] == p.block_table[0]    # full block stays shared
    assert bm.ref[shared_last] == 1
    bm.check_invariants([p, c])


def test_no_free_blocks_raises():
    bm = BlockManager(2, 2)
    s = Sequence([1, 2, 3, 4, 5])
    with pytest.raises(NoFreeBlocks):
        bm.allocate(s, 5)
