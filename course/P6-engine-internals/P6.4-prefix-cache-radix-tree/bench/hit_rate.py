"""Prefix-hit rate: block hashing (P6.3) vs radix tree (P6.4) on two synthetic traces. T0.

Run: uv run python course/P6-engine-internals/P6.4-prefix-cache-radix-tree/bench/hit_rate.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "platform" / "engine" / "v1"))
from s2s_engine.block_manager import BlockManager  # noqa: E402
from s2s_engine.radix_cache import RadixCache  # noqa: E402
from s2s_engine.sequence import Sequence  # noqa: E402

BS = 16


def multi_turn(rng, convs=20, turns=6):
    out = []
    for _ in range(convs):
        hist = [rng.randint(0, 999) for _ in range(rng.randint(20, 60))]
        for _ in range(turns):
            hist = hist + [rng.randint(0, 999) for _ in range(rng.randint(5, 40))]       # new user message
            out.append(list(hist))
            hist = hist + [rng.randint(0, 999) for _ in range(rng.randint(10, 80))]      # the answer
    return out


def few_shot(rng, n=200):
    header = [rng.randint(0, 999) for _ in range(rng.randint(30, 50))]
    return [header + [rng.randint(0, 999) for _ in range(rng.randint(3, 20))] for _ in range(n)]


def block_hash_hits(prompts):
    bm = BlockManager(num_blocks=1 << 16, block_size=BS)
    hit = 0
    for p in prompts:
        s = Sequence(p)
        hit += bm.match_prefix(s)
        bm.allocate(s, len(p) - s.num_computed)
        s.num_computed = len(p)
        bm.commit(s)
        bm.free_seq(s)
    return hit


def radix_hits(prompts):
    rc, hit, slot = RadixCache(), 0, 0
    for p in prompts:
        n, _, _ = rc.match_prefix(p)
        hit += min(n, len(p) - 1)                       # same rule: the last prompt token is always computed
        rc.insert(p, list(range(slot, slot + len(p))))
        slot += len(p)
    return hit


rng = random.Random(0)
print(f"trace       | prompt tokens | block-hash (bs={BS}) | radix")
for name, prompts in (("multi-turn", multi_turn(rng)), ("few-shot", few_shot(rng))):
    total = sum(len(p) for p in prompts)
    print(f"{name:11s} | {total:13d} | {block_hash_hits(prompts) / total:19.3f} | {radix_hits(prompts) / total:.3f}")
