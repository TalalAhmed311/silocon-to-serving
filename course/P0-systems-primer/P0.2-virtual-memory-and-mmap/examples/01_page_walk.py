"""01_page_walk.py — simulate x86-64 4-level address translation with a small TLB.

Run:      uv run python examples/01_page_walk.py
Expected: first, 3 virtual addresses split into PML4/PDPT/PD/PT indices + offset (check one by hand),
          then hit rates for sequential, strided-4KiB and random access through a 64-entry LRU TLB:
          sequential ≈ 99.9%, stride-4KiB = 0% after warm-up, random-over-1GiB ≈ 0%.
Hardware: T0. Deterministic (seed 0).
"""
from __future__ import annotations

import random
from collections import OrderedDict

PAGE = 4096


def split(va: int) -> dict[str, int]:
    # 9 bits per level because each table is one 4 KiB page of 512 8-byte entries.
    return {
        "PML4": (va >> 39) & 0x1FF,
        "PDPT": (va >> 30) & 0x1FF,
        "PD": (va >> 21) & 0x1FF,
        "PT": (va >> 12) & 0x1FF,
        "offset": va & 0xFFF,
    }


class TLB:
    """Fully associative LRU TLB. Real TLBs are set-associative and multi-level; LRU is enough to see the effect."""

    def __init__(self, entries: int):
        self.entries, self.map = entries, OrderedDict()
        self.hits = self.misses = 0

    def lookup(self, va: int, page: int = PAGE) -> bool:
        vpn = va // page
        if vpn in self.map:
            self.map.move_to_end(vpn)
            self.hits += 1
            return True
        self.misses += 1          # a miss = a page walk: up to 4 dependent memory reads
        self.map[vpn] = True
        if len(self.map) > self.entries:
            self.map.popitem(last=False)
        return False

    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, self.hits + self.misses)


def run(pattern: str, n: int = 100_000, entries: int = 64, page: int = PAGE) -> float:
    rng = random.Random(0)
    tlb = TLB(entries)
    base = 0x7F0000000000
    for k in range(n):
        if pattern == "sequential":
            va = base + 4 * k                      # one float after another
        elif pattern == "stride-4KiB":
            va = base + PAGE * (k % 10_000)        # one access per page, 10k pages > 64 entries
        else:
            va = base + rng.randrange(1 << 30)     # anywhere in 1 GiB
        tlb.lookup(va, page)
    return tlb.hit_rate


if __name__ == "__main__":
    for va in (0x00007F3A12345678, 0x0000000000401000, 0x00007FFFFFFFF000):
        s = split(va)
        print(f"{va:#018x} -> " + " ".join(f"{k}={v:#x}" if k == "offset" else f"{k}={v}" for k, v in s.items()))
    print("\n| pattern | 4 KiB pages hit rate | 2 MiB pages hit rate |\n|---|---|---|")
    for p in ("sequential", "stride-4KiB", "random-1GiB"):
        print(f"| {p} | {100 * run(p):.1f}% | {100 * run(p, page=2 << 20):.1f}% |")
    print("\nWith 2 MiB pages, 64 entries cover 128 MiB, so the stride pattern (40 MiB span) fits entirely.")
