"""Your paged KV block manager. Same interface as platform/engine/v1/s2s_engine/block_manager.py.
Start from the docstrings; the README sections 1–4 give the rules. chain_hash is provided."""
from __future__ import annotations

from collections import OrderedDict, deque

from s2s_engine.block_manager import NoFreeBlocks, chain_hash  # noqa: F401
from s2s_engine.sequence import Sequence


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.ref = [0] * num_blocks
        self.free: deque[int] = deque(range(num_blocks))
        self.cached_free: OrderedDict[int, None] = OrderedDict()
        self.hash_of: dict[int, bytes] = {}
        self.block_of: dict[bytes, int] = {}
        self.copy_ops: list[tuple[int, int]] = []
        self.stats = {"prefix_hit_tokens": 0, "prefix_query_tokens": 0, "cow_copies": 0, "evicted_cached": 0}

    @property
    def num_free(self) -> int:
        return len(self.free) + len(self.cached_free)

    def blocks_for(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    def utilization(self) -> float:
        return 1.0 - self.num_free / self.num_blocks

    def match_prefix(self, seq: Sequence) -> int:
        """TODO (ex. 3): reuse cached FULL prompt blocks (chained hashes), leaving ≥ 1 prompt token uncomputed.
        Set seq.num_computed; return tokens reused. Return 0 when prefix caching is off or seq has blocks."""
        raise NotImplementedError

    def can_allocate(self, seq: Sequence, num_new_tokens: int) -> bool:
        """TODO: enough free blocks to cover num_computed + num_new_tokens (+1 if a CoW copy is needed)?"""
        raise NotImplementedError

    def allocate(self, seq: Sequence, num_new_tokens: int) -> None:
        """TODO (ex. 1, 2): CoW a shared partial last block (queue (src, dst) in copy_ops), then append new blocks.
        Raise NoFreeBlocks if it can't."""
        raise NotImplementedError

    def commit(self, seq: Sequence) -> None:
        """TODO (ex. 3): register chained hashes of blocks that are now full (min(num_computed, len(tokens)))."""
        raise NotImplementedError

    def free_seq(self, seq: Sequence) -> None:
        """TODO: release every block (tail first); hashed blocks with ref 0 go to cached_free, others to free."""
        raise NotImplementedError

    def fork(self, parent: Sequence, child: Sequence) -> None:
        """TODO (ex. 2): child shares parent's blocks (ref += 1 each) and num_computed."""
        raise NotImplementedError

    def take_copy_ops(self) -> list[tuple[int, int]]:
        ops, self.copy_ops = self.copy_ops, []
        return ops

    def slot(self, seq: Sequence, pos: int) -> int:
        return seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size

    def check_invariants(self, live: list[Sequence]) -> None:
        count = [0] * self.num_blocks
        for s in live:
            for b in s.block_table:
                count[b] += 1
        assert count == self.ref, "ref counts disagree with block tables"
        referenced = {b for b in range(self.num_blocks) if self.ref[b] > 0}
        assert referenced.isdisjoint(self.free) and referenced.isdisjoint(self.cached_free)
        assert len(referenced) + len(self.free) + len(self.cached_free) == self.num_blocks
