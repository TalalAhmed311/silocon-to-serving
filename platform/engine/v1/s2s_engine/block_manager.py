"""block_manager.py — paged KV-cache bookkeeping (D2 grown up; P6.3). No tensors here: block ids only.

* Fixed-size blocks of `block_size` token slots; a sequence owns a block table (logical block i → physical id).
* Reference counts: blocks can be shared (prefix reuse, fork); a block returns to the pool when its count hits 0.
* Prefix reuse by hashing FULL blocks (vLLM v1 style): h_i = H(h_{i−1}, tokens of block i). A freed block that has a
  hash stays matchable ("cached-free", LRU) until its slot is needed for something else.
* Copy-on-write: appending into a shared, partially-filled last block first copies it (fork / parallel sampling);
  the runner executes the queued (src → dst) copies before the next forward pass.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict, deque

from .sequence import Sequence


class NoFreeBlocks(RuntimeError):
    pass


def chain_hash(prev: bytes, tokens: list[int] | tuple[int, ...]) -> bytes:
    """Content hash of a full block, chained on its prefix. sha256 (not Python's hash) so it is stable across
    processes — the property vLLM needs for multi-process prefix caching."""
    h = hashlib.sha256(prev)
    h.update(b",".join(str(t).encode() for t in tokens))
    return h.digest()


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.ref = [0] * num_blocks
        self.free: deque[int] = deque(range(num_blocks))         # never-hashed (or invalidated) free blocks
        self.cached_free: OrderedDict[int, None] = OrderedDict()  # ref 0 but still holds a reusable hashed prefix (LRU)
        self.hash_of: dict[int, bytes] = {}                      # block id → content hash (full blocks only)
        self.block_of: dict[bytes, int] = {}                     # content hash → block id
        self.copy_ops: list[tuple[int, int]] = []                # pending CoW copies (src, dst)
        self.stats = {"prefix_hit_tokens": 0, "prefix_query_tokens": 0, "cow_copies": 0, "evicted_cached": 0}

    # ---- capacity ---------------------------------------------------------------------------------------------
    @property
    def num_free(self) -> int:
        return len(self.free) + len(self.cached_free)

    def blocks_for(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    def utilization(self) -> float:
        return 1.0 - self.num_free / self.num_blocks

    # ---- allocation primitives ----------------------------------------------------------------------------------
    def _take(self) -> int:
        if self.free:
            b = self.free.popleft()
        elif self.cached_free:
            b, _ = self.cached_free.popitem(last=False)          # evict the least recently freed cached block
            h = self.hash_of.pop(b)
            if self.block_of.get(h) == b:
                del self.block_of[h]
            self.stats["evicted_cached"] += 1
        else:
            raise NoFreeBlocks()
        self.ref[b] = 1
        return b

    def _release(self, b: int) -> None:
        self.ref[b] -= 1
        assert self.ref[b] >= 0, f"double free of block {b}"
        if self.ref[b] == 0:
            if b in self.hash_of:
                self.cached_free[b] = None                       # keep the content around for future prefix hits
            else:
                self.free.append(b)

    def _acquire_cached(self, b: int) -> None:
        if self.ref[b] == 0:
            self.cached_free.pop(b, None)
        self.ref[b] += 1

    # ---- sequence-level API -------------------------------------------------------------------------------------
    def match_prefix(self, seq: Sequence) -> int:
        """On first admission: reuse cached full blocks of the prompt. Returns the number of tokens reused.
        At least one prompt token is always left uncomputed, so the model produces logits for the next token."""
        if not self.enable_prefix_caching or seq.block_table:
            return 0
        bs, prev, n = self.block_size, b"", 0
        usable = (len(seq.prompt) - 1) // bs                    # full blocks we may skip
        self.stats["prefix_query_tokens"] += usable * bs
        for i in range(usable):
            h = chain_hash(prev, seq.prompt[i * bs:(i + 1) * bs])
            b = self.block_of.get(h)
            if b is None:
                break
            self._acquire_cached(b)
            seq.block_table.append(b)
            prev, n = h, n + bs
        seq.num_computed = n
        self.stats["prefix_hit_tokens"] += n
        return n

    def can_allocate(self, seq: Sequence, num_new_tokens: int) -> bool:
        need = self.blocks_for(seq.num_computed + num_new_tokens) - len(seq.block_table)
        cow = 1 if self._needs_cow(seq) else 0
        return max(0, need) + cow <= self.num_free

    def _needs_cow(self, seq: Sequence) -> bool:
        if not seq.block_table:
            return False
        last = seq.block_table[-1]
        partial = seq.num_computed % self.block_size != 0
        return partial and self.ref[last] > 1

    def allocate(self, seq: Sequence, num_new_tokens: int) -> None:
        """Make the block table cover num_computed + num_new_tokens slots (and un-share a partial last block)."""
        if not self.can_allocate(seq, num_new_tokens):
            raise NoFreeBlocks()
        if self._needs_cow(seq):
            src = seq.block_table[-1]
            dst = self._take()
            self.copy_ops.append((src, dst))
            self.stats["cow_copies"] += 1
            self._release(src)
            seq.block_table[-1] = dst
        while len(seq.block_table) < self.blocks_for(seq.num_computed + num_new_tokens):
            seq.block_table.append(self._take())

    def commit(self, seq: Sequence) -> None:
        """After a forward pass: register hashes for blocks that just became full (prompt AND generated tokens)."""
        if not self.enable_prefix_caching:
            return
        bs, toks = self.block_size, seq.tokens
        full = min(seq.num_computed, len(toks)) // bs
        prev = b""
        for i in range(full):
            b = seq.block_table[i]
            h = self.hash_of.get(b)
            if h is None:
                h = chain_hash(prev, toks[i * bs:(i + 1) * bs])
                if h in self.block_of and self.block_of[h] != b:
                    prev = h                                     # identical content already cached elsewhere: keep both
                    continue
                self.hash_of[b] = h
                self.block_of[h] = b
            prev = h

    def free_seq(self, seq: Sequence) -> None:
        for b in reversed(seq.block_table):                      # tail first: prefixes stay cached longest
            self._release(b)
        seq.block_table = []

    def fork(self, parent: Sequence, child: Sequence) -> None:
        """child shares all of parent's blocks (parallel sampling / beam search). Writes trigger CoW."""
        child.block_table = list(parent.block_table)
        child.num_computed = parent.num_computed
        for b in child.block_table:
            self.ref[b] += 1

    def take_copy_ops(self) -> list[tuple[int, int]]:
        ops, self.copy_ops = self.copy_ops, []
        return ops

    def slot(self, seq: Sequence, pos: int) -> int:
        """Physical cache slot of token position pos: block_table[pos // bs] · bs + pos % bs."""
        return seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size

    def check_invariants(self, live: list[Sequence]) -> None:
        """For tests: reference counts equal table occurrences; free + cached_free + referenced = all blocks."""
        count = [0] * self.num_blocks
        for s in live:
            for b in s.block_table:
                count[b] += 1
        assert count == self.ref, "ref counts disagree with block tables"
        referenced = {b for b in range(self.num_blocks) if self.ref[b] > 0}
        assert referenced.isdisjoint(self.free) and referenced.isdisjoint(self.cached_free)
        assert len(referenced) + len(self.free) + len(self.cached_free) == self.num_blocks
        assert set(self.free).isdisjoint(self.cached_free)
        for h, b in self.block_of.items():
            assert self.hash_of.get(b) == h
