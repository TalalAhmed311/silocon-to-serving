"""scheduler.py — continuous batching with chunked prefill, a token budget, preemption by recompute, and priority
classes with an anti-starvation bound (P6.2). Pure Python; the model is somebody else's problem.

One call to step() decides what the next forward pass computes:
  1. RUNNING sequences first, in admission order: decodes (1 token each) and unfinished prefill chunks.
     If the KV pool can't hold a running sequence's next tokens, PREEMPT the lowest-priority, most recently admitted
     running sequence (recompute: free its blocks, put it back at the front of the waiting queue) and retry.
  2. Then ADMIT waiting sequences by effective priority (priority + waited_steps // aging_steps, then arrival), while
     token budget, max_num_seqs and free blocks allow. Admission never preempts running work.
Invariants (property-tested): Σ scheduled tokens ≤ max_num_batched_tokens; a sequence is in exactly one queue;
every admitted request eventually finishes (no deadlock, no starvation beyond the aging bound).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .block_manager import BlockManager
from .sequence import Sequence, SeqStatus


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 2048
    chunked_prefill: bool = True        # False: a prompt must fit in one step's budget or wait
    aging_steps: int = 0                # > 0: +1 effective priority per aging_steps waited (anti-starvation)


@dataclass
class Chunk:
    seq: Sequence
    start: int                          # first position computed in this step (== seq.num_computed when scheduled)
    num_tokens: int

    @property
    def produces_token(self) -> bool:
        """Does this chunk reach the end of the sequence's known tokens (→ sample the next token)?"""
        return self.start + self.num_tokens == len(self.seq)


@dataclass
class SchedulerOutput:
    chunks: list[Chunk] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    copy_ops: list[tuple[int, int]] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return sum(c.num_tokens for c in self.chunks)

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class Scheduler:
    def __init__(self, cfg: SchedulerConfig, blocks: BlockManager):
        self.cfg, self.blocks = cfg, blocks
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.step_count = 0

    # ---- queues -------------------------------------------------------------------------------------------------
    def add(self, seq: Sequence) -> None:
        seq.status = SeqStatus.WAITING
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def effective_priority(self, seq: Sequence) -> int:
        bonus = seq.waited_steps // self.cfg.aging_steps if self.cfg.aging_steps > 0 else 0
        return seq.priority + bonus

    def _pick_waiting(self) -> Sequence | None:
        if not self.waiting:
            return None
        # highest effective priority; preempted sequences first among equals (they were admitted before); then arrival
        return max(self.waiting, key=lambda s: (self.effective_priority(s), s.num_preemptions > 0, -s.arrival, -s.seq_id))

    def _preempt(self, victim: Sequence, out: SchedulerOutput) -> None:
        self.running.remove(victim)
        self.blocks.free_seq(victim)
        victim.num_computed = 0                    # recompute: prompt + generated tokens are re-prefilled later
        victim.num_preemptions += 1
        victim.waited_steps = 0
        victim.status = SeqStatus.WAITING
        self.waiting.append(victim)
        out.preempted.append(victim)

    def _victim(self, exclude: Sequence, protected: set[int]) -> Sequence | None:
        """Lowest effective priority, then most recently admitted; never one already placed in this step's batch."""
        cands = [s for s in self.running if s is not exclude and s.seq_id not in protected]
        if not cands:
            return None
        return min(cands, key=lambda s: (self.effective_priority(s), -self.running.index(s)))

    # ---- the step ---------------------------------------------------------------------------------------------------
    def step(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.cfg.max_num_batched_tokens
        scheduled: set[int] = set()
        # 1. running sequences (admission order)
        for seq in list(self.running):
            if seq not in self.running:                # preempted earlier in this loop
                continue
            n = seq.num_uncomputed
            if n <= 0:
                continue
            if self.cfg.chunked_prefill:
                n = min(n, budget)
            elif n > budget:
                continue
            if n == 0:
                break                                  # budget exhausted
            while not self.blocks.can_allocate(seq, n):
                victim = self._victim(exclude=seq, protected=scheduled) or seq
                self._preempt(victim, out)
                if victim is seq:
                    break
            if seq not in self.running:
                continue
            self.blocks.allocate(seq, n)
            out.chunks.append(Chunk(seq, seq.num_computed, n))
            scheduled.add(seq.seq_id)
            budget -= n
        # 2. admission — skipped in a step that had to preempt (memory is tight; don't thrash)
        while not out.preempted and budget > 0 and len(self.running) < self.cfg.max_num_seqs:
            seq = self._pick_waiting()
            if seq is None:
                break
            if not seq.block_table and seq.num_computed == 0:
                self.blocks.match_prefix(seq)          # prefix-cache hits skip work (P6.3)
            n = seq.num_uncomputed
            if self.cfg.chunked_prefill:
                n = min(n, budget)
            elif n > budget:
                self.blocks.free_seq(seq)
                seq.num_computed = 0
                break                                  # don't let a smaller request jump the queue
            if not self.blocks.can_allocate(seq, n):
                self.blocks.free_seq(seq)              # don't sit on prefix-hit blocks while waiting (livelock)
                seq.num_computed = 0
                break                                  # wait for memory; admission never preempts running work
            self.blocks.allocate(seq, n)
            self.waiting.remove(seq)
            seq.status = SeqStatus.RUNNING
            seq.waited_steps = 0
            self.running.append(seq)
            out.chunks.append(Chunk(seq, seq.num_computed, n))
            scheduled.add(seq.seq_id)
            budget -= n
        for s in self.waiting:
            s.waited_steps += 1
        out.copy_ops = self.blocks.take_copy_ops()
        self.step_count += 1
        return out

    # ---- after the forward pass -------------------------------------------------------------------------------------
    def update(self, out: SchedulerOutput, sampled: dict[int, int]) -> list[Sequence]:
        """Advance num_computed; append sampled tokens (seq_id → token) for chunks that produced one; finish and free
        completed sequences. Returns the sequences that finished in this step."""
        finished = []
        for c in out.chunks:
            s = c.seq
            s.num_computed = c.start + c.num_tokens
            self.blocks.commit(s)
            if c.produces_token and s.seq_id in sampled:
                s.output.append(sampled[s.seq_id])
                if s.check_finished():
                    s.status = SeqStatus.FINISHED
                    self.running.remove(s)
                    self.blocks.free_seq(s)
                    finished.append(s)
        return finished
