"""Your scheduler. Same interface as platform/engine/v1/s2s_engine/scheduler.py, so LLMEngine can drive it.

Use these from the engine (don't reimplement them): BlockManager (can_allocate / allocate / match_prefix / free_seq /
commit / take_copy_ops), Sequence (num_computed, num_uncomputed, tokens, check_finished), Chunk, SchedulerOutput.
Work through exercises 01 → 04; each adds a feature to step().
"""
from __future__ import annotations

from s2s_engine.block_manager import BlockManager
from s2s_engine.scheduler import Chunk, SchedulerConfig, SchedulerOutput  # noqa: F401  (Chunk: build your chunks)
from s2s_engine.sequence import Sequence, SeqStatus


class Scheduler:
    def __init__(self, cfg: SchedulerConfig, blocks: BlockManager):
        self.cfg, self.blocks = cfg, blocks
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []

    def add(self, seq: Sequence) -> None:
        seq.status = SeqStatus.WAITING
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def step(self) -> SchedulerOutput:
        """TODO: (1) running sequences first, (2) then admissions; honour cfg.max_num_batched_tokens, cfg.max_num_seqs,
        cfg.chunked_prefill and free blocks; preempt by recompute (ex. 3); effective priority with aging (ex. 4).
        End with out.copy_ops = self.blocks.take_copy_ops()."""
        raise NotImplementedError

    def update(self, out: SchedulerOutput, sampled: dict[int, int]) -> list[Sequence]:
        """TODO: for each chunk: num_computed = start + num_tokens; blocks.commit(seq); if the chunk produced a token,
        append sampled[seq_id], and if seq.check_finished(): mark FINISHED, remove from running, free its blocks.
        Return the finished sequences."""
        raise NotImplementedError
