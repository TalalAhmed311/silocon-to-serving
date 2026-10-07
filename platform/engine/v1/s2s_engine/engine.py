"""engine.py — the LLMEngine loop (P6.1): add requests, then step() until done.

    step():  out = scheduler.step()            # what to compute (budget, memory, priorities)
             logits = runner.execute(out)      # one forward pass over every scheduled chunk
             tokens = sample(logits)           # per-sequence params, per-sequence seeded RNG
             finished = scheduler.update(out, tokens)

Synchronous by design — the overlap of CPU scheduling with GPU compute (vLLM v1's async scheduling) is exercise 6.
Requests that can NEVER fit (prompt + max_tokens needs more blocks than the pool has) are rejected at add time:
otherwise preempt-by-recompute would loop forever on them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .block_manager import BlockManager
from .sampling import sample_np
from .scheduler import Scheduler, SchedulerConfig
from .sequence import SamplingParams, Sequence, SeqStatus


class RequestTooLarge(ValueError):
    pass


@dataclass
class EngineStats:
    steps: int = 0
    tokens_computed: int = 0
    tokens_generated: int = 0
    preemptions: int = 0
    step_seconds: list = field(default_factory=list)


class LLMEngine:
    def __init__(self, runner, num_blocks: int, block_size: int = 16, sched: SchedulerConfig | None = None,
                 enable_prefix_caching: bool = True, eos_id: int | None = None):
        self.runner = runner
        self.blocks = BlockManager(num_blocks, block_size, enable_prefix_caching)
        self.scheduler = Scheduler(sched or SchedulerConfig(), self.blocks)
        self.eos_id = eos_id
        self.rngs: dict[int, np.random.Generator] = {}
        self.seqs: dict[int, Sequence] = {}
        self.stats = EngineStats()

    def add_request(self, prompt: list[int], params: SamplingParams | None = None, priority: int = 0) -> Sequence:
        params = params or SamplingParams()
        if not prompt:
            raise ValueError("empty prompt")
        if self.eos_id is not None and not params.ignore_eos and self.eos_id not in params.stop_token_ids:
            params.stop_token_ids = (*params.stop_token_ids, self.eos_id)
        need = self.blocks.blocks_for(len(prompt) + params.max_tokens)
        if need > self.blocks.num_blocks:
            raise RequestTooLarge(f"needs {need} KV blocks, pool has {self.blocks.num_blocks}")
        cfg = self.scheduler.cfg
        if not cfg.chunked_prefill and len(prompt) + params.max_tokens > cfg.max_num_batched_tokens:
            # without chunking, a preempted request is re-prefilled in ONE step: prompt + generated must fit the budget
            raise RequestTooLarge(f"prompt + max_tokens exceeds max_num_batched_tokens={cfg.max_num_batched_tokens} "
                                  "with chunked prefill off")
        seq = Sequence(list(prompt), params, priority)
        self.rngs[seq.seq_id] = np.random.default_rng(params.seed if params.seed is not None else seq.seq_id)
        self.seqs[seq.seq_id] = seq
        self.scheduler.add(seq)
        return seq

    def has_unfinished(self) -> bool:
        return self.scheduler.has_work()

    def step(self) -> list[Sequence]:
        t0 = time.perf_counter()
        out = self.scheduler.step()
        self.stats.preemptions += len(out.preempted)
        if out.is_empty:
            if out.preempted or self.scheduler.running or not self.scheduler.waiting:
                return []
            raise RuntimeError("scheduler made no progress with an idle pool (budget too small for any chunk?)")
        logits = self.runner.execute(out)
        sampled = {}
        for sid, row in logits.items():
            seq = self.seqs[sid]
            sampled[sid] = sample_np(np.asarray(row, dtype=np.float32), seq.params, self.rngs[sid])
            if seq.first_token_time is None:
                seq.first_token_time = time.monotonic()
        finished = self.scheduler.update(out, sampled)
        for s in finished:
            self.rngs.pop(s.seq_id, None)
        self.stats.steps += 1
        self.stats.tokens_computed += out.num_tokens
        self.stats.tokens_generated += len(sampled)
        self.stats.step_seconds.append(time.perf_counter() - t0)
        return finished

    def generate(self, prompts: list[list[int]], params: SamplingParams | list[SamplingParams] | None = None,
                 max_steps: int = 1_000_000) -> list[list[int]]:
        """Offline batch API: returns each prompt's output tokens, in input order."""
        plist = params if isinstance(params, list) else [params] * len(prompts)
        seqs = [self.add_request(p, (sp and SamplingParams(**vars(sp))) or None) for p, sp in zip(prompts, plist)]
        for _ in range(max_steps):
            if not self.has_unfinished():
                break
            self.step()
        else:
            raise RuntimeError("max_steps exceeded")
        assert all(s.status is SeqStatus.FINISHED for s in seqs)
        return [s.output for s in seqs]
