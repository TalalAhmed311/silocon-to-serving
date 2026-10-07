"""sequence.py — the request data model (nano-vllm's sequence.py ↔ vLLM v1 `Request`)."""
from __future__ import annotations

import enum
import itertools
import time
from dataclasses import dataclass, field

_ids = itertools.count()


class SeqStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class SamplingParams:
    temperature: float = 0.0        # 0 → greedy
    top_k: int = 0                  # 0 → off
    top_p: float = 1.0
    min_p: float = 0.0
    max_tokens: int = 16
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False


@dataclass
class Sequence:
    prompt: list[int]
    params: SamplingParams = field(default_factory=SamplingParams)
    priority: int = 0               # larger = more important (P6.2 exercise 4)
    seq_id: int = field(default_factory=lambda: next(_ids))
    arrival: float = field(default_factory=time.monotonic)
    output: list[int] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    block_table: list[int] = field(default_factory=list)
    num_computed: int = 0           # tokens whose K/V are in the cache (prefix-cache hits count)
    num_preemptions: int = 0
    waited_steps: int = 0           # steps spent waiting since arrival or last preemption (anti-starvation)
    first_token_time: float | None = None
    finish_reason: str | None = None

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.output

    def __len__(self) -> int:
        return len(self.prompt) + len(self.output)

    @property
    def num_uncomputed(self) -> int:
        """Tokens still to run through the model: the (rest of the) prompt, or 1 for a decode step."""
        return len(self) - self.num_computed

    @property
    def in_prefill(self) -> bool:
        return self.num_computed < len(self.prompt)

    def check_finished(self) -> bool:
        if not self.output:
            return False
        if len(self.output) >= self.params.max_tokens:
            self.finish_reason = "length"
        elif self.output[-1] in self.params.stop_token_ids and not self.params.ignore_eos:
            self.finish_reason = "stop"
        else:
            return False
        return True
