"""Scheduler + engine invariants with the fake model (P6.2 exercises 1–4). T0.

The fake model's logits are a pure function of the token history, so a correct engine produces the SAME greedy
outputs however it batches, chunks, preempts or prefix-caches. That one property catches most scheduler bugs."""
import random

import numpy as np
import pytest

from s2s_engine import LLMEngine, SamplingParams, SchedulerConfig
from s2s_engine.engine import RequestTooLarge
from s2s_engine.model_runner import FakeRunner


def reference(runner, prompt, n):
    toks = list(prompt)
    for _ in range(n):
        toks.append(int(np.argmax(runner.logits_for(toks))))
    return toks[len(prompt):]


def workload(seed, n=12):
    rng = random.Random(seed)
    shared = [rng.randint(0, 50) for _ in range(rng.randint(0, 6))]        # some prompts share a prefix
    out = []
    for _ in range(n):
        p = (shared if rng.random() < 0.5 else []) + [rng.randint(0, 50) for _ in range(rng.randint(1, 10))]
        out.append((p, rng.randint(1, 8)))      # ≤ 24 tokens: fits every config's pool
    return out


CONFIGS = [
    dict(num_blocks=512, block_size=4, sched=SchedulerConfig(64, 4096)),                        # roomy
    dict(num_blocks=512, block_size=4, sched=SchedulerConfig(64, 7)),                           # chunked prefill
    dict(num_blocks=10, block_size=4, sched=SchedulerConfig(64, 64)),                           # memory-bound → preempt
    dict(num_blocks=14, block_size=2, sched=SchedulerConfig(3, 5)),                             # everything tight
    dict(num_blocks=64, block_size=4, sched=SchedulerConfig(64, 64), enable_prefix_caching=False),
    dict(num_blocks=64, block_size=4, sched=SchedulerConfig(64, 64, chunked_prefill=False)),
]


@pytest.mark.parametrize("cfg", range(len(CONFIGS)))
@pytest.mark.parametrize("seed", range(5))
def test_outputs_independent_of_scheduling(cfg, seed):
    runner = FakeRunner()
    kw = dict(CONFIGS[cfg])
    eng = LLMEngine(runner, **kw)
    jobs = workload(seed)
    outs = eng.generate([p for p, _ in jobs], [SamplingParams(max_tokens=m) for _, m in jobs])
    for (p, m), o in zip(jobs, outs):
        assert o == reference(runner, p, m)
    budget = kw["sched"].max_num_batched_tokens
    assert max(runner.calls) <= budget                         # token budget never exceeded
    assert eng.blocks.num_free == eng.blocks.num_blocks        # everything freed at the end
    eng.blocks.check_invariants([])


def test_memory_pressure_actually_preempts():
    runner = FakeRunner()
    eng = LLMEngine(runner, num_blocks=10, block_size=4, sched=SchedulerConfig(64, 64))
    jobs = [([i] * 9, 20) for i in range(5)]
    eng.generate([p for p, _ in jobs], [SamplingParams(max_tokens=m) for _, m in jobs])
    assert eng.stats.preemptions > 0


def test_never_fits_is_rejected_not_livelocked():
    eng = LLMEngine(FakeRunner(), num_blocks=4, block_size=4)
    with pytest.raises(RequestTooLarge):
        eng.add_request(list(range(10)), SamplingParams(max_tokens=10))


def test_prefix_cache_saves_compute():
    runner = FakeRunner()
    eng = LLMEngine(runner, num_blocks=64, block_size=4)
    prompt = list(range(33))
    eng.generate([prompt], SamplingParams(max_tokens=1))
    first = eng.stats.tokens_computed
    eng.generate([prompt], SamplingParams(max_tokens=1))
    second = eng.stats.tokens_computed - first
    assert second == 1 and first == 33                         # 32 tokens = 8 full blocks reused


def test_seeded_sampling_reproducible():
    def run():
        eng = LLMEngine(FakeRunner(), num_blocks=64, block_size=4)
        return eng.generate([[1, 2, 3], [4, 5]], SamplingParams(temperature=1.0, top_p=0.9, max_tokens=8, seed=7))
    assert run() == run()


def _starvation_run(aging_steps, steps=40):
    """One low-priority request; every step a new high-priority one arrives. Returns the step it finished, or None."""
    eng = LLMEngine(FakeRunner(), num_blocks=64, block_size=4, sched=SchedulerConfig(1, 64, aging_steps=aging_steps))
    low = eng.add_request([1, 2, 3], SamplingParams(max_tokens=1), priority=0)
    for step in range(steps):
        eng.add_request([7, step], SamplingParams(max_tokens=1), priority=1)
        eng.step()
        if low.finish_reason:
            return step
    return None


def test_priorities_starve_without_aging():
    assert _starvation_run(aging_steps=0) is None


def test_aging_bounds_waiting():
    # effective priority = priority + waited // aging_steps: after 2·aging_steps waited steps the low request beats
    # any fresh priority-1 arrival, so it finishes within that bound (+1 for the step that runs it).
    done = _starvation_run(aging_steps=3)
    assert done is not None and done <= 2 * 3 + 1
