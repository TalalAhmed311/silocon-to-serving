"""Tests for exercises/my_scheduler.py (S2S_SOLUTIONS=1 runs them against the reference scheduler). T0."""
import importlib.util
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "platform" / "engine" / "v1"))
from s2s_engine import LLMEngine, SamplingParams, SchedulerConfig  # noqa: E402
from s2s_engine.model_runner import FakeRunner  # noqa: E402

if os.environ.get("S2S_SOLUTIONS") == "1":
    from s2s_engine.scheduler import Scheduler  # noqa: E402
else:
    _spec = importlib.util.spec_from_file_location("my_scheduler", Path(__file__).with_name("my_scheduler.py"))
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    Scheduler = _m.Scheduler


def engine(num_blocks, block_size, sched, **kw):
    runner = FakeRunner()
    eng = LLMEngine(runner, num_blocks, block_size, sched, **kw)
    eng.scheduler = Scheduler(sched, eng.blocks)
    return eng, runner


def reference(runner, prompt, n):
    toks = list(prompt)
    for _ in range(n):
        toks.append(int(np.argmax(runner.logits_for(toks))))
    return toks[len(prompt):]


def jobs(seed, n=10):
    rng = random.Random(seed)
    return [([rng.randint(0, 50) for _ in range(rng.randint(1, 14))], rng.randint(1, 8)) for _ in range(n)]


def run_and_check(eng, runner, js, budget):
    outs = eng.generate([p for p, _ in js], [SamplingParams(max_tokens=m) for _, m in js])
    for (p, m), o in zip(js, outs):
        assert o == reference(runner, p, m)
    assert max(runner.calls) <= budget
    assert eng.blocks.num_free == eng.blocks.num_blocks


@pytest.mark.parametrize("seed", range(4))
def test_fcfs_all_finish_within_budget(seed):
    cfg = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=64, chunked_prefill=False)
    eng, runner = engine(256, 4, cfg)
    run_and_check(eng, runner, jobs(seed), 64)


def test_fcfs_respects_max_num_seqs():
    cfg = SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64)
    eng, runner = engine(256, 4, cfg)
    for p, m in jobs(0):
        eng.add_request(p, SamplingParams(max_tokens=m))
    while eng.has_unfinished():
        eng.step()
        assert len(eng.scheduler.running) <= 2


@pytest.mark.parametrize("budget", [1, 3, 7])
def test_chunked_prefill_small_budget(budget):
    cfg = SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=budget)
    eng, runner = engine(256, 4, cfg)
    run_and_check(eng, runner, jobs(1), budget)


def test_chunked_prefill_lets_decodes_continue():
    cfg = SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=8)
    eng, runner = engine(256, 4, cfg)
    short = eng.add_request([1, 2], SamplingParams(max_tokens=20))
    eng.step()                                                     # short prefill → first token
    eng.add_request(list(range(60)), SamplingParams(max_tokens=1))
    for _ in range(5):                                             # long prompt is chunked; short keeps decoding
        n = len(short.output)
        eng.step()
        assert len(short.output) == n + 1


@pytest.mark.parametrize("seed", range(4))
def test_preempt_under_memory_pressure(seed):
    cfg = SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=32)
    eng, runner = engine(8, 4, cfg)                                # 32 slots; every job fits alone (≤ 22 tokens)
    run_and_check(eng, runner, jobs(seed), 32)


def test_preempt_happens():
    cfg = SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=64)
    eng, runner = engine(10, 4, cfg)
    js = [([i] * 9, 20) for i in range(5)]
    run_and_check(eng, runner, js, 64)
    assert eng.stats.preemptions > 0


def _starve(aging):
    cfg = SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=64, aging_steps=aging)
    eng, _ = engine(64, 4, cfg)
    low = eng.add_request([1, 2, 3], SamplingParams(max_tokens=1), priority=0)
    for step in range(40):
        eng.add_request([7, step], SamplingParams(max_tokens=1), priority=1)
        eng.step()
        if low.finish_reason:
            return step
    return None


def test_priority_strict_starves():
    assert _starve(0) is None


@pytest.mark.parametrize("aging", [1, 2, 5])
def test_priority_aging_bound(aging):
    done = _starve(aging)
    assert done is not None and done <= 2 * aging + 1
