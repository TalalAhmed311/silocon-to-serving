"""Token budget vs time-to-first-token for a long prompt among running decodes — in engine steps (fake model). T0.

Run: uv run python course/P6-engine-internals/P6.2-scheduler/bench/budget_sweep.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "platform" / "engine" / "v1"))
from s2s_engine import LLMEngine, SamplingParams, SchedulerConfig  # noqa: E402
from s2s_engine.model_runner import FakeRunner  # noqa: E402

print("budget | steps to long prompt's 1st token | max tokens in a step | decodes stalled (steps)")
for budget in (64, 128, 256, 512, 2048):
    runner = FakeRunner()
    eng = LLMEngine(runner, num_blocks=4096, block_size=16, sched=SchedulerConfig(64, budget))
    decs = [eng.add_request([i, i + 1], SamplingParams(max_tokens=200, ignore_eos=True)) for i in range(32)]
    eng.step()
    long = eng.add_request(list(range(2000)), SamplingParams(max_tokens=1))
    steps, stalled = 0, 0
    while not long.output:
        before = [len(d.output) for d in decs]
        eng.step()
        steps += 1
        stalled += any(len(d.output) == b for d, b in zip(decs, before))
    print(f"{budget:6d} | {steps:32d} | {max(runner.calls):20d} | {stalled}")
