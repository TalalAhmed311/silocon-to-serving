"""Block size vs internal fragmentation and prefix-hit rate (P6.3 exercise 5 starter). T0, fake model.

Run: uv run python course/P6-engine-internals/P6.3-paged-kv-block-manager/bench/utilization.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "platform" / "engine" / "v1"))
from s2s_engine import LLMEngine, SamplingParams, SchedulerConfig  # noqa: E402
from s2s_engine.model_runner import FakeRunner  # noqa: E402

rng = random.Random(0)
system = [rng.randint(0, 100) for _ in range(150)]
trace = [(system + [rng.randint(0, 100) for _ in range(rng.randint(5, 120))], rng.randint(10, 80)) for _ in range(60)]

print("block_size | mean slot utilization | prefix-hit rate | preemptions")
for bs in (1, 8, 16, 32):
    total_slots = 8192
    eng = LLMEngine(FakeRunner(), num_blocks=total_slots // bs, block_size=bs, sched=SchedulerConfig(16, 512))
    for p, m in trace:
        eng.add_request(p, SamplingParams(max_tokens=m, ignore_eos=True))
    util = []
    while eng.has_unfinished():
        eng.step()
        used = sum(s.num_computed for s in eng.scheduler.running)
        alloc = sum(len(s.block_table) for s in eng.scheduler.running) * bs
        if alloc:
            util.append(used / alloc)
    st = eng.blocks.stats
    hit = st["prefix_hit_tokens"] / max(1, sum(len(p) for p, _ in trace))
    print(f"{bs:10d} | {sum(util) / len(util):21.3f} | {hit:15.3f} | {eng.stats.preemptions}")
