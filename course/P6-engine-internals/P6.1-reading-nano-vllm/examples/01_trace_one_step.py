"""Trace #0 v1's step loop with the fake model: what the scheduler decides, step by step (P6.1). T0.

Run: uv run python course/P6-engine-internals/P6.1-reading-nano-vllm/examples/01_trace_one_step.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "platform" / "engine" / "v1"))
from s2s_engine import LLMEngine, SamplingParams, SchedulerConfig  # noqa: E402
from s2s_engine.model_runner import FakeRunner  # noqa: E402

eng = LLMEngine(FakeRunner(), num_blocks=8, block_size=4, sched=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=8))
reqs = [eng.add_request(list(range(10)), SamplingParams(max_tokens=4)),
        eng.add_request(list(range(10)), SamplingParams(max_tokens=3)),        # same prompt: prefix-cache candidate
        eng.add_request([7, 7, 7], SamplingParams(max_tokens=6))]
names = {s.seq_id: "ABC"[i] for i, s in enumerate(reqs)}

step = 0
while eng.has_unfinished():
    out = eng.scheduler.step()
    logits = eng.runner.execute(out)
    from s2s_engine.sampling import sample_np                               # noqa: E402
    sampled = {sid: sample_np(l, eng.seqs[sid].params, eng.rngs[sid]) for sid, l in logits.items()}
    fin = eng.scheduler.update(out, sampled)
    step += 1
    chunks = ", ".join(f"{names[c.seq.seq_id]}[{c.start}:{c.start + c.num_tokens}]" for c in out.chunks)
    print(f"step {step:2d} | {out.num_tokens:2d} tokens | {chunks:32s} | free blocks {eng.blocks.num_free} "
          f"| preempted {[names[s.seq_id] for s in out.preempted]} | sampled {[(names[k], v) for k, v in sampled.items()]}"
          + (f" | finished {[names[s.seq_id] for s in fin]}" if fin else ""))
print(f"all {len(reqs)} requests finished in {step} steps; prefix-cache hit tokens: {eng.blocks.stats['prefix_hit_tokens']}")
