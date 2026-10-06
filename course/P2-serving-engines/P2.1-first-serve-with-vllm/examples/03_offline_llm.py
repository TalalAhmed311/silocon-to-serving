"""03_offline_llm.py — vLLM's offline batch API: no server, maximum throughput for a fixed list of prompts.

Run:      python course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/03_offline_llm.py --model /opt/models/<model> [--n 256]
Expected: total generated tokens / wall time — much higher than one-at-a-time, because the engine batches everything.
Hardware: T2.
"""
import argparse
import time

from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--max-tokens", type=int, default=256)
a = ap.parse_args()
llm = LLM(model=a.model, max_model_len=4096)
prompts = [f"Write a haiku about GPU number {i}." for i in range(a.n)]
sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=a.max_tokens, seed=0)
t0 = time.perf_counter()
outs = llm.generate(prompts, sp)
dt = time.perf_counter() - t0
toks = sum(len(o.outputs[0].token_ids) for o in outs)
print(f"{a.n} prompts, {toks} generated tokens in {dt:.1f} s -> {toks / dt:.0f} output tok/s (offline, fully batched)")
print("sample:", outs[0].outputs[0].text.strip()[:200])
