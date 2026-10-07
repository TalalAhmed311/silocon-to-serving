"""check_results.py — sanity-check the numbers you recorded in results/p21.json (T0; catches typos and impossibilities).

Run: uv run python course/P2-serving-engines/P2.1-first-serve-with-vllm/exercises/check_results.py results/p21.json
"""
import json
import sys

REQUIRED = ["gpu", "model", "vllm_version", "kv_tokens_reported", "kv_tokens_predicted", "decode_tok_s_batch1_graphs",
            "decode_tok_s_batch1_eager", "ceiling_tok_s_batch1"]
r = json.load(open(sys.argv[1]))
problems = [f"missing {k}" for k in REQUIRED if k not in r]
if not problems:
    ratio = r["kv_tokens_reported"] / max(1, r["kv_tokens_predicted"])
    if not 0.95 <= ratio <= 1.05:
        problems.append(f"exercise 1: KV ratio {ratio:.3f} outside 0.95–1.05")
    if r["decode_tok_s_batch1_graphs"] > r["ceiling_tok_s_batch1"]:
        problems.append("measured decode is ABOVE the bandwidth ceiling: check the ceiling inputs (weights dtype? GPU?)")
    if r["decode_tok_s_batch1_graphs"] < r["decode_tok_s_batch1_eager"]:
        problems.append("exercise 3: eager faster than graphs at batch 1 is unusual: was the first run warmed up?")
    if "client_tok_s_batch1" in r and "vllm_bench_latency_tok_s" in r and r["vllm_bench_latency_tok_s"]:
        gap = abs(r["client_tok_s_batch1"] - r["vllm_bench_latency_tok_s"]) / r["vllm_bench_latency_tok_s"]
        if gap > 0.10:
            problems.append(f"exercise 4: HTTP vs bench differ by {100 * gap:.0f}% (> 10%)")
print("\n".join(problems) if problems else "results look consistent")
sys.exit(1 if problems else 0)
