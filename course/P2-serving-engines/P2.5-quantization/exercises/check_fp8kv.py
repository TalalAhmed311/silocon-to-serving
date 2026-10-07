"""check_fp8kv.py — sanity-check results/p25_fp8kv.json (T0)."""
import json
import sys

r = json.load(open(sys.argv[1]))
p = []
ratio = r["kv_tokens_fp8"] / max(1, r["kv_tokens_bf16"])
if not 1.6 <= ratio <= 2.2:
    p.append(f"KV ratio {ratio:.2f}: expected ≈2 — was --kv-cache-dtype fp8 really applied?")
if r["knee_fp8kv"] < 0.9 * r["knee_bf16"]:
    p.append("FP8 KV moved the knee left by >10%: unusual; check the runs used the same workload seed")
if abs(r["score_fp8kv"] - r["score_bf16"]) > 0.05:
    p.append("score delta > 5 points: large for KV quantization; re-run with a larger --limit before concluding")
print("\n".join(p) or "results look consistent")
sys.exit(1 if p else 0)
