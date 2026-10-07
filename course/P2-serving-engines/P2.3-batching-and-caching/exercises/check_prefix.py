"""check_prefix.py — sanity-check results/p23.json (T0)."""
import json
import sys

r = json.load(open(sys.argv[1]))
need = ["knee_on_rps", "knee_off_rps", "ttft_p50_on_s_at_low_rate", "ttft_p50_off_s_at_low_rate", "prefix_hit_rate"]
missing = [k for k in need if k not in r]
problems = [f"missing {k}" for k in missing]
if not missing:
    if not 0 <= r["prefix_hit_rate"] <= 1:
        problems.append("prefix_hit_rate must be a fraction in [0, 1]")
    if r["prefix_hit_rate"] > 0.5 and r["ttft_p50_on_s_at_low_rate"] >= r["ttft_p50_off_s_at_low_rate"]:
        problems.append("high hit rate but no TTFT win: was the 'off' run really without prefix caching?")
    if r["knee_on_rps"] < r["knee_off_rps"]:
        problems.append("knee moved LEFT with caching on: check that both runs used the same workload seed")
print("\n".join(problems) or "results look consistent")
sys.exit(1 if problems else 0)
