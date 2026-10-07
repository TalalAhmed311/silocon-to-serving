"""04_d3_vs_log.py — compare your D3 (P1.5) KV-capacity prediction with what vLLM logged at start-up.

Run:      python course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/04_d3_vs_log.py --log vllm.log \
            --model-config /opt/models/<model>/config.json --gpu L4 [--mem-util 0.90] [--kv-dtype bf16] [--weights bf16]
Expected: predicted vs reported KV tokens and their ratio. If the log pattern doesn't match (vLLM wording changes),
          the script prints the candidate lines so you can update the regex in P1.5's calculator.
Hardware: T0 (reads a log file).
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "platform"))
from capacity.core import Model, plan  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
ap.add_argument("--model-config", required=True)
ap.add_argument("--gpu", required=True)
ap.add_argument("--mem-util", type=float, default=0.90)
ap.add_argument("--weights", default="bf16")
ap.add_argument("--kv-dtype", default="bf16")
ap.add_argument("--activation-gb", type=float, default=1.0)
a = ap.parse_args()
text = Path(a.log).read_text(errors="replace")
m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)   # UNVERIFIED wording for v0.31.0 — see docstring
p = plan(Model.from_file(a.model_config), a.gpu, a.weights, a.kv_dtype, mem_util=a.mem_util, activation_gb=a.activation_gb)
print(f"D3 predicted KV tokens: {p.max_kv_tokens:,}  (weights {p.weights_gb_per_gpu:.1f} GiB, KV budget {p.kv_budget_gb_per_gpu:.1f} GiB)")
if not m:
    print("Could not find the KV size line. Candidate lines:")
    for line in text.splitlines():
        if re.search(r"kv cache|KV cache|blocks|concurrency", line):
            print("  ", line.strip()[:200])
    sys.exit(1)
got = int(m.group(1).replace(",", ""))
print(f"vLLM reported KV tokens: {got:,}   ratio reported/predicted = {got / p.max_kv_tokens:.3f}")
print("Tune --activation-gb (and check --mem-util) until the ratio is within 0.95–1.05; record both in results/p21.json.")
