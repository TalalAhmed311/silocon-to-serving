"""bench.py — #10 measurement: acceptance rate, tokens per target call and wall-clock speedup vs target-only decoding.

Run (T2):  python -m specdec.bench --target <hf repo or path> --draft <smaller same-tokenizer model> --k 1 2 4 8 \
             --prompts prompts.txt --tokens 128        (PYTHONPATH=platform)
Output:    | method | k | α | tokens/target call | ms/token | speedup |  + results/specdec.json
Note:      the prototype recomputes the prefix every call (no KV cache), so absolute ms/token is poor; compare methods
           within this harness, and compare against vLLM's built-in spec decode (P2.6 exercise 3) for real numbers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import generate
from .hf_models import HFModel


def target_only(target: HFModel, prefix, n, seed):
    rng, out = np.random.default_rng(seed), list(prefix)
    for _ in range(n):
        p = target.probs(out)
        out.append(int(rng.choice(len(p), p=p)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--tokens", type=int, default=128)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.target)
    load = lambda p: HFModel(AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev))  # noqa: E731
    target, draft = load(a.target), load(a.draft)
    prompts = [l.strip() for l in open(a.prompts) if l.strip()]
    rows = []
    t0 = time.perf_counter()
    for i, p in enumerate(prompts):
        target_only(target, tok(p)["input_ids"], a.tokens, i)
    base_ms = 1e3 * (time.perf_counter() - t0) / (len(prompts) * a.tokens)
    rows.append({"method": "target only", "k": 0, "alpha": None, "tok_per_call": 1.0, "ms_per_token": base_ms, "speedup": 1.0})
    for k in a.k:
        t0, alphas, tpc = time.perf_counter(), [], []
        for i, p in enumerate(prompts):
            _, st = generate(tok(p)["input_ids"], draft, target, k, a.tokens, seed=i)
            alphas.append(st.acceptance_rate)
            tpc.append(st.tokens_per_target_call)
        ms = 1e3 * (time.perf_counter() - t0) / (len(prompts) * a.tokens)
        rows.append({"method": "draft model", "k": k, "alpha": float(np.mean(alphas)), "tok_per_call": float(np.mean(tpc)),
                     "ms_per_token": ms, "speedup": base_ms / ms})
    print("| method | k | α | tokens/target call | ms/token | speedup |\n|---|---|---|---|---|---|")
    for r in rows:
        alpha = "—" if r["alpha"] is None else f"{r['alpha']:.2f}"
        print(f"| {r['method']} | {r['k']} | {alpha} | {r['tok_per_call']:.2f} | {r['ms_per_token']:.1f} | {r['speedup']:.2f}x |")
    Path("results").mkdir(exist_ok=True)
    Path("results/specdec.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
