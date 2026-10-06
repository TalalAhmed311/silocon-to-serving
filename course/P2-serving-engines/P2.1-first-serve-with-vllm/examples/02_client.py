"""02_client.py — one non-streaming and one streaming completion against an OpenAI-compatible server; TTFT/ITL.

Run:      python course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/02_client.py --url http://127.0.0.1:8000 [--api-key K]
Works against vLLM (T2) and platform/mockllm (T0).
"""
import argparse
import json
import os
import time

import httpx
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--api-key", default=os.environ.get("S2S_API_KEY"))
ap.add_argument("--max-tokens", type=int, default=64)
a = ap.parse_args()
H = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
model = httpx.get(f"{a.url}/v1/models", headers=H).json()["data"][0]["id"]
msgs = [{"role": "user", "content": "Explain in two sentences why LLM decode is memory-bound."}]
r = httpx.post(f"{a.url}/v1/chat/completions", headers=H, timeout=120,
               json={"model": model, "messages": msgs, "max_tokens": a.max_tokens, "temperature": 0})
r.raise_for_status()
print("non-streaming:", r.json()["choices"][0]["message"]["content"][:300], "\nusage:", r.json()["usage"])

t0, stamps = time.perf_counter(), []
with httpx.stream("POST", f"{a.url}/v1/chat/completions", headers=H, timeout=120,
                  json={"model": model, "messages": msgs, "max_tokens": a.max_tokens, "temperature": 0, "stream": True}) as s:
    for line in s.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            if json.loads(line[6:])["choices"][0]["delta"].get("content"):
                stamps.append(time.perf_counter())
itl = np.diff(stamps) * 1e3
print(f"streaming: {len(stamps)} chunks, TTFT {1e3 * (stamps[0] - t0):.0f} ms, ITL p50 {np.median(itl):.1f} ms p90 {np.percentile(itl, 90):.1f} ms")
