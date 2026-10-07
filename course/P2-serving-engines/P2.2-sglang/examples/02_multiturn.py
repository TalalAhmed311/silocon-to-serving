"""02_multiturn.py — replay multi-turn conversations; report TTFT per turn (the prefix-cache workload).

Run:      python course/P2-serving-engines/P2.2-sglang/examples/02_multiturn.py --url http://127.0.0.1:30000 \
            [--conversations 8] [--turns 6] [--system-tokens 1500] [--api-key K]
Output:   | turn | history tokens (approx) | p50 TTFT ms | p90 TTFT ms |
          With a prefix cache TTFT stays ~flat across turns; without one it grows with the history.
Hardware: T2 (vLLM or SGLang), or T0 against mockllm (the mock has no prefix cache: TTFT grows with history).
"""
import argparse
import asyncio
import json
import os
import time

import httpx
import numpy as np

SYSTEM = "You are a meticulous assistant. " * 300


async def turn(c, url, H, model, messages, max_tokens):
    t0 = time.perf_counter()
    first, text = None, []
    async with c.stream("POST", f"{url}/v1/chat/completions", headers=H, timeout=300,
                        json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True}) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                piece = json.loads(line[6:])["choices"][0]["delta"].get("content")
                if piece:
                    first = first or time.perf_counter()
                    text.append(piece)
    return first - t0, "".join(text)


async def conversation(c, url, H, model, turns, sys_words, max_tokens, ttfts):
    messages = [{"role": "system", "content": " ".join(SYSTEM.split()[:sys_words])}]
    for k in range(turns):
        messages.append({"role": "user", "content": f"Question {k}: summarize the previous answer and add one fact."})
        t, reply = await turn(c, url, H, model, messages, max_tokens)
        ttfts[k].append(t)
        messages.append({"role": "assistant", "content": reply})


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--api-key", default=os.environ.get("S2S_API_KEY"))
    ap.add_argument("--conversations", type=int, default=8)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--system-tokens", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()
    H = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
    async with httpx.AsyncClient() as c:
        model = (await c.get(f"{a.url}/v1/models", headers=H)).json()["data"][0]["id"]
        ttfts = [[] for _ in range(a.turns)]
        await asyncio.gather(*[conversation(c, a.url, H, model, a.turns, a.system_tokens, a.max_tokens, ttfts)
                               for _ in range(a.conversations)])
    print("| turn | history tokens (approx) | p50 TTFT ms | p90 TTFT ms |\n|---|---|---|---|")
    for k, ts in enumerate(ttfts):
        hist = a.system_tokens + k * (12 + a.max_tokens)
        print(f"| {k + 1} | {hist} | {1e3 * np.median(ts):.0f} | {1e3 * np.percentile(ts, 90):.0f} |")


if __name__ == "__main__":
    asyncio.run(main())
