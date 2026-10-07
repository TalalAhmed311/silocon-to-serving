"""Exercise 4 solution."""
import json
import time

import httpx


async def stream_completion(client: httpx.AsyncClient, base_url: str, prompt: str, max_tokens: int) -> dict:
    body = {"model": "any", "prompt": prompt, "max_tokens": max_tokens, "stream": True}
    t0 = time.perf_counter()                      # before the request is written: includes connection + server time
    stamps, text, usage = [], [], None
    async with client.stream("POST", f"{base_url}/v1/completions", json=body, timeout=60) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue                           # SSE comments/keep-alives/blank separators
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            piece = ev["choices"][0].get("text") or ""
            if piece:
                stamps.append(time.perf_counter())
                text.append(piece)
            if ev.get("usage"):
                usage = ev["usage"]
    return {"text": "".join(text), "ttft": stamps[0] - t0, "itl": [b - a for a, b in zip(stamps, stamps[1:])],
            "n_tokens": len(stamps), "usage": usage, "e2e": stamps[-1] - t0}
