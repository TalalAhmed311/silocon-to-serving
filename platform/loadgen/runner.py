"""runner.py — send a Workload to a server open-loop (requests go out on schedule, never waiting for earlier ones)
and record per-request timings. Open loop matters: a closed-loop client slows down when the server does, which hides
queueing (coordinated omission, P2.4).
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx

from .workload import Request, Workload, prompt_text


@dataclass
class Result:
    req: Request
    sent: float = 0.0
    first: float | None = None
    stamps: list[float] = field(default_factory=list)
    error: str | None = None
    usage: dict | None = None

    @property
    def ttft(self):
        return None if self.first is None else self.first - self.sent

    @property
    def e2e(self):
        return None if not self.stamps else self.stamps[-1] - self.sent

    @property
    def tpot(self):
        n = len(self.stamps)
        return None if n < 2 else (self.stamps[-1] - self.stamps[0]) / (n - 1)

    @property
    def itls(self):
        return [b - a for a, b in zip(self.stamps, self.stamps[1:])]


async def _one(c: httpx.AsyncClient, url: str, model: str, headers: dict, r: Result, shared: int, t0: float, timeout: float):
    await asyncio.sleep(max(0.0, t0 + r.req.t - time.perf_counter()))
    body = {"model": model, "prompt": prompt_text(r.req, shared), "max_tokens": r.req.output_tokens, "temperature": 0,
            "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}}
    r.sent = time.perf_counter()
    try:
        async with c.stream("POST", f"{url}/v1/completions", json=body, headers=headers, timeout=timeout) as resp:
            if resp.status_code != 200:
                r.error = f"HTTP {resp.status_code}"
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                if ev.get("usage"):
                    r.usage = ev["usage"]
                if ev.get("choices") and ev["choices"][0].get("text"):
                    now = time.perf_counter()
                    r.first = r.first or now
                    r.stamps.append(now)
    except Exception as e:  # noqa: BLE001 — timeouts/resets are data, not crashes
        r.error = type(e).__name__


async def run(workload: Workload, url: str, model: str = "any", api_key: str | None = None, timeout: float = 600) -> list[Result]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    results = [Result(q) for q in workload.requests()]
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=200)
    async with httpx.AsyncClient(limits=limits) as c:
        t0 = time.perf_counter()
        await asyncio.gather(*[_one(c, url, model, headers, r, workload.shared_prefix_tokens, t0, timeout) for r in results])
    return results
