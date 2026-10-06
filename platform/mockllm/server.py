"""mockllm/server.py — an OpenAI-compatible fake LLM server that *behaves* like a GPU serving engine.

Why: P1.3 (metrics), P2.3 (#4 load test), P2.7 (#6 gateway), P3.5 (observability) and P3.6 (autoscaling) all need a
backend with realistic queueing, batching and latency behaviour that runs on a laptop and in CI.

Cost model (one engine "step" = one iteration of a continuous-batching scheduler, like vLLM's):
    step_ms = base_step_ms + per_seq_ms * running + prefill_ms_per_token * (prompt tokens admitted this step)
    - at most max_num_seqs sequences run at once; the rest wait in a FIFO queue (-> TTFT grows under load)
    - each running sequence emits one token per step (-> ITL grows with batch size)
    - "KV cache": each sequence holds prompt+generated tokens; admission stops when kv_capacity_tokens is reached
These are made-up but shaped like the real thing; P2 replaces them with vLLM measured on a GPU.

Endpoints: POST /v1/completions, POST /v1/chat/completions (both support "stream": true, SSE),
           GET /health, GET /metrics (Prometheus; vLLM-style metric names so dashboards/alerts carry over).
Auth:      if MOCKLLM_API_KEY is set, requests need "Authorization: Bearer <key>".
Faults:    MOCKLLM_FAIL_RATE (HTTP 503 before any work) and MOCKLLM_FAIL_MIDSTREAM_RATE (stream cut after 3 tokens),
           seeded by MOCKLLM_SEED — for testing gateway retries/fallbacks (P2.7).

Run:       uv run uvicorn mockllm.server:app --app-dir platform --port 8001     (from the repo root)
           MOCKLLM_MAX_NUM_SEQS=8 MOCKLLM_PER_SEQ_MS=2 uv run uvicorn mockllm.server:app --app-dir platform --port 8001
           (the --app-dir dance: a top-level package called "platform" would shadow the standard library module)
Hardware:  T0.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest


def _env(name: str, default: float) -> float:
    return float(os.environ.get(f"MOCKLLM_{name}", default))


@dataclass
class EngineConfig:
    base_step_ms: float = _env("BASE_STEP_MS", 8.0)          # weight-read time of one decode step (memory-bound)
    per_seq_ms: float = _env("PER_SEQ_MS", 0.25)             # extra per running sequence (KV reads, attention)
    prefill_ms_per_token: float = _env("PREFILL_MS_PER_TOKEN", 0.05)
    max_num_seqs: int = int(_env("MAX_NUM_SEQS", 64))
    kv_capacity_tokens: int = int(_env("KV_CAPACITY_TOKENS", 200_000))
    model_name: str = os.environ.get("MOCKLLM_MODEL", "mock-llama-8b")
    time_scale: float = _env("TIME_SCALE", 1.0)              # <1 speeds everything up (tests)
    fail_rate: float = _env("FAIL_RATE", 0.0)                # fraction of requests answered with HTTP 503 (gateway tests)
    fail_midstream_rate: float = _env("FAIL_MIDSTREAM_RATE", 0.0)  # fraction of streams cut after a few tokens


@dataclass
class Seq:
    rid: str
    prompt_tokens: int
    max_tokens: int
    arrival: float
    out: asyncio.Queue = field(default_factory=asyncio.Queue)
    generated: int = 0
    first_token_at: float | None = None


class Engine:
    """A single-threaded asyncio scheduler loop: admit -> step -> emit, forever."""

    def __init__(self, cfg: EngineConfig, registry: CollectorRegistry):
        self.cfg, self.waiting, self.running = cfg, [], []
        self.kv_used = 0
        self._task: asyncio.Task | None = None
        r = registry
        self.m_waiting = Gauge("vllm:num_requests_waiting", "Requests waiting", registry=r)
        self.m_running = Gauge("vllm:num_requests_running", "Requests running", registry=r)
        self.m_kv = Gauge("vllm:kv_cache_usage_perc", "KV cache usage fraction", registry=r)
        buckets = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 60)
        self.m_ttft = Histogram("vllm:time_to_first_token_seconds", "TTFT", buckets=buckets, registry=r)
        self.m_itl = Histogram("vllm:inter_token_latency_seconds", "ITL", buckets=buckets, registry=r)
        self.m_queue = Histogram("vllm:request_queue_time_seconds", "Queue time", buckets=buckets, registry=r)
        self.m_e2e = Histogram("vllm:e2e_request_latency_seconds", "E2E latency", buckets=buckets, registry=r)
        self.m_prompt = Counter("vllm:prompt_tokens", "Prompt tokens processed", registry=r)
        self.m_gen = Counter("vllm:generation_tokens", "Generated tokens", registry=r)
        self.m_preempt = Counter("vllm:num_preemptions", "Preemptions", registry=r)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._loop())

    def submit(self, seq: Seq) -> None:
        self.waiting.append(seq)
        self.m_waiting.set(len(self.waiting))
        self.start()

    async def _loop(self) -> None:
        c = self.cfg
        while True:
            if not self.waiting and not self.running:
                await asyncio.sleep(0.001)
                continue
            # admit FIFO while there is batch room and KV room for the whole request (prompt + max output)
            admitted_prompt = 0
            now = time.monotonic()
            while self.waiting and len(self.running) < c.max_num_seqs:
                s = self.waiting[0]
                need = s.prompt_tokens + s.max_tokens
                if self.kv_used + need > c.kv_capacity_tokens:
                    break
                self.waiting.pop(0)
                self.kv_used += need
                self.running.append(s)
                admitted_prompt += s.prompt_tokens
                self.m_queue.observe(now - s.arrival)
                self.m_prompt.inc(s.prompt_tokens)
            step_ms = c.base_step_ms + c.per_seq_ms * len(self.running) + c.prefill_ms_per_token * admitted_prompt
            await asyncio.sleep(step_ms * c.time_scale / 1000)
            t = time.monotonic()
            done = []
            for s in self.running:
                s.generated += 1
                if s.first_token_at is None:
                    s.first_token_at = t
                    self.m_ttft.observe(t - s.arrival)
                else:
                    self.m_itl.observe(step_ms * c.time_scale / 1000)
                self.m_gen.inc()
                s.out.put_nowait(f" tok{s.generated}")
                if s.generated >= s.max_tokens:
                    done.append(s)
            for s in done:
                self.running.remove(s)
                self.kv_used -= s.prompt_tokens + s.max_tokens
                self.m_e2e.observe(t - s.arrival)
                s.out.put_nowait(None)  # end of stream
            self.m_waiting.set(len(self.waiting))
            self.m_running.set(len(self.running))
            self.m_kv.set(self.kv_used / c.kv_capacity_tokens)


registry = CollectorRegistry()
cfg = EngineConfig()
engine = Engine(cfg, registry)
app = FastAPI(title="mockllm")


def _auth(req: Request) -> None:
    key = os.environ.get("MOCKLLM_API_KEY")
    if key and req.headers.get("authorization") != f"Bearer {key}":
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _count_tokens(text: str) -> int:
    return max(1, len(text.split()))  # whitespace "tokenizer": good enough for a cost model


_rng = random.Random(int(os.environ.get("MOCKLLM_SEED", "0")))


async def _generate(req: Request, prompt_tokens: int, chat: bool):
    body = await req.json()
    if _rng.random() < cfg.fail_rate:
        return JSONResponse({"error": {"message": "injected failure", "type": "overloaded"}}, status_code=503)
    cut_after = 3 if _rng.random() < cfg.fail_midstream_rate else None
    max_tokens = int(body.get("max_tokens", 16))
    seq = Seq(rid=f"cmpl-{uuid.uuid4().hex[:12]}", prompt_tokens=prompt_tokens, max_tokens=max_tokens, arrival=time.monotonic())
    engine.submit(seq)
    created = int(time.time())

    def chunk(text: str | None, finish: str | None) -> dict:
        if chat:
            delta = {"content": text} if text is not None else {}
            return {"id": seq.rid, "object": "chat.completion.chunk", "created": created, "model": cfg.model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return {"id": seq.rid, "object": "text_completion", "created": created, "model": cfg.model_name,
                "choices": [{"index": 0, "text": text or "", "finish_reason": finish}]}

    usage = lambda: {"prompt_tokens": prompt_tokens, "completion_tokens": seq.generated,  # noqa: E731
                     "total_tokens": prompt_tokens + seq.generated}
    if body.get("stream"):
        async def sse():
            sent = 0
            while (tok := await seq.out.get()) is not None:
                yield f"data: {json.dumps(chunk(tok, None))}\n\n"
                sent += 1
                if cut_after is not None and sent >= cut_after:
                    raise ConnectionResetError("injected mid-stream failure")   # the client sees a truncated stream
            final = chunk(None, "length")
            final["usage"] = usage()
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")
    text = ""
    while (tok := await seq.out.get()) is not None:
        text += tok
    if chat:
        resp = {"id": seq.rid, "object": "chat.completion", "created": created, "model": cfg.model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "length"}]}
    else:
        resp = {"id": seq.rid, "object": "text_completion", "created": created, "model": cfg.model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": "length"}]}
    resp["usage"] = usage()
    return JSONResponse(resp)


@app.post("/v1/completions")
async def completions(req: Request):
    _auth(req)
    body = await req.json()
    return await _generate(req, _count_tokens(str(body.get("prompt", ""))), chat=False)


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    _auth(req)
    body = await req.json()
    text = " ".join(str(m.get("content", "")) for m in body.get("messages", []))
    return await _generate(req, _count_tokens(text), chat=True)


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": cfg.model_name, "object": "model", "owned_by": "mockllm"}]}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(registry).decode(), media_type=CONTENT_TYPE_LATEST)
