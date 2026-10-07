"""server.py — #0 v1 behind an OpenAI-compatible HTTP API (P6.7), so #6 (gateway) and #4 (loadgen) treat it like vLLM.

Prompts are token ids (a list, or whitespace-separated integers). Any other text is mapped one word → one id by a
stable hash — a stand-in so #4 loadgen works; real tokenization is out of scope.
The text field returns space-separated token ids, which is what loadgen counts.
One background task drives engine.step() whenever there is work; each request awaits its own asyncio.Queue of tokens.
Metric names follow vLLM (`vllm:num_requests_running`, `vllm:time_to_first_token_seconds`, …) so the P3.5 dashboards
and alerts carry over unchanged.

Run (T0, fake model):  S2S_RUNNER=fake uv run uvicorn s2s_engine.server:app --app-dir platform/engine/v1 --port 8002
Run (T2, GPU):         S2S_RUNNER=torch S2S_MODEL=build/tiny-llama S2S_NUM_BLOCKS=4096 uv run uvicorn ...   TODO(run-on: T2)
Auth:                  S2S_API_KEY → requires "Authorization: Bearer <key>" (never deploy without it, or the gateway).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from .engine import LLMEngine, RequestTooLarge
from .scheduler import SchedulerConfig
from .sequence import SamplingParams


def build_engine() -> LLMEngine:
    kind = os.environ.get("S2S_RUNNER", "fake")
    nb, bs = int(os.environ.get("S2S_NUM_BLOCKS", 1024)), int(os.environ.get("S2S_BLOCK_SIZE", 16))
    sched = SchedulerConfig(max_num_seqs=int(os.environ.get("S2S_MAX_NUM_SEQS", 64)),
                            max_num_batched_tokens=int(os.environ.get("S2S_MAX_BATCHED_TOKENS", 2048)))
    if kind == "fake":
        from .model_runner import FakeRunner
        runner = FakeRunner(int(os.environ.get("S2S_VOCAB", 101)))
    elif kind == "numpy":
        from .model_runner import NumpyPagedRunner
        runner = NumpyPagedRunner(os.environ["S2S_MODEL"], nb, bs)
    elif kind == "torch":
        from .model_runner import TorchRunner
        runner = TorchRunner(os.environ["S2S_MODEL"], nb, bs)
        if os.environ.get("S2S_CUDA_GRAPHS", "1") == "1":
            from .cuda_graph import capture_decode
            capture_decode(runner)
    else:
        raise ValueError(kind)
    eos = os.environ.get("S2S_EOS_ID")
    eng = LLMEngine(runner, nb, bs, sched, eos_id=int(eos) if eos else None)
    if kind == "torch" and getattr(runner, "graphs", None):
        eng.blocks._take()                    # reserve block 0: padded graph rows write there (cuda_graph.py)
    return eng


MODEL = os.environ.get("S2S_MODEL_NAME", "s2s-engine-v1")
app = FastAPI(title="s2s-engine v1")
reg = CollectorRegistry()
M_RUN = Gauge("vllm:num_requests_running", "running sequences", registry=reg)
M_WAIT = Gauge("vllm:num_requests_waiting", "waiting sequences", registry=reg)
M_KV = Gauge("vllm:kv_cache_usage_perc", "fraction of KV blocks in use", registry=reg)
M_PRE = Counter("vllm:num_preemptions_total", "preemptions", registry=reg)
M_GEN = Counter("vllm:generation_tokens_total", "generated tokens", registry=reg)
M_PROMPT = Counter("vllm:prompt_tokens_total", "prompt tokens", registry=reg)
M_TTFT = Histogram("vllm:time_to_first_token_seconds", "TTFT", registry=reg,
                   buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))
M_ITL = Histogram("vllm:inter_token_latency_seconds", "ITL", registry=reg,
                  buckets=(0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28))
M_E2E = Histogram("vllm:e2e_request_latency_seconds", "end-to-end latency", registry=reg,
                  buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60))

state: dict = {}


@app.on_event("startup")
async def _start() -> None:
    state["engine"] = build_engine()
    state["queues"] = {}                      # seq_id → asyncio.Queue of tokens (None = done)
    state["wake"] = asyncio.Event()
    state["lock"] = asyncio.Lock()            # engine state is touched by one step OR one add_request at a time
    state["loop"] = asyncio.create_task(_drive())


async def _drive() -> None:
    eng: LLMEngine = state["engine"]
    while True:
        if not eng.has_unfinished():
            state["wake"].clear()
            await state["wake"].wait()
        async with state["lock"]:
            before = {sid: len(s.output) for sid, s in eng.seqs.items()}
            pre = eng.stats.preemptions
            await asyncio.to_thread(eng.step) # model work off the event loop
        M_PRE.inc(eng.stats.preemptions - pre)
        for sid, n in before.items():
            s = eng.seqs[sid]
            q = state["queues"].get(sid)
            for t in s.output[n:]:
                M_GEN.inc()
                if q:
                    q.put_nowait(t)
            if s.finish_reason:
                if q:
                    q.put_nowait(None)
                eng.seqs.pop(sid, None)
        M_RUN.set(len(eng.scheduler.running))
        M_WAIT.set(len(eng.scheduler.waiting))
        M_KV.set(eng.blocks.utilization())
        await asyncio.sleep(0)


def _auth(req: Request) -> None:
    key = os.environ.get("S2S_API_KEY")
    if key and req.headers.get("authorization") != f"Bearer {key}":
        raise HTTPException(401, "unauthorized")


def _prompt_ids(body: dict) -> list[int]:
    if "messages" in body:
        text = " ".join(m.get("content", "") for m in body["messages"])
    else:
        text = body.get("prompt", "")
    if isinstance(text, list):
        return [int(t) for t in text]
    words = str(text).split()
    try:
        return [int(t) for t in words]
    except ValueError:
        # Not token ids: a stand-in "tokenizer" (one stable id per word) so #4 loadgen's word prompts work and
        # identical prefixes still map to identical ids (prefix caching stays measurable). Not a real tokenizer.
        V = _vocab()
        return [int.from_bytes(hashlib.sha256(w.encode()).digest()[:4], "little") % V for w in words]


def _vocab() -> int:
    r = state["engine"].runner
    return getattr(r, "vocab_size", None) or r.cfg.vocab_size


async def _handle(req: Request, chat: bool):
    _auth(req)
    body = await req.json()
    ids = _prompt_ids(body)
    params = SamplingParams(temperature=float(body.get("temperature", 0.0)), top_k=int(body.get("top_k", 0)),
                            top_p=float(body.get("top_p", 1.0)), min_p=float(body.get("min_p", 0.0)),
                            max_tokens=int(body.get("max_tokens", 16)), seed=body.get("seed"),
                            ignore_eos=bool(body.get("ignore_eos", False)))
    eng: LLMEngine = state["engine"]
    q: asyncio.Queue = asyncio.Queue()
    async with state["lock"]:
        try:
            seq = eng.add_request(ids, params, priority=int(body.get("priority", 0)))
        except (RequestTooLarge, ValueError) as e:
            raise HTTPException(400, str(e))
        state["queues"][seq.seq_id] = q
    state["wake"].set()
    M_PROMPT.inc(len(ids))
    rid, t0, kind = f"cmpl-{uuid.uuid4().hex[:16]}", time.monotonic(), "chat.completion" if chat else "text_completion"

    async def tokens():
        last = None
        try:
            while (t := await q.get()) is not None:
                now = time.monotonic()
                (M_TTFT if last is None else M_ITL).observe(now - (t0 if last is None else last))
                last = now
                yield t
        finally:
            state["queues"].pop(seq.seq_id, None)
            M_E2E.observe(time.monotonic() - t0)

    def chunk(text, finish=None):
        if chat:
            return {"id": rid, "object": "chat.completion.chunk", "model": MODEL,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]}
        return {"id": rid, "object": "text_completion", "model": MODEL,
                "choices": [{"index": 0, "text": text, "finish_reason": finish}]}

    if body.get("stream"):
        async def sse():
            async for t in tokens():
                yield f"data: {json.dumps(chunk(f'{t} '))}\n\n"
            yield f"data: {json.dumps(chunk('', seq.finish_reason))}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")
    out = [t async for t in tokens()]
    text = " ".join(map(str, out))
    usage = {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}
    choice = {"index": 0, "finish_reason": seq.finish_reason}
    choice.update({"message": {"role": "assistant", "content": text}} if chat else {"text": text})
    return JSONResponse({"id": rid, "object": kind, "created": int(time.time()), "model": MODEL,
                         "choices": [choice], "usage": usage})


@app.post("/v1/completions")
async def completions(req: Request):
    return await _handle(req, chat=False)


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    return await _handle(req, chat=True)


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "s2s"}]}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)
