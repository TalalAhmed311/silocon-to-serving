"""app.py — #6: the gateway service (FastAPI). OpenAI-compatible in, OpenAI-compatible out.

Request path: authenticate tenant (Bearer key, compared by SHA-256) -> per-tenant rate limit (429 + Retry-After) ->
token-budget check with an estimate -> pick backends for the model -> try them in order with retries (only before the
first streamed byte) -> stream through -> commit actual usage from the final `usage` chunk -> metrics.

Run:  GATEWAY_CONFIG=platform/gateway/example.yaml uv run uvicorn gateway.app:app --app-dir platform --port 9000
Hardware: T0 (against mockllm backends) / anywhere.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import random
import time

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from .config import Config, key_hash
from .ratelimit import TokenBucket, TokenBudget
from .retry import RetryBudget, backoff_s, should_retry
from .router import Router
from .tracing import inject, tracer


class Gateway:
    def __init__(self, cfg: Config, clock=time.monotonic):
        self.cfg = cfg
        self.router = Router(cfg.backends, cfg.routing)
        self.tenants = {t.key_sha256: t for t in cfg.tenants}
        self.buckets = {t.name: TokenBucket(t.rps, t.burst, clock) for t in cfg.tenants}
        self.budgets = {t.name: TokenBudget(t.tokens_per_window, t.window_s, clock) for t in cfg.tenants}
        self.retry_budget = RetryBudget(cfg.retry_budget_ratio)
        self.client = httpx.AsyncClient(limits=httpx.Limits(max_connections=None, max_keepalive_connections=100))
        self.rng = random.Random(0)
        r = self.registry = CollectorRegistry()
        self.m_req = Counter("gateway_requests", "Requests", ["tenant", "model", "backend", "outcome"], registry=r)
        self.m_tok = Counter("gateway_tokens", "LLM tokens", ["tenant", "model", "backend", "kind"], registry=r)
        self.m_lat = Histogram("gateway_request_seconds", "End-to-end latency through the gateway", ["backend"], registry=r)
        self.m_ttft = Histogram("gateway_ttft_seconds", "Time to first byte from the backend", ["backend"], registry=r)
        self.m_retry = Counter("gateway_retries", "Retries and fallbacks", ["reason"], registry=r)
        self.m_inflight = Gauge("gateway_inflight", "In-flight requests per backend", ["backend"], registry=r)

    # ---- auth / limits ----
    def tenant_for(self, req: Request):
        auth = req.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            raise HTTPException(401, "missing API key")
        h = key_hash(auth[7:])
        for stored, t in self.tenants.items():
            if hmac.compare_digest(stored, h):            # constant-time compare
                return t
        raise HTTPException(401, "invalid API key")

    @staticmethod
    def estimate_tokens(body: dict) -> int:
        text = body.get("prompt") or " ".join(str(m.get("content", "")) for m in body.get("messages", []))
        return len(str(text)) // 4 + int(body.get("max_tokens", 256))   # ~4 chars/token heuristic + output cap

    # ---- proxying ----
    async def forward(self, path: str, req: Request):
        with tracer().start_as_current_span("gateway.request") as span:
            resp = await self._forward(path, req, span)
            span.set_attribute("http.status_code", getattr(resp, "status_code", 200))
            return resp

    async def _forward(self, path: str, req: Request, span):
        tenant = self.tenant_for(req)
        body = await req.json()
        model = body.get("model", "")
        span.set_attribute("s2s.tenant", tenant.name)
        span.set_attribute("s2s.model", model)
        span.set_attribute("s2s.stream", bool(body.get("stream")))
        if "*" not in tenant.models and model not in tenant.models:
            raise HTTPException(403, f"tenant {tenant.name} may not use model {model}")
        if not self.buckets[tenant.name].allow():
            ra = self.buckets[tenant.name].retry_after()
            return JSONResponse({"error": {"message": "rate limit exceeded", "type": "rate_limit"}}, status_code=429,
                                headers={"Retry-After": f"{max(1, round(ra))}"})
        est = self.estimate_tokens(body)
        if not self.budgets[tenant.name].check(est):
            return JSONResponse({"error": {"message": "token budget exhausted", "type": "budget"}}, status_code=429)
        backends = self.router.candidates(model)
        if not backends:
            raise HTTPException(404, f"no backend serves model {model}")
        if body.get("stream"):
            body.setdefault("stream_options", {"include_usage": True})   # so we can bill actual usage
        self.retry_budget.on_request()
        return await (self._stream if body.get("stream") else self._unary)(path, body, tenant, model, backends)

    def _headers(self, b):
        h = {"Authorization": f"Bearer {b.api_key}"} if b.api_key else {}
        return inject(h)          # W3C traceparent: backend-side spans join this trace

    async def _unary(self, path, body, tenant, model, backends):
        attempt, last = 0, None
        for b in self._attempt_order(backends):
            t0 = time.perf_counter()
            self.router.acquire(b)
            self.m_inflight.labels(b.name).inc()
            with tracer().start_as_current_span("gateway.backend_attempt") as aspan:
                aspan.set_attribute("s2s.backend", b.name)
                aspan.set_attribute("s2s.attempt", attempt)
                try:
                    r = await self.client.post(f"{b.url}{path}", json=body, headers=self._headers(b), timeout=b.timeout_s)
                    status, exc = r.status_code, None
                except httpx.HTTPError as e:
                    r, status, exc = None, None, e
                finally:
                    self.router.release(b)
                    self.m_inflight.labels(b.name).dec()
                aspan.set_attribute("http.status_code", status or 0)
                if r is not None and status == 200:
                    u = r.json().get("usage", {})
                    aspan.set_attribute("s2s.prompt_tokens", int(u.get("prompt_tokens", 0)))
                    aspan.set_attribute("s2s.completion_tokens", int(u.get("completion_tokens", 0)))
            if r is not None and status == 200:
                self.router.mark(b, True)
                self.m_lat.labels(b.name).observe(time.perf_counter() - t0)
                usage = r.json().get("usage", {})
                self._bill(tenant, model, usage, b.name)
                self.m_req.labels(tenant.name, model, b.name, "ok").inc()
                return JSONResponse(r.json())
            last = (status, str(exc) if exc else (r.text[:200] if r is not None else ""))
            self.router.mark(b, False) if (exc or (status and status >= 500)) else None
            self.m_req.labels(tenant.name, model, b.name, f"err_{status or type(exc).__name__}").inc()
            if not should_retry(status, exc, False, attempt, self.cfg.max_attempts, self.retry_budget):
                break
            self.m_retry.labels("unary").inc()
            await asyncio.sleep(backoff_s(attempt, rng=self.rng))
            attempt += 1
        return JSONResponse({"error": {"message": f"all backends failed: {last}", "type": "upstream"}}, status_code=502)

    def _attempt_order(self, backends):
        """Fallback order: best backend first, then the others; cycle if max_attempts > number of backends."""
        for i in range(self.cfg.max_attempts):
            yield backends[i % len(backends)]

    async def _stream(self, path, body, tenant, model, backends):
        gw = self

        async def gen():
            attempt, started = 0, False
            for b in gw._attempt_order(backends):
                t0 = time.perf_counter()
                gw.router.acquire(b)
                gw.m_inflight.labels(b.name).inc()
                usage, status, exc = None, None, None
                try:
                    async with gw.client.stream("POST", f"{b.url}{path}", json=body, headers=gw._headers(b), timeout=b.timeout_s) as r:
                        status = r.status_code
                        if status == 200:
                            async for line in r.aiter_lines():
                                if not line:
                                    continue
                                if not started:
                                    gw.m_ttft.labels(b.name).observe(time.perf_counter() - t0)
                                started = True                      # from here on, no retries: bytes reached the client
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    ev = json.loads(line[6:])
                                    usage = ev.get("usage") or usage
                                yield line + "\n\n"
                            gw.router.mark(b, True)
                            gw._bill(tenant, model, usage or {}, b.name)
                            gw.m_req.labels(tenant.name, model, b.name, "ok").inc()
                            gw.m_lat.labels(b.name).observe(time.perf_counter() - t0)
                            return
                except httpx.HTTPError as e:
                    exc = e
                finally:
                    gw.router.release(b)
                    gw.m_inflight.labels(b.name).dec()
                gw.router.mark(b, False)
                gw.m_req.labels(tenant.name, model, b.name, f"err_{status or type(exc).__name__}").inc()
                if started:
                    # Mid-stream failure: we cannot retry transparently. Tell the client explicitly and stop.
                    err = {"error": {"message": "upstream stream interrupted", "type": "upstream_interrupted"}}
                    yield f"data: {json.dumps(err)}\n\n"
                    gw._bill(tenant, model, usage or {}, b.name)
                    return
                if not should_retry(status, exc, started, attempt, gw.cfg.max_attempts, gw.retry_budget):
                    break
                gw.m_retry.labels("stream").inc()
                await asyncio.sleep(backoff_s(attempt, rng=gw.rng))
                attempt += 1
            yield f"data: {json.dumps({'error': {'message': 'all backends failed', 'type': 'upstream'}})}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    def _bill(self, tenant, model, usage: dict, backend: str) -> None:
        p, c = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        self.budgets[tenant.name].commit(p + c)
        self.m_tok.labels(tenant.name, model, backend, "prompt").inc(p)
        self.m_tok.labels(tenant.name, model, backend, "completion").inc(c)


def build_app(cfg: Config, clock=time.monotonic) -> FastAPI:
    gw = Gateway(cfg, clock)
    app = FastAPI(title="s2s-gateway")
    app.state.gw = gw

    @app.post("/v1/completions")
    async def completions(req: Request):
        return await gw.forward("/v1/completions", req)

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        return await gw.forward("/v1/chat/completions", req)

    @app.get("/v1/models")
    async def models(req: Request):
        gw.tenant_for(req)
        names = sorted({m for b in cfg.backends for m in b.models})
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "s2s-gateway"} for m in names]}

    @app.get("/health")
    async def health():
        return {"status": "ok", "backends": gw.router.healthy}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(generate_latest(gw.registry).decode(), media_type=CONTENT_TYPE_LATEST)

    return app


app = build_app(Config.load(os.environ["GATEWAY_CONFIG"])) if os.environ.get("GATEWAY_CONFIG") else None
