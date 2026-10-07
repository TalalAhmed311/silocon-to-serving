import os

os.environ["S2S_TRACING"] = "memory"           # must be set before the gateway's tracer is configured

import socket  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402
from pathlib import Path  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "platform"))
from gateway import tracing  # noqa: E402
from gateway.tracing import MEMORY_EXPORTER  # noqa: E402

tracing.configure("memory")   # independent of test order / earlier imports
from s2s.gatewaytest import running_gateway  # noqa: E402
from s2s.mockserver import running_mock  # noqa: E402


def test_one_trace_two_attempts():
    MEMORY_EXPORTER.clear()
    with running_mock(FAIL_RATE=1.0, TIME_SCALE=0.1) as bad, running_mock(TIME_SCALE=0.1) as good:
        with running_gateway([bad, good]) as gw:
            # the router's first choice is random among equals; send until we observe a retried request
            for _ in range(10):
                MEMORY_EXPORTER.clear()
                r = httpx.post(f"{gw}/v1/completions", headers={"Authorization": "Bearer k"},
                               json={"model": "mock-llama-8b", "prompt": "hi", "max_tokens": 2})
                assert r.status_code == 200
                spans = MEMORY_EXPORTER.get_finished_spans()
                attempts = [s for s in spans if s.name == "gateway.backend_attempt"]
                if len(attempts) == 2:
                    break
    root = next(s for s in spans if s.name == "gateway.request")
    assert {s.context.trace_id for s in spans} == {root.context.trace_id}
    assert all(a.parent.span_id == root.context.span_id for a in attempts)
    assert sorted(a.attributes["http.status_code"] for a in attempts) == [200, 503]


class _Capture(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):  # noqa: N802
        _Capture.seen.append(self.headers.get("traceparent"))
        self.rfile.read(int(self.headers.get("content-length", 0)))
        body = b'{"choices":[{"text":"x"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}'
        self.send_response(200); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_traceparent_reaches_backend():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    srv = HTTPServer(("127.0.0.1", port), _Capture)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with running_gateway([f"http://127.0.0.1:{port}"]) as gw:
        httpx.post(f"{gw}/v1/completions", headers={"Authorization": "Bearer k"},
                   json={"model": "mock-llama-8b", "prompt": "hi", "max_tokens": 1})
    srv.shutdown()
    assert _Capture.seen and _Capture.seen[-1] and _Capture.seen[-1].startswith("00-")


def test_stream_spans():
    pytest.skip("exercise 3: add gateway.stream_attempt spans to Gateway._stream, then delete this line")
