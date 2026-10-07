import asyncio
import importlib
import os
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from s2s.exercise import load_impl

m = load_impl(__file__, "client")
ROOT = Path(__file__).resolve().parents[5]


@pytest.fixture(scope="module")
def server():
    uvicorn = pytest.importorskip("uvicorn")
    os.environ["MOCKLLM_TIME_SCALE"] = "0.2"          # 5x faster steps: keeps the test quick
    sys.path.insert(0, str(ROOT / "platform"))
    srv_mod = importlib.import_module("mockllm.server")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(srv_mod.app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    th.join(timeout=5)


def server_ttft_sum(url):
    for line in httpx.get(f"{url}/metrics").text.splitlines():
        if line.startswith("vllm:time_to_first_token_seconds_sum"):
            return float(line.split()[-1])
    return 0.0


def test_stream(server):
    before = server_ttft_sum(server)

    async def go():
        async with httpx.AsyncClient() as c:
            return await m.stream_completion(c, server, "one two three four", 12)
    r = asyncio.run(go())
    server_ttft = server_ttft_sum(server) - before   # this request's server-side TTFT
    assert r["n_tokens"] == 12
    assert r["usage"]["completion_tokens"] == 12
    assert r["ttft"] >= server_ttft > 0
    assert all(g > 0 for g in r["itl"])
    assert sum(r["itl"]) == pytest.approx(r["e2e"] - r["ttft"], rel=1e-6)
