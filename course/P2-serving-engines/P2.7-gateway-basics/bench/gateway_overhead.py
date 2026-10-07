"""gateway_overhead.py — latency the gateway adds over calling the backend directly (localhost, mock backend).

Run:      uv run python course/P2-serving-engines/P2.7-gateway-basics/bench/gateway_overhead.py [--n 300]
Method:   starts one mockllm and the gateway in-process on free ports; 300 sequential non-streaming 1-token requests
          each way (after 20 warm-up); reports p50/p99 and the difference. Hardware: T0.
"""
import argparse
import hashlib
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import uvicorn
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "platform"))
sys.path.insert(0, str(ROOT / "course/common/python"))
from gateway.app import build_app  # noqa: E402
from gateway.config import Config  # noqa: E402
from s2s.mockserver import running_mock  # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def timed(url, headers, n):
    out = []
    with httpx.Client() as c:
        for i in range(n + 20):
            t0 = time.perf_counter()
            c.post(f"{url}/v1/completions", headers=headers, json={"model": "mock-llama-8b", "prompt": "x", "max_tokens": 1}).raise_for_status()
            if i >= 20:
                out.append(time.perf_counter() - t0)
    return np.array(out) * 1e3


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()
with running_mock(BASE_STEP_MS=1, PER_SEQ_MS=0) as backend:
    cfgd = {"backends": [{"name": "m", "url": backend, "models": ["mock-llama-8b"]}],
            "tenants": [{"name": "t", "key_sha256": hashlib.sha256(b"k").hexdigest(), "rps": 1e6, "burst": 1e6}]}
    p = Path(tempfile.mkdtemp()) / "gw.yaml"
    p.write_text(yaml.safe_dump(cfgd))
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(build_app(Config.load(p)), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.01)
    direct = timed(backend, {}, a.n)
    via = timed(f"http://127.0.0.1:{port}", {"Authorization": "Bearer k"}, a.n)
    srv.should_exit = True
print("| path | p50 ms | p99 ms | added p50 ms |\n|---|---|---|---|")
print(f"| direct to backend | {np.median(direct):.2f} | {np.percentile(direct, 99):.2f} | — |")
print(f"| through gateway | {np.median(via):.2f} | {np.percentile(via, 99):.2f} | {np.median(via) - np.median(direct):.2f} |")
