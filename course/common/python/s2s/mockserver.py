"""Run platform/mockllm on a free localhost port in a background thread (for T0 tests; nothing leaves the machine).

    from s2s.mockserver import running_mock
    with running_mock(TIME_SCALE="0.1", MAX_NUM_SEQS="8") as url:
        ...  # url = "http://127.0.0.1:<port>"

MOCKLLM_* settings are read when mockllm.server is imported, so each call imports a fresh copy of the module.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import socket
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


@contextlib.contextmanager
def running_mock(**settings: str):
    import uvicorn

    old = {k: os.environ.get(f"MOCKLLM_{k}") for k in settings}
    os.environ.update({f"MOCKLLM_{k}": str(v) for k, v in settings.items()})
    try:
        spec = importlib.util.spec_from_file_location(f"mockllm_server_{id(settings)}", ROOT / "platform/mockllm/server.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(f"MOCKLLM_{k}", None)
            else:
                os.environ[f"MOCKLLM_{k}"] = v
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(mod.app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.025)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        th.join(timeout=5)
