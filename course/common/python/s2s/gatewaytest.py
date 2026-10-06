"""Run platform/gateway in front of given backend URLs on a free localhost port (T0 tests)."""
import contextlib
import hashlib
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


@contextlib.contextmanager
def running_gateway(backend_urls: list[str], max_attempts: int = 3, **tenant):
    import uvicorn
    import yaml

    sys.path.insert(0, str(ROOT / "platform"))
    from gateway.app import build_app
    from gateway.config import Config

    cfgd = {"max_attempts": max_attempts, "retry_budget_ratio": 1.0,
            "backends": [{"name": f"b{i}", "url": u, "models": ["mock-llama-8b"]} for i, u in enumerate(backend_urls)],
            "tenants": [{"name": "t", "key_sha256": hashlib.sha256(b"k").hexdigest(), "rps": 1e6, "burst": 1e6, **tenant}]}
    p = Path(tempfile.mkdtemp()) / "gw.yaml"
    p.write_text(yaml.safe_dump(cfgd))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(build_app(Config.load(p)), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    while not srv.started:
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.should_exit = True
        th.join(timeout=5)
