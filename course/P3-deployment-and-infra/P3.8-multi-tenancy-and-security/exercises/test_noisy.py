import concurrent.futures as cf
import statistics
import time

import httpx
import pytest

from s2s.gatewaytest import running_gateway
from s2s.mockserver import running_mock

MOCK = dict(TIME_SCALE=0.1, MAX_NUM_SEQS=4)


def req(gw, key, max_tokens=32):
    t = time.perf_counter()
    r = httpx.post(f"{gw}/v1/completions", headers={"Authorization": f"Bearer {key}"},
                   json={"model": "mock-llama-8b", "prompt": "hi", "max_tokens": max_tokens}, timeout=120)
    return r.status_code, time.perf_counter() - t


def victim(gw, n=15):
    out = [req(gw, "kb") for _ in range(n)]
    assert all(c == 200 for c, _ in out), out
    return statistics.quantiles([s for _, s in out], n=10)[-1]      # p90


def run(flood_tokens):
    tenants = [{"name": "a", "key": "ka", "rps": 5, "burst": 5}, {"name": "b", "key": "kb", "rps": 100, "burst": 100}]
    with running_mock(**MOCK) as be, running_gateway([be], max_attempts=1, tenants=tenants) as gw:
        base = victim(gw)
        with cf.ThreadPoolExecutor(100) as ex:
            flood = [ex.submit(req, gw, "ka", flood_tokens) for _ in range(100)]
            time.sleep(0.2)
            loaded = victim(gw)
            codes = [f.result()[0] for f in flood]
    return base, loaded, codes


def test_rate_limit_protects_b():
    base, loaded, codes = run(flood_tokens=32)
    assert codes.count(429) >= 90, codes.count(429)
    assert loaded < 3 * base + 0.05, (base, loaded)


@pytest.mark.xfail(reason="exercise 4: long outputs within the rps limit", strict=False)
def test_long_outputs_within_rps():
    base, loaded, codes = run(flood_tokens=2000)
    assert loaded < 3 * base + 0.05, (base, loaded)
