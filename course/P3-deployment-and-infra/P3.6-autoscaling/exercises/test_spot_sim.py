"""P3.6 exercise 4 (T0 simulation): a spot node is reclaimed under load; no client request may fail.

Two mock backends behind the #6 gateway. Mid-run, backend b1 is "interrupted": uvicorn's graceful shutdown
stops accepting connections and finishes in-flight requests, the same as preStop + drain within the 2-min notice.
New attempts to b1 get connection refused before any byte is sent, so the gateway retries them on b0.

test_drain_then_kill: passes as shipped (that is the baseline).
test_hard_kill_streams: your task. A hard kill (no drain) breaks in-flight streams; make the gateway / client policy
keep failures within the stated budget, then remove the xfail.
"""
import concurrent.futures as cf
import contextlib
import threading
import time

import httpx
import pytest

from s2s.gatewaytest import running_gateway
from s2s.mockserver import running_mock

REQ = {"model": "mock-llama-8b", "prompt": "hi", "max_tokens": 16}


def fire(gw, n, spacing_s):
    def one(i):
        time.sleep(i * spacing_s)
        try:
            r = httpx.post(f"{gw}/v1/completions", headers={"Authorization": "Bearer k"}, json=REQ, timeout=30)
            return r.status_code
        except httpx.HTTPError as e:
            return repr(e)
    with cf.ThreadPoolExecutor(32) as ex:
        return list(ex.map(one, range(n)))


def test_drain_then_kill():
    with running_mock(TIME_SCALE=0.1) as a, contextlib.ExitStack() as spot:
        b = spot.enter_context(running_mock(TIME_SCALE=0.1))
        with running_gateway([a, b], max_attempts=3) as gw:
            killer = threading.Timer(1.0, spot.close)       # the interruption, 1 s into the run
            killer.start()
            codes = fire(gw, n=120, spacing_s=0.02)
            killer.join()
    failed = [c for c in codes if c != 200]
    assert not failed, f"{len(failed)} failed: {failed[:5]}"


@pytest.mark.xfail(reason="exercise 4: hard kill of in-flight streams; see 04-spot.md", strict=False)
def test_hard_kill_streams():
    pytest.fail("TODO: implement the hard-kill scenario and your mitigation (04-spot.md)")
