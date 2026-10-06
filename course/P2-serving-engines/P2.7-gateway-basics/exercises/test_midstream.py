import json

import httpx

from s2s.gatewaytest import running_gateway
from s2s.mockserver import running_mock


def stream_events(gw):
    events = []
    with httpx.Client() as c:
        with c.stream("POST", f"{gw}/v1/completions", headers={"Authorization": "Bearer k"}, timeout=30,
                      json={"model": "mock-llama-8b", "prompt": "hi", "max_tokens": 10, "stream": True}) as r:
            for line in r.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    events.append(json.loads(line[6:]))
    return events


def test_midstream_failure_is_explicit_not_spliced():
    with running_mock(FAIL_MIDSTREAM_RATE=1.0, TIME_SCALE=0.1) as cut, running_mock(TIME_SCALE=0.1) as healthy:
        with running_gateway([cut, healthy], max_attempts=2) as gw:
            ev = stream_events(gw)
    texts = [e["choices"][0]["text"] for e in ev if "choices" in e and e["choices"][0].get("text")]
    errors = [e for e in ev if "error" in e]
    # The gateway may have routed this request to either backend first (least-outstanding, random tie-break):
    if errors:
        assert len(texts) == 3 and errors[-1]["error"]["type"] == "upstream_interrupted"
    else:
        assert len(texts) == 10               # served entirely by the healthy backend


def test_failure_before_first_byte_is_retried():
    with running_mock(FAIL_RATE=1.0, TIME_SCALE=0.1) as bad, running_mock(TIME_SCALE=0.1) as good:
        with running_gateway([bad, good]) as gw:
            for _ in range(10):
                ev = stream_events(gw)
                assert not any("error" in e for e in ev)
                assert sum(1 for e in ev if "choices" in e and e["choices"][0].get("text")) == 10
