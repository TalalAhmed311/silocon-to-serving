"""OpenAI-compatible server over the fake model (P6.7). T0."""
import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from s2s_engine import server  # noqa: E402
from s2s_engine.model_runner import FakeRunner  # noqa: E402


def greedy(prompt, n):
    r, toks = FakeRunner(), list(prompt)
    for _ in range(n):
        toks.append(int(np.argmax(r.logits_for(toks))))
    return toks[len(prompt):]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("S2S_RUNNER", "fake")
    monkeypatch.delenv("S2S_API_KEY", raising=False)
    with TestClient(server.app) as c:
        yield c


def test_completion_matches_greedy_reference(client):
    r = client.post("/v1/completions", json={"prompt": "5 17 3", "max_tokens": 6})
    assert r.status_code == 200
    body = r.json()
    assert [int(t) for t in body["choices"][0]["text"].split()] == greedy([5, 17, 3], 6)
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 6, "total_tokens": 9}
    assert body["choices"][0]["finish_reason"] == "length"


def test_streaming_and_metrics(client):
    with client.stream("POST", "/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "hello world"}], "max_tokens": 4, "stream": True}) as r:
        lines = [l for l in r.iter_lines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines[:-1]]
    assert sum(1 for c in chunks if c["choices"][0]["delta"]["content"].strip()) == 4
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    m = client.get("/metrics").text
    assert "vllm:generation_tokens_total" in m and "vllm:time_to_first_token_seconds" in m


def test_auth(client, monkeypatch):
    monkeypatch.setenv("S2S_API_KEY", "k")
    assert client.post("/v1/completions", json={"prompt": "1", "max_tokens": 1}).status_code == 401
    ok = client.post("/v1/completions", json={"prompt": "1", "max_tokens": 1}, headers={"Authorization": "Bearer k"})
    assert ok.status_code == 200


def test_too_large_is_400(client):
    r = client.post("/v1/completions", json={"prompt": " ".join(["1"] * 10), "max_tokens": 10_000_000})
    assert r.status_code == 400
