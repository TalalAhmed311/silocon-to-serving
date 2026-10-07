import json
import os

import pytest

from s2s.exercise import load_impl
from tenancy import audit as ref

verify = ref.verify if os.environ.get("S2S_SOLUTIONS") == "1" else load_impl(__file__, "audit_impl").verify
KEY = b"test-only-hmac-key"


@pytest.fixture
def log(tmp_path):
    p = tmp_path / "audit.jsonl"
    clock = iter(range(1000))
    a = ref.AuditLog(p, key=KEY, clock=lambda: next(clock))
    for i in range(20):
        a.append(tenant=f"t{i % 3}", action="completion", model="m", outcome="ok", tokens=i)
    return p, a.last


def lines(p):
    return p.read_text().splitlines()


def write(p, ls):
    p.write_text("\n".join(ls) + "\n")


def test_intact(log):
    p, last = log
    assert verify(p, KEY) == []
    assert verify(p, KEY, checkpoint=last) == []


def test_edit(log):
    p, _ = log
    ls = lines(p)
    e = json.loads(ls[6]); e["rec"]["tokens"] = 9999; ls[6] = json.dumps(e)  # noqa: E702
    write(p, ls)
    probs = verify(p, KEY)
    assert probs and "line 7" in probs[0]


def test_delete(log):
    p, _ = log
    ls = lines(p)
    del ls[6]
    write(p, ls)
    probs = verify(p, KEY)
    assert probs and "line 7" in probs[0]


def test_swap(log):
    p, _ = log
    ls = lines(p)
    ls[6], ls[7] = ls[7], ls[6]
    write(p, ls)
    probs = verify(p, KEY)
    assert probs and "line 7" in probs[0]


def test_truncate_needs_checkpoint(log):
    p, last = log
    write(p, lines(p)[:15])
    assert verify(p, KEY) == []
    assert verify(p, KEY, checkpoint=last)


def test_wrong_key(log):
    p, _ = log
    q = p.with_name("forged.jsonl")
    clock = iter(range(1000))
    a = ref.AuditLog(q, key=b"attacker-key", clock=lambda: next(clock))
    a.append(tenant="t0", action="completion", model="m", outcome="ok", tokens=0)
    probs = verify(q, KEY)
    assert probs and "line 1" in probs[0]


def test_reopen_continues_chain(log):
    p, last = log
    a = ref.AuditLog(p, key=KEY)
    assert a.last == last
    a.append(tenant="t0", action="rotate_key", outcome="ok")
    assert verify(p, KEY) == []
