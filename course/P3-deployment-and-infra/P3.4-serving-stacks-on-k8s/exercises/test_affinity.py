import inspect
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "platform"))
from gateway.config import Backend  # noqa: E402
from gateway.router import Router  # noqa: E402

if "session_id" not in inspect.signature(Router.candidates).parameters:
    pytest.skip("exercise 5: add session_id to Router.candidates (rendezvous hashing)", allow_module_level=True)

B = [Backend(name=n, url=f"http://{n}", models=["m"]) for n in ("a", "b", "c")]


def first(r, s):
    return r.candidates("m", session_id=s)[0].name


def test_sticky_and_spread():
    r = Router(B)
    owners = {s: first(r, s) for s in (f"s{i}" for i in range(100))}
    assert all(first(r, s) == o for s, o in owners.items())
    assert max(Counter(owners.values()).values()) <= 60


def test_only_failed_backends_sessions_move():
    r = Router(B)
    before = {s: first(r, s) for s in (f"s{i}" for i in range(100))}
    r.mark(B[0], False)
    after = {s: first(r, s) for s in before}
    moved = {s for s in before if before[s] != after[s]}
    assert moved == {s for s, o in before.items() if o == "a"}
