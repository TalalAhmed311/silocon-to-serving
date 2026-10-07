import itertools

import httpx
import pytest

from s2s.gatewaytest import running_gateway
from s2s.mockserver import running_mock
from tenancy import keys

REQ = {"model": "mock-llama-8b", "prompt": "hi", "max_tokens": 2}


def call(gw, key):
    return httpx.post(f"{gw}/v1/completions", headers={"Authorization": f"Bearer {key}"}, json=REQ, timeout=10).status_code


def test_generate_and_rotate_config():
    old, new = keys.generate(), keys.generate()
    assert old.startswith("s2s_") and len(old) > 40 and old != new
    t0 = {"name": "a", "key_sha256": keys.key_hash(old)}
    t1 = keys.start_rotation(t0, new)
    assert t1["key_sha256"] == keys.key_hash(new) and t1["previous_key_sha256"] == [keys.key_hash(old)]
    assert keys.finish_rotation(t1)["previous_key_sha256"] == []
    assert t0 == {"name": "a", "key_sha256": keys.key_hash(old)}          # inputs not mutated


def test_both_keys_work_mid_rotation_and_old_fails_after():
    old, new = keys.generate(), keys.generate()
    with running_mock(TIME_SCALE=0.05) as be:
        with running_gateway([be], tenants=[{"name": "a", "key": new, "previous_key_sha256": [keys.key_hash(old)]}]) as gw:
            assert call(gw, old) == 200
            assert call(gw, new) == 200
            assert call(gw, "s2s_wrong") == 401
        with running_gateway([be], tenants=[{"name": "a", "key": new}]) as gw:
            assert call(gw, new) == 200
            assert call(gw, old) == 401


@pytest.mark.xfail(reason="exercise 2, task 2", strict=False)
def test_no_failed_requests_during_rollout():
    old, new = keys.generate(), keys.generate()
    with running_mock(TIME_SCALE=0.05) as be:
        with running_gateway([be], tenants=[{"name": "a", "key": old}]) as gw_a, \
             running_gateway([be], tenants=[{"name": "a", "key": new, "previous_key_sha256": [keys.key_hash(old)]}]) as gw_b:
            gws = itertools.cycle([gw_a, gw_b])
            codes = [call(next(gws), old if i < 20 else new) for i in range(40)]
    raise NotImplementedError("TODO: assert on `codes`, then explain why half of the new-key calls still fail "
                              "and what the rollout order must be (03: phase 1 must complete before handing out `new`)")
