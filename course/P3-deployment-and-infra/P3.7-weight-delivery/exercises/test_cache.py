"""WeightCache end to end against a DirStore (reference implementation)."""
import json
import os

import pytest

from weights import cache as c
from weights import manifest


class Clock:
    t = 0.0

    def __call__(self):
        self.t += 1
        return self.t


def publish(reg, model, version, nbytes):
    d = reg / model / version
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(os.urandom(nbytes))
    (d / "manifest.json").write_text(json.dumps(manifest.build(d, model, version)))


@pytest.fixture
def env(tmp_path):
    reg = tmp_path / "registry"
    for v, n in (("v1", 400), ("v2", 400), ("v3", 400)):
        publish(reg, "m", v, n)
    return reg, tmp_path / "nvme"


def test_miss_then_hit(env):
    reg, nvme = env
    wc = c.WeightCache(nvme, 1000, c.DirStore(reg), clock=Clock())
    p = wc.get("m", "v1")
    assert (p / "model.safetensors").stat().st_size == 400
    assert wc.get("m", "v1") == p
    assert (wc.misses, wc.hits) == (1, 1)
    assert not any((nvme / ".tmp").iterdir())                  # temp dir cleaned up by the rename


def test_lru_eviction_and_pins(env):
    reg, nvme = env
    wc = c.WeightCache(nvme, 900, c.DirStore(reg), clock=Clock())   # room for two 400 B (+ small manifest) versions
    wc.get("m", "v1"); wc.get("m", "v2")                           # noqa: E702
    wc.get("m", "v1")                                              # v1 is now most recent
    wc.get("m", "v3")                                              # evicts v2
    assert {p.name for p in (nvme / "m").iterdir()} == {"v1", "v3"}
    wc.pin("m", "v1"); wc.pin("m", "v3")                           # noqa: E702
    with pytest.raises(c.CacheFull):
        wc.get("m", "v2")


def test_corrupt_registry_is_rejected(env):
    reg, nvme = env
    (reg / "m" / "v2" / "model.safetensors").write_bytes(os.urandom(400))   # same size, different bytes
    wc = c.WeightCache(nvme, 1000, c.DirStore(reg))
    with pytest.raises(IOError):
        wc.get("m", "v2")
    assert not (nvme / "m" / "v2").exists()


def test_recency_survives_restart(env):
    reg, nvme = env
    clk = Clock()
    wc = c.WeightCache(nvme, 900, c.DirStore(reg), clock=clk)
    wc.get("m", "v1"); wc.get("m", "v2"); wc.get("m", "v1")        # noqa: E702
    wc2 = c.WeightCache(nvme, 900, c.DirStore(reg), clock=clk)     # "pod restart"
    assert wc2.get("m", "v1") and wc2.hits == 1
    wc2.get("m", "v3")
    assert not (nvme / "m" / "v2").exists()
