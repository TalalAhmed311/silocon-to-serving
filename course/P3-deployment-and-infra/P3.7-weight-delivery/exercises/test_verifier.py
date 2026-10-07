import os
from pathlib import Path

import pytest

from s2s.exercise import load_impl
from weights import manifest as ref

if os.environ.get("S2S_SOLUTIONS") == "1":
    verify = ref.verify
else:
    verify = load_impl(__file__, "verify_impl").verify


@pytest.fixture
def model(tmp_path: Path):
    (tmp_path / "config.json").write_text('{"dim": 8}')
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(os.urandom(4096))
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(os.urandom(1000))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "tokenizer.json").write_text("{}")
    return tmp_path, ref.build(tmp_path, "tiny", "v1")


def test_clean(model):
    root, man = model
    assert man["total_bytes"] == 4096 + 1000 + len('{"dim": 8}') + 2
    assert verify(root, man) == []


def test_missing(model):
    root, man = model
    (root / "sub" / "tokenizer.json").unlink()
    assert verify(root, man) == ["missing: sub/tokenizer.json"]


def test_truncated_is_size_error(model):
    root, man = model
    p = root / "model-00002-of-00002.safetensors"
    p.write_bytes(p.read_bytes()[:-1])
    (probs,) = verify(root, man)
    assert probs.startswith("size: model-00002-of-00002.safetensors")


def test_bitflip_is_hash_error(model):
    root, man = model
    p = root / "model-00001-of-00002.safetensors"
    b = bytearray(p.read_bytes())
    b[100] ^= 1
    p.write_bytes(bytes(b))
    assert verify(root, man) == ["sha256: model-00001-of-00002.safetensors"]


def test_unexpected(model):
    root, man = model
    (root / "evil.py").write_text("import os")
    (root / "manifest.json").write_text("{}")       # the manifest itself is never "unexpected"
    assert verify(root, man) == ["unexpected: evil.py"]
