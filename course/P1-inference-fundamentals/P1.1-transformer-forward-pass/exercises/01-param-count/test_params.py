import importlib.util
import math
from pathlib import Path

import pytest

from s2s.exercise import load_impl

m = load_impl(__file__, "params")


def test_matches_tensor_shapes(configs):
    spec = importlib.util.spec_from_file_location("p11_conftest", Path(__file__).resolve().parents[1] / "conftest.py")
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    weight_shapes = cf.weight_shapes
    for c in configs:
        assert m.count_params(c) == sum(math.prod(s) for s in weight_shapes(c).values())


def test_matches_hf_if_available(configs):
    transformers = pytest.importorskip("transformers")
    for c in configs[:2]:
        cfg = transformers.LlamaConfig(**c, max_position_embeddings=64)
        model = transformers.LlamaForCausalLM(cfg)
        assert m.count_params(c) == sum(p.numel() for p in model.parameters())
