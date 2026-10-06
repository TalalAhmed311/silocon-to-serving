import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "platform"))
from specdec import hf_models  # noqa: E402
from specdec.core import generate  # noqa: E402

if not hasattr(hf_models, "HFModelCached"):
    pytest.skip("exercise 3: add HFModelCached to platform/specdec/hf_models.py", allow_module_level=True)


def test_cached_equals_uncached_greedy():
    torch.manual_seed(0)
    mk = lambda L: transformers.LlamaForCausalLM(transformers.LlamaConfig(  # noqa: E731
        hidden_size=64, intermediate_size=128, num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=128, max_position_embeddings=256)).eval()
    t, d = mk(2), mk(1)
    a, _ = generate([1, 2, 3], hf_models.HFModel(d, 1e-4), hf_models.HFModel(t, 1e-4), 4, 24, seed=0)
    b, _ = generate([1, 2, 3], hf_models.HFModelCached(d, 1e-4), hf_models.HFModelCached(t, 1e-4), 4, 24, seed=0)
    assert a == b
