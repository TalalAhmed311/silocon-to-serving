"""Shared fixtures for P1.1 exercises: random Llama configs and a weight-shape generator (no weights allocated)."""
import random

import pytest


def random_configs(n: int = 4, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        H = rng.choice([4, 8, 16])
        Hkv = rng.choice([h for h in (1, 2, 4, 8, 16) if H % h == 0 and h <= H])
        h = rng.choice([16, 32, 64])
        out.append(dict(hidden_size=H * h, intermediate_size=rng.choice([2, 3, 4]) * H * h + rng.choice([0, 32]),
                        num_hidden_layers=rng.randint(1, 6), num_attention_heads=H, num_key_value_heads=Hkv,
                        vocab_size=rng.choice([256, 1000, 32000]), tie_word_embeddings=rng.random() < 0.5))
    return out


def weight_shapes(c: dict) -> dict[str, tuple]:
    """The tensors a Llama checkpoint with config c contains (HF names), as shapes only."""
    d, H, Hkv, V = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["vocab_size"]
    h, f = d // H, c["intermediate_size"]
    s = {"model.embed_tokens.weight": (V, d), "model.norm.weight": (d,)}
    for l in range(c["num_hidden_layers"]):
        p = f"model.layers.{l}."
        s.update({p + "input_layernorm.weight": (d,), p + "post_attention_layernorm.weight": (d,),
                  p + "self_attn.q_proj.weight": (H * h, d), p + "self_attn.k_proj.weight": (Hkv * h, d),
                  p + "self_attn.v_proj.weight": (Hkv * h, d), p + "self_attn.o_proj.weight": (d, H * h),
                  p + "mlp.gate_proj.weight": (f, d), p + "mlp.up_proj.weight": (f, d), p + "mlp.down_proj.weight": (d, f)})
    if not c["tie_word_embeddings"]:
        s["lm_head.weight"] = (V, d)
    return s


@pytest.fixture
def configs():
    return random_configs()
