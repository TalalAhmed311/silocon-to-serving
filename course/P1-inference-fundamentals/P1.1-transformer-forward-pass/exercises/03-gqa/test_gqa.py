from s2s.exercise import load_impl

m = load_impl(__file__, "gqa")
BASE = dict(hidden_size=4096, num_hidden_layers=32, num_attention_heads=32)


def test_mha():
    assert m.kv_bytes_per_token({**BASE, "num_key_value_heads": 32}) == 524_288
    assert m.kv_saving_vs_mha({**BASE, "num_key_value_heads": 32}) == 1


def test_gqa_8():
    assert m.kv_bytes_per_token({**BASE, "num_key_value_heads": 8}) == 131_072
    assert m.kv_saving_vs_mha({**BASE, "num_key_value_heads": 8}) == 4


def test_mqa_fp8():
    assert m.kv_bytes_per_token({**BASE, "num_key_value_heads": 1}, bytes_per_elem=1) == 8_192


def test_missing_key_means_mha():
    assert m.kv_bytes_per_token(dict(BASE)) == 524_288
