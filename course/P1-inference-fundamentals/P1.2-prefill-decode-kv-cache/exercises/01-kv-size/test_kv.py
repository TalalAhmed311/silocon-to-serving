from s2s.exercise import load_impl

m = load_impl(__file__, "kv")
GiB = 2**30


def test_mha_7b_shape():
    # 32 layers, 32 KV heads, head_dim 128, fp16: 512 KiB per token
    assert m.kv_bytes(32, 32, 128, 1) == 524_288
    assert m.kv_bytes(32, 32, 128, 4096) == 2 * GiB


def test_gqa_8b_shape_fp8():
    assert m.kv_bytes(32, 8, 128, 1, bytes_per_elem=1) == 65_536


def test_capacity():
    assert m.max_tokens(8 * GiB, 32, 8, 128) == 65_536            # 8 GiB / 128 KiB
    assert m.max_sequences(8 * GiB, 2048, 32, 8, 128) == 32
    assert m.max_sequences(8 * GiB, 2048, 32, 8, 128, bytes_per_elem=1) == 64


def test_edges():
    assert m.kv_bytes(32, 8, 128, 0) == 0
    assert m.max_tokens(1000, 32, 8, 128) == 0
