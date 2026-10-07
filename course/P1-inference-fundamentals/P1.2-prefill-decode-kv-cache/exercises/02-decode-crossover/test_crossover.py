from s2s.exercise import load_impl

m = load_impl(__file__, "crossover")
MODEL = dict(P=8e9, d=4096, L=32, kv_bytes_per_token=131072)
GPU = dict(peak_flops=100e12, bw=1e12)  # made-up round numbers: ridge = 100 FLOP/byte


def test_short_context_crosses_near_ridge():
    B = m.crossover_batch(**MODEL, t=1, **GPU)
    assert B is not None and 95 <= B <= 105   # weights dominate: intensity ≈ B


def test_long_context_never_crosses():
    assert m.crossover_batch(**MODEL, t=32768, **GPU) is None


def test_more_context_needs_bigger_batch():
    b1 = m.crossover_batch(**MODEL, t=64, **GPU)
    b2 = m.crossover_batch(**MODEL, t=256, **GPU)
    assert b1 is not None and (b2 is None or b2 >= b1)
