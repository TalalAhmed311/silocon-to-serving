import numpy as np

from s2s.exercise import load_impl

m = load_impl(__file__, "ring")
rng = np.random.default_rng(1)


def test_memory_is_bounded():
    r = m.RingKV(8, 2, 32)
    size = r.nbytes()
    for p in range(1000):
        r.write(0, p, rng.standard_normal(32).astype(np.float32), rng.standard_normal(32).astype(np.float32))
    assert r.nbytes() == size == 2 * 2 * 8 * 32 * 4


def test_positions_in_order_after_wrap():
    r = m.RingKV(4, 1, 2)
    for p in range(10):
        r.write(0, p, np.full(2, p, np.float32), np.full(2, -p, np.float32))
    K, V, pos = r.read(0, 9)
    assert list(pos) == [6, 7, 8, 9]
    assert list(K[:, 0]) == [6, 7, 8, 9]


def test_equals_full_attention_inside_window():
    H, KVH, hd, W = 4, 2, 8, 16
    r = m.RingKV(W, 1, KVH * hd)
    Ks, Vs = [], []
    for p in range(10):
        k, v = rng.standard_normal(KVH * hd).astype(np.float32), rng.standard_normal(KVH * hd).astype(np.float32)
        r.write(0, p, k, v); Ks.append(k); Vs.append(v)
    q = rng.standard_normal((H, hd)).astype(np.float32)
    K, V, _ = r.read(0, 9)
    got = m.windowed_attention(q, K, V, H, KVH, hd)
    want = m.windowed_attention(q, np.stack(Ks), np.stack(Vs), H, KVH, hd)
    np.testing.assert_allclose(got, want, atol=1e-6)
