import numpy as np
import pytest

from s2s.exercise import load_impl

m = load_impl(__file__, "nexttoken")

VOCAB = ["the", "cat", "sat", "mat", "."]
LOGITS = np.array([2.0, 1.0, 0.0, 3.0, -1.0])


def test_greedy():
    assert m.greedy(LOGITS, VOCAB) == "mat"


def test_greedy_tie_takes_first():
    assert m.greedy(np.array([1.0, 5.0, 5.0]), ["a", "b", "c"]) == "b"


def test_top_k_order_and_probs():
    out = m.top_k(LOGITS, VOCAB, 3)
    assert [t for t, _ in out] == ["mat", "the", "cat"]
    e = np.exp(LOGITS - LOGITS.max())
    full = e / e.sum()
    np.testing.assert_allclose([p for _, p in out], [full[3], full[0], full[1]], rtol=1e-9)
    # exp(3)/(exp(3)+exp(2)+exp(1)+exp(0)+exp(-1)) = 0.6364 to 4 places
    assert out[0][1] == pytest.approx(0.6364, abs=1e-4)


def test_probabilities_are_over_the_whole_vocab():
    out = m.top_k(LOGITS, VOCAB, 2)
    assert sum(p for _, p in out) < 1.0


def test_low_temperature_sharpens_high_flattens():
    cold = m.top_k(LOGITS, VOCAB, 1, temperature=0.1)[0][1]
    base = m.top_k(LOGITS, VOCAB, 1, temperature=1.0)[0][1]
    hot = m.top_k(LOGITS, VOCAB, 1, temperature=10.0)[0][1]
    assert cold > base > hot
    assert cold > 0.99
    assert m.top_k(LOGITS, VOCAB, 1, temperature=0.1)[0][0] == "mat"


def test_large_logits_ok():
    out = m.top_k(np.array([1000.0, 1001.0]), ["a", "b"], 2)
    assert out[0][0] == "b"
    assert out[0][1] == pytest.approx(0.7310586, rel=1e-6)


def test_bad_temperature():
    with pytest.raises(ValueError):
        m.top_k(LOGITS, VOCAB, 2, temperature=0)


def test_k_larger_than_vocab():
    assert len(m.top_k(LOGITS, VOCAB, 50)) == len(VOCAB)
