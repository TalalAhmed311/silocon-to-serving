import numpy as np

from s2s.exercise import load_impl

m = load_impl(__file__, "mlp")


def test_silu_values():
    np.testing.assert_allclose(m.silu(np.array([0.0])), [0.0], atol=1e-12)
    # 1 / (1 + e^-1) = 0.7310586
    np.testing.assert_allclose(m.silu(np.array([1.0])), [0.7310586], rtol=1e-6)
    # negative inputs are squashed towards 0 but go slightly below it
    assert m.silu(np.array([-1.0]))[0] < 0
    np.testing.assert_allclose(m.silu(np.array([-1.0])), [-0.2689414], rtol=1e-6)


def test_zero_weights_return_output_bias():
    x = np.array([1.0, 2.0, 3.0])
    W1, b1 = np.zeros((4, 3)), np.zeros(4)
    W2, b2 = np.zeros((2, 4)), np.array([0.5, -0.5])
    np.testing.assert_allclose(m.mlp_forward(x, W1, b1, W2, b2), [0.5, -0.5])


def test_hand_example():
    # 1 input, 1 hidden unit, 1 output: y = 2 * silu(3 * 1 + 0) + 1 = 2 * 2.8577 + 1
    y = m.mlp_forward(np.array([1.0]), np.array([[3.0]]), np.array([0.0]), np.array([[2.0]]), np.array([1.0]))
    np.testing.assert_allclose(y, [2 * 3 / (1 + np.exp(-3)) + 1], rtol=1e-9)
    np.testing.assert_allclose(y, [6.7153], atol=1e-3)


def test_matches_manual_loops():
    rng = np.random.default_rng(3)
    d_in, d_h, d_out = 5, 7, 3
    x = rng.standard_normal(d_in)
    W1, b1 = rng.standard_normal((d_h, d_in)), rng.standard_normal(d_h)
    W2, b2 = rng.standard_normal((d_out, d_h)), rng.standard_normal(d_out)
    h = np.array([sum(W1[i, j] * x[j] for j in range(d_in)) + b1[i] for i in range(d_h)])
    h = h / (1 + np.exp(-h))
    want = np.array([sum(W2[i, j] * h[j] for j in range(d_h)) + b2[i] for i in range(d_out)])
    np.testing.assert_allclose(m.mlp_forward(x, W1, b1, W2, b2), want, rtol=1e-9)


def test_param_count():
    # 3*4 + 4 + 4*2 + 2 = 26
    assert m.param_count(3, 4, 2) == 26
    # 784 inputs, 128 hidden, 10 outputs: 100,352 + 128 + 1,280 + 10
    assert m.param_count(784, 128, 10) == 101_770
