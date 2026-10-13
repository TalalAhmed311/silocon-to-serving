import numpy as np

from s2s.exercise import load_impl

m = load_impl(__file__, "views")


def make():
    return np.arange(12, dtype=np.float32).reshape(3, 4)


def test_row_is_a_view():
    a = make()
    r = m.row(a, 1)
    assert np.array_equal(r, [4, 5, 6, 7])
    assert np.shares_memory(a, r)
    r[0] = -1
    assert a[1, 0] == -1


def test_column_is_a_view_with_a_stride():
    a = make()
    c = m.column(a, 2)
    assert np.array_equal(c, [2, 6, 10])
    assert np.shares_memory(a, c)
    assert c.strides == (16,)             # skips a whole row of 4 float32 = 16 bytes per step
    c[:] = 0
    assert np.all(a[:, 2] == 0)


def test_transposed_is_a_view_and_swaps_strides():
    a = make()
    t = m.transposed(a)
    assert t.shape == (4, 3)
    assert t.strides == (a.strides[1], a.strides[0])
    assert np.shares_memory(a, t)
    assert t[3, 2] == a[2, 3]


def test_every_other_row_is_a_view():
    a = make()
    s = m.every_other_row(a)
    assert np.array_equal(s, a[[0, 2]])
    assert np.shares_memory(a, s)
    assert s.strides[0] == 2 * a.strides[0]


def test_column_copy_is_independent_and_contiguous():
    a = make()
    c = m.column_copy(a, 1)
    assert np.array_equal(c, [1, 5, 9])
    assert not np.shares_memory(a, c)
    assert c.flags["C_CONTIGUOUS"]
    c[:] = 99
    assert a[0, 1] == 1


def test_inputs_are_not_modified():
    a = make()
    before = a.copy()
    for f in (lambda: m.row(a, 0), lambda: m.column(a, 0), lambda: m.transposed(a), lambda: m.every_other_row(a), lambda: m.column_copy(a, 0)):
        f()
    assert np.array_equal(a, before)
