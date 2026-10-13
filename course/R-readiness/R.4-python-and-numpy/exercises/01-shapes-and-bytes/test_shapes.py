import numpy as np

from s2s.exercise import load_impl

m = load_impl(__file__, "shapes")
MiB = 2**20


def test_num_elements():
    assert m.num_elements((3, 4)) == 12
    assert m.num_elements((2, 3, 5)) == 30
    assert m.num_elements(()) == 1          # a scalar has one element
    assert m.num_elements((4, 0, 7)) == 0


def test_llama3_8b_q_proj():
    # q_proj of one Llama-3-8B layer: 4096 x 4096
    assert m.nbytes((4096, 4096), "float32") == 64 * MiB
    assert m.nbytes((4096, 4096), "bfloat16") == 32 * MiB
    assert m.nbytes((4096, 4096), "int8") == 16 * MiB


def test_dtype_sizes():
    assert m.nbytes((10,), "float16") == 20
    assert m.nbytes((10,), "float64") == 80
    assert m.nbytes((5, 2), "int32") == 40


def test_accepts_numpy_dtypes():
    assert m.nbytes((8,), np.float32) == 32
    assert m.nbytes((8,), np.dtype("int16")) == 16


def test_agrees_with_numpy():
    a = np.zeros((7, 5, 3), dtype=np.float32)
    assert m.nbytes(a.shape, a.dtype) == a.nbytes


def test_matmul_flops():
    assert m.matmul_flops(1, 1, 1) == 2
    assert m.matmul_flops(2, 3, 4) == 48
    # one token through q_proj: (1, 4096) @ (4096, 4096)
    assert m.matmul_flops(1, 4096, 4096) == 33_554_432
    assert m.matmul_flops(0, 5, 5) == 0
