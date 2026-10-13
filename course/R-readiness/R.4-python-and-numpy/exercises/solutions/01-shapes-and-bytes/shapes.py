"""Exercise 1 reference solution."""
import math

import numpy as np

BYTES = {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2, "int64": 8, "int32": 4, "int16": 2, "int8": 1, "fp8": 1}


def num_elements(shape: tuple[int, ...]) -> int:
    return math.prod(shape)          # math.prod(()) == 1


def nbytes(shape: tuple[int, ...], dtype: str) -> int:
    per = BYTES[dtype] if isinstance(dtype, str) and dtype in BYTES else np.dtype(dtype).itemsize
    return num_elements(shape) * per


def matmul_flops(m: int, k: int, n: int) -> int:
    return 2 * m * k * n
