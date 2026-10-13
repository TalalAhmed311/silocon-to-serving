"""Exercise 1 starter."""
import numpy as np

# Bytes per element for the dtype names the course uses. numpy has no bfloat16, so it is listed by hand.
BYTES = {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2, "int64": 8, "int32": 4, "int16": 2, "int8": 1, "fp8": 1}


def num_elements(shape: tuple[int, ...]) -> int:
    """Product of the dimensions. A scalar, shape (), has 1 element. Any 0 dimension gives 0."""
    raise NotImplementedError


def nbytes(shape: tuple[int, ...], dtype: str) -> int:
    """Bytes a dense array of this shape and dtype occupies. `dtype` is a name from BYTES or anything np.dtype accepts."""
    raise NotImplementedError


def matmul_flops(m: int, k: int, n: int) -> int:
    """FLOPs to multiply an (m, k) matrix by a (k, n) matrix. Count a multiply and an add as 2 FLOPs."""
    raise NotImplementedError
