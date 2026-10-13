"""Exercise 1: dot product and matrix-vector product with plain loops."""


def dot(a: list[float], b: list[float]) -> float:
    """Sum of a[i] * b[i]. Raise ValueError if the lengths differ."""
    # TODO: loop over the pairs and add up the products.
    raise NotImplementedError


def matvec(W: list[list[float]], x: list[float]) -> list[float]:
    """W has m rows of n numbers, x has n numbers. Return y with m numbers: y[i] = dot(W[i], x)."""
    # TODO: one dot product per row of W.
    raise NotImplementedError


def matvec_flops(m: int, n: int) -> int:
    """FLOPs for an (m, n) matrix times an n-vector. One multiply and one add count as 2 FLOPs."""
    # TODO: m rows, each with n multiply-adds.
    raise NotImplementedError
