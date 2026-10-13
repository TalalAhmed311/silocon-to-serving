"""Exercise 1 (solution): dot product and matrix-vector product with plain loops."""


def dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("length mismatch")
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def matvec(W: list[list[float]], x: list[float]) -> list[float]:
    return [dot(row, x) for row in W]


def matvec_flops(m: int, n: int) -> int:
    return 2 * m * n
