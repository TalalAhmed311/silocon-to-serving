# Exercise 1: dot product and matrix-vector product

Implement `dot`, `matvec` and `matvec_flops` in `linalg.py` with plain Python loops (no NumPy inside your functions). A matrix-vector product is one dot product per row. The test compares your loops against NumPy's `@` and checks the FLOP count: 2 per multiply-add, so an (m, n) matrix times a vector costs `2 * m * n`.

```bash
uv run pytest course/R-readiness/R.5-maths-and-ml-basics/exercises/01-dot-and-matvec
```
