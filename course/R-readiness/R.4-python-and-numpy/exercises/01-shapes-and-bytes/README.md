# Exercise 1 — Shapes and bytes (easy)

Implement in `shapes.py`:

```python
num_elements(shape) -> int
nbytes(shape, dtype) -> int          # dtype: "float32", "bfloat16", "int8", ... or a numpy dtype
matmul_flops(m, k, n) -> int         # 2 * m * k * n
```

Do the arithmetic by hand first: a `4096 x 4096` float32 matrix is 16,777,216 elements x 4 bytes = 67,108,864 bytes = 64 MiB. In bfloat16 it is half of that. The same matrix is the `q_proj` weight of one Llama-3-8B layer.

**Test:** `uv run pytest course/R-readiness/R.4-python-and-numpy/exercises/01-shapes-and-bytes`
